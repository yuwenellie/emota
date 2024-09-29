from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize
from deep_translator import GoogleTranslator
import speech_recognition as sr
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()
user_data = {}
# This api is free btw
genai.configure(api_key="AIzaSyC1b1w5om9odJeDpqzcHzqSFrMFlz1Cc3c")


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text')
    dest_language = data.get('language', 'en')
    if text:
        translated = GoogleTranslator(source='auto', target=dest_language).translate(text)
        user_data['last_text'] = text
        return jsonify({
            'original_text': text,
            'translated_text': translated,
            'language': dest_language,
            'duration': data.get('duration', 'N/A')
        })


@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    audio_file = request.files.get('audio')
    if audio_file:
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        user_data['last_text'] = text
        return jsonify({'transcribed_text': text})


@app.route('/get-user-text', methods=['GET'])
def get_user_text():
    return jsonify({'last_text': user_data.get('last_text', 'No text available')})


@app.route('/analyze-text', methods=['POST'])
def analyze_last_text():
    text = user_data.get('last_text', '')
    if text:
        return jsonify(analyze_sentiment(text))


def analyze_sentiment(text):
    sentences = sent_tokenize(text)
    results = {}
    for sentence in sentences:
        sentiment_dict = analyzer.polarity_scores(sentence)
        compound_score = sentiment_dict['compound']
        sentiment_emoji = get_sentiment_emoji(compound_score)
        results[sentence] = {'scores': sentiment_dict, 'emoji': sentiment_emoji}
    return results


def get_sentiment_emoji(sentiment):
    emoji_mapping = {
        "negative": "😞",
        "neutral": "😐",
        "positive": "😄",
    }
    if sentiment < -0.05:
        return emoji_mapping["negative"]
    elif sentiment > 0.05:
        return emoji_mapping["positive"]
    else:
        return emoji_mapping["neutral"]


@app.route('/generate-story', methods=['POST'])
def generate_story():
    last_text = user_data.get('last_text', '')
    sentiment_results = analyze_sentiment(last_text)
    emojis = [details['emoji'] for sentence, details in sentiment_results.items()]
    num_lines = len(emojis)
    story = create_story_with_genai(emojis, num_lines)
    artist_name = story.split('\n')[0].strip()
    image_urls = scrape_artist_images(artist_name)
    return jsonify({'story': story, 'images': image_urls})


def create_story_with_genai(emojis, num_lines):
    prompt = "Suggest one artist associated with this emoji: " + ", ".join(emojis)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text


@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment_results = {}
    if request.method == 'POST':
        user_input = request.form['text']
        sentiment_results = analyze_sentiment(user_input)
    return render_template('index.html', results=sentiment_results)


def scrape_artist_images(artist_name):
    query = f"{artist_name} painting"
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={requests.utils.quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    image_elements = soup.find_all('img', limit=4)
    image_urls = [img['src'] for img in image_elements if 'src' in img.attrs]
    return image_urls


if __name__ == '__main__':
    app.run(debug=True)
