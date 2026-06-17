import nltk
nltk.download('punkt')
nltk.download('stopwords')
import nltk
nltk.download('punkt')
nltk.download('stopwords')import json
import string
import nltk
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# NLTK downloads (safe mode)
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Load FAQ data
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

questions = list(faqs.keys())
answers = list(faqs.values())

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Preprocess FAQs
processed_questions = [preprocess(q) for q in questions]

# TF-IDF model
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)

def get_answer(user_input):
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    index = similarity.argmax()
    score = similarity[0][index]

    if score < 0.3:
        return "Sorry, I don't understand your question."

    return answers[index]

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chat route
@app.route("/get")
def chat():
    user_message = request.args.get("msg")
    response = get_answer(user_message)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
    import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)