import json
import string
import os
import nltk
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# ==============================
# NLTK SAFE DOWNLOAD (IMPORTANT)
# ==============================
try:
    nltk.data.find('tokenizers/punkt_tab')
except:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ==============================
# LOAD FAQ DATA
# ==============================
with open("faq_data.json", "r") as file:
    faqs = json.load(file)

questions = list(faqs.keys())
answers = list(faqs.values())

# ==============================
# TEXT PREPROCESSING
# ==============================
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# ==============================
# TF-IDF MODEL
# ==============================
processed_questions = [preprocess(q) for q in questions]

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

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def chat():
    user_message = request.args.get("msg")
    response = get_answer(user_message)
    return jsonify({"answer": response})

# ==============================
# RUN (RAILWAY READY)
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)