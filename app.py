from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report 
import os

nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

# Load dataset 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "data", "spam2.csv")

chunksize = 10000
df_iter = pd.read_csv(csv_path, chunksize=chunksize)
df = pd.concat(df_iter)
df = df[['text', 'label_num']]
print(df.shape)

# Preprocessing
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text):
        text = ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    processed_words = [stemmer.stem(word) for word in words]
    return " ".join(processed_words)

df['processed_text'] = df['text'].apply(preprocess_text)

# Vectorization (keep sparse)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label_num']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Multinomial Naive Bayes
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

@app.route("/")
def home():
    return " Email Spam Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = classifier.predict(vectorized)[0]

    return jsonify({
        "message": message,
        "prediction": "Spam" if prediction == 1 else "Ham"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

