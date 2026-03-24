import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="centered")

st.title("🐦 Tweet Sentiment Analysis App")
st.write("Analyze tweet sentiment using Machine Learning")

# -----------------------------
# Load Dataset (GitHub friendly)
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("twitter_sentiment_small.csv")

try:
    data = load_data()
except:
    st.error("Dataset file 'twitter_sentiment_small.csv' not found!")
    st.stop()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['clean_text'] = data['text'].apply(clean_text)

# -----------------------------
# Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
log_model = LogisticRegression(max_iter=2000, class_weight='balanced')
nb_model = MultinomialNB()

log_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

# -----------------------------
# Sidebar Model Comparison
# -----------------------------
st.sidebar.header("📊 Model Comparison")

accuracy_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy (%)": [round(log_acc*100,2), round(nb_acc*100,2)]
})

st.sidebar.table(accuracy_table)

# -----------------------------
# Model Selection
# -----------------------------
model_choice = st.selectbox(
    "Choose Model",
    ["Logistic Regression", "Naive Bayes"]
)

model = log_model if model_choice == "Logistic Regression" else nb_model

# -----------------------------
# Session History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# User Input
# -----------------------------
st.subheader("Enter a Tweet")
user_input = st.text_area("Type your tweet here...")

if st.button("Analyze Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a tweet")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]
        confidence = round(np.max(probabilities)*100, 2)

        # Emoji
        emoji = "😊" if prediction.lower()=="positive" else "😞" if prediction.lower()=="negative" else "😐"

        st.subheader(f"Prediction: {prediction} {emoji}")
        st.write(f"Confidence: {confidence}%")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(probabilities, labels=model.classes_, autopct='%1.1f%%')
        ax.set_title("Probability Distribution")
        st.pyplot(fig)

        # Save history
        st.session_state.history.append({
            "Tweet": user_input,
            "Prediction": prediction,
            "Confidence (%)": confidence,
            "Model": model_choice
        })

# -----------------------------
# Show History
# -----------------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "history.csv", "text/csv")

st.markdown("---")
st.write("Deployed using Streamlit 🚀")
