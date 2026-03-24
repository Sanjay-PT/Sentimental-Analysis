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
st.set_page_config(page_title="Advanced Tweet Sentiment Analyzer", layout="centered")

st.title("🐦 Advanced Tweet Sentiment Analysis")
st.write("Now with Model Comparison & History Tracking 🚀")
data = pd.read_csv("tweets.csv")
# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("twitter_sentiment.csv")

try:
    data = load_data()
except:
    st.error("Dataset file 'twitter_sentiment.csv' not found!")
    st.stop()

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['clean_text'] = data['text'].apply(clean_text)

# -----------------------------
# Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Models
# -----------------------------
log_model = LogisticRegression()
nb_model = MultinomialNB()

log_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)

log_acc = accuracy_score(y_test, log_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

# -----------------------------
# Sidebar Model Info
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
    "Choose Model for Prediction",
    ["Logistic Regression", "Naive Bayes"]
)

if model_choice == "Logistic Regression":
    model = log_model
else:
    model = nb_model

# -----------------------------
# Prediction History Storage
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
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector_input = vectorizer.transform([cleaned])

        prediction = model.predict(vector_input)[0]
        probabilities = model.predict_proba(vector_input)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        # Emoji
        if prediction.lower() == "positive":
            emoji = "😊"
        elif prediction.lower() == "negative":
            emoji = "😞"
        else:
            emoji = "😐"

        st.subheader(f"Prediction: {prediction} {emoji}")
        st.write(f"Confidence: {confidence}%")

        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(probabilities, labels=model.classes_, autopct='%1.1f%%')
        ax.set_title("Sentiment Probability Distribution")
        st.pyplot(fig)

        # Save to history
        st.session_state.history.append({
            "Tweet": user_input,
            "Prediction": prediction,
            "Confidence (%)": confidence,
            "Model Used": model_choice
        })

# -----------------------------
# Display History
# -----------------------------
if st.session_state.history:
    st.subheader("📜 Prediction History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)

    # Download Button
    csv = history_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇ Download History as CSV",
        csv,
        "sentiment_history.csv",
        "text/csv"
    )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("Enhanced Version with Model Comparison & History Tracking 🚀")
