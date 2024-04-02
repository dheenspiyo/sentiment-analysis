import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
def preprocess_text(text):
    # Remove mentions (e.g., @username)
    text = re.sub(r'@\w+', '', text)

    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map symbols
        "\U0001F700-\U0001F77F"  # Alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric shapes
        "\U0001F800-\U0001F8FF"  # Miscellaneous Symbols and Arrows
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Extended-A
        "\U0001FA70-\U0001FAFF"  # Extended-B
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+"
    )
    text = emoji_pattern.sub(r'', text)

    # Remove other special characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    return text.strip()

# Load the model
model = load_model('sentiment.h5')
import pickle

# Load tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# Function to predict sentiment
def predict_sentiment(comment):
    comment = preprocess_text(comment)
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=70, padding='pre')
    sentiment_probabilities = model.predict(padded_sequence)
    sentiment = np.argmax(sentiment_probabilities)
    return sentiment

# Title of the app
st.title('Twitter Sentiment Analysis')

# Text input for user to enter a comment
comment = st.text_input('Enter your comment:')

# Button to trigger sentiment prediction
if st.button('Predict Sentiment'):
    # Predict sentiment
    sentiment = predict_sentiment(comment)
    if sentiment == 0:
        st.write('Sentiment: Negative')
    elif sentiment == 1:
        st.write('Sentiment: Neutral')
    else:
        st.write('Sentiment: Positive')
