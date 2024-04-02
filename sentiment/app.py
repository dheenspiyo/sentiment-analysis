import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow
import pandas as pd
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle

model=load_model('sentiment.h5')
# Load tokenizer from file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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
# Function to predict sentiment
def predict_sentiment(comment):
    comment = preprocess_text(comment)
    sequence = tokenizer.texts_to_sequences([comment])
    padded_sequence = pad_sequences(sequence, maxlen=70, padding='pre')
    sentiment_probabilities = model.predict(padded_sequence)
    sentiment = np.argmax(sentiment_probabilities)
    return sentiment

# Create a Streamlit app interface



def home():
    st.subheader("Home")
    st.write("Welcome to the Sentiment Analysis App")
    imageha = mpimg.imread('img3.jpeg')     
    st.image(imageha)
    st.write('By using LSTM Model to predict  Sentiment in comments more Accurately')
    st.header('About Dataset')
    st.write("Given a message and an entity, the task is to judge the sentiment of the message about the entity. There are three classes in this dataset: Positive, Negative and Neutral. We regard messages that are not relevant to the entity (i.e. Irrelevant) as Neutral.")

    
    st.header('Sentiment')
    col1, col2,col3 = st.columns(3)

    
    col1.header(" Happy 😊")
   


    col2.header("Sad 😭")
    
    
    col3.header("Neutral 😐 ")
    
    
   
    


def prediction():
    
    st.subheader("Sentiment Prediction")

    imagehb = mpimg.imread('imags.jpg')     
    st.image(imagehb)
    # Upload audio file
    comment = st.text_input('Enter your comment:')
    if st.button('Predict Sentiment'):
        # Predict sentiment
        sentiment = predict_sentiment(comment)
        if sentiment == 0:
            st.write('Sentiment: Negative')
        elif sentiment == 1:
            st.write('Sentiment: Neutral')
        else:
            st.write('Sentiment: Positive')

       

    



def main():
    st.set_page_config(layout="wide")
    st.title("Sentiment Analysis App")
# Create the tab layout
    tabs = ["Home", "Prediction"]
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    
   
main()
