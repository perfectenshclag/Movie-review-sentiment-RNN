import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the word index and reverse word index
word_ind = imdb.get_word_index()
reverse_word_ind = {value: key for key, value in word_ind.items()}

# Load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Function to decode review to human-readable text
def decode_review(encoded_review):
    return ' '.join([reverse_word_ind.get(i - 3, '?') for i in encoded_review])

# Function to preprocess the user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_ind.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Function to predict sentiment
def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    pred = model.predict(preprocess_input)
    sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
    return sentiment, pred[0][0]

# Streamlit app UI
st.title("Movie Review Sentiment Classifier 🎬")
st.subheader("Enter a movie review to classify its sentiment:")
st.write("This model predicts whether a movie review is **positive** or **negative** based on the text you enter.")

# Text input from user
user_inp = st.text_area('Type your review here...', height=150)

# Classify button
if st.button("Classify Review"):
    # Display loading spinner while processing
    with st.spinner('Analyzing sentiment...'):
        preprocess_input = preprocess_text(user_inp)
        prediction = model.predict(preprocess_input)
        sentiments = 'Positive' if prediction[0][0] > 0.5 else "Negative"
        score = prediction[0][0]
        
    # Display sentiment
    if sentiments == 'Positive':
        st.success(f'Sentiment: {sentiments}')
    else:
        st.error(f'Sentiment: {sentiments}')

    # Display prediction score as progress bar and formatted score
    st.write(f"Prediction Confidence: **{score:.2f}**")
    st.progress(score)  # Display a progress bar with confidence score

# Add footer
st.markdown("<hr>", unsafe_allow_html=True)
st.write("**Note:** This classifier is trained on the IMDB dataset and might not generalize to all types of reviews.")
