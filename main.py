import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st


word_ind=imdb.get_word_index()
reverse_word_ind={value: key for key, value in word_ind.items()}

model= load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_ind.get(i-3,'?') for i in encoded_review])

def preprocess_text(text):
    words=text.lower().split()
    
    encoded_review =[word_ind.get(word,2)+ 3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    pred=model.predict(preprocess_input)
    sentiment = 'Positive' if pred[0][0]>0.5 else 'Negative'
    return sentiment,pred[0][0]

user_inp=st.text_area('Movie Review')

if st.button("Classify"):
    preprocess_input=preprocess_text(user_inp)
    
    prediction=model.predict(preprocess_input)
    sentiments='Positive' if prediction[0][0] >0.5 else "Negative"
    
    
    st.write(f'Sentiment:{sentiments}')
    st.write(f'Prediction Score: {prediction[0][0 ]}')