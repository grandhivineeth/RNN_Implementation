# importing libraries 

import numpy as np 
import pandas as pd
import tensorflow as tf 
import streamlit as st
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## Decoding the mapping

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the model 
model = load_model('simplernn_model.h5', compile=False)


# Function to decode review 
def decode_review(encoded_review): 
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review]) 

# function to preprocess the user input 
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


# prediction function 
def predict_sentiment(review): 
    preprocess_txt = preprocess_text(review)
    prediction = model.predict(preprocess_txt)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    return sentiment, prediction[0][0]


# Designing the streamlit app 
st.title('IMDb Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify if it\'s a Positive or Negative review')

# User nut 
user_input = st.text_area('Movie Review')

if st.button('Classify'): 
    preprocess_input = preprocess_text(user_input)
    prediction = model.predict(preprocess_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else 'Negative'

    # Display result 
    st.write('**Prediction: **', sentiment)
    st.write('**Prediction Score: **', round(prediction[0][0], 3))
else: 
    st.write('Please enter a movie review.')