import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model.load_model('next_word_lstm_model.h5')

# load the tokenizer
with open('tokenizer_hamlet.pickle', 'rb') as handle:
    Tokenizerizer = pickle.load(handle)
  

def predict_next_word(model, tokenizer, text_sequence, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text_sequence])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


# streamlit app

st.title("Next Word Prediction using LSTM")
user_input = st.text_area("Enter your text here:")
if st.button("Predict Next Word"):
    max_seq = model.input_shape[1]+1
    next_word = predict_next_word(model, Tokenizerizer, user_input, max_seq)
    st.write(f"The predicted next word is: {next_word}")