import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
# from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
data = pd.read_csv("Twitter_Data.csv")
model = load_model("model.h5")
tokenizer = None  # Define the tokenizer used during training

def predict_sentiment(sentence):
    global tokenizer, model
    # Tokenize the sentence
    sentence_seq = tokenizer.texts_to_sequences([sentence])

    # Pad the sequence
    sentence_seq = pad_sequences(sentence_seq, maxlen=tweets.shape[1])

    # Predict the sentiment
    prediction = model.predict(sentence_seq)

    # Get the predicted label
    labels = ["negative", "neutral", "positive"]
    predicted_label = labels[np.argmax(prediction)]

    return predicted_label

def perform_sentiment_analysis():
    sentence = entry.get()

    if sentence:
        predicted_sentiment = predict_sentiment(sentence)
        messagebox.showinfo("Sentiment Prediction", f"The predicted sentiment is: {predicted_sentiment}")
    else:
        messagebox.showwarning("Input Error", "Please enter a sentence.")

# Read the data
data = pd.read_csv("Twitter_Data.csv")

# Preprocess the data
data = data.sample(frac=1).reset_index(drop=True)
labels = pd.get_dummies(data.category)
labels.columns = ["negative", "neutral", "positive"]
data = data.drop(columns="category")

# Check for missing values in the "clean_text" column
missing_values = data["clean_text"].isnull().sum()

if missing_values > 0:
    # Handle missing values (e.g., replace with an empty string)
    data["clean_text"].fillna("", inplace=True)

# Tokenize the text data
tokenizer = Tokenizer(num_words=8150, lower=True, split=" ", oov_token="~")
tokenizer.fit_on_texts(data["clean_text"])
word_index = tokenizer.word_index
data["clean_text"] = tokenizer.texts_to_sequences(data["clean_text"])

# Pad the sequences
tweets = pad_sequences(data["clean_text"])

# Create the main window
window = tk.Tk()
window.title("Sentiment Analysis")
window.geometry("800x400")

# Create a label and an entry for sentence input
label = tk.Label(window, text="Enter a sentence:")
label.pack()
entry = tk.Entry(window, width=40)
entry.pack()

# Create a button to trigger sentiment prediction
button = tk.Button(window, text="Predict Sentiment", command=perform_sentiment_analysis)
button.pack()

# Run the main event loop
window.mainloop()