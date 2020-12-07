# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import sklearn
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

model_path = 'review_sentiment.h5'

model = load_model(model_path)

@app.route('/', methods = ['GET'])
def home():
    return render_template("home.html")

@app.route('/predict', methods = ['GET', 'POST'])

def predict():
    if request.method == 'POST':
        text = request.form['text']
        
        corpus = []
        vocab_size = 5000
        review_max_length = 400
        
        text = re.sub('[^a-zA-z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [word for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
        corpus.append(text)
        
        onehot_repr = [one_hot(word, vocab_size) for word in corpus]
        
        embedded_docs = pad_sequences(sequences= onehot_repr, maxlen= review_max_length, dtype='int32', padding='pre')
        
        X = np.array(embedded_docs).astype(np.float32)
        
        class1 = model.predict_classes(X)
        
        if class1 == 1:
            result = 'positive'
        elif class1 == 0:
            result = 'negative'
        
    return render_template("home.html", prediction_text = "The review is {}".format(result))

if __name__ == "__main__":
    app.run()

        
        
        
        
        

