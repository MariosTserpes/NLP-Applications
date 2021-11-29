import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
import re

from nltk.corpus import stopwords
nltk.download("stopwords")

from nltk.stem.porter import PorterStemmer
#stopwords.words("english")

dataset = pd.read_csv("C:\marios\datasets\Restaurant_Reviews.tsv", sep = "\t", quoting = 3)

def cleaning_text(input_df, column = None):
    global corpus
    corpus = []
    porter_stemmer = PorterStemmer()
    
    for i in range(len(dataset)):
        review = re.sub("[^a-zA-Z]", " ", input_df[column][i])
        review = review.lower()
        review = review.split()
        review = [porter_stemmer.stem(word) for word in review if word not in stopwords.words("english")]
        review = " ".join(review)
        
        corpus.append(review)

cleaning_text(dataset, column = "Review")
