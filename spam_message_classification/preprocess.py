import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocessing(input_file):
    global raw_df, ham, spam, data
    '''
    Struggling with inbalanced class
    '''
    raw_df = pd.read_csv(f"{input_file}", sep = "\t")
    
    ham = raw_df[raw_df["label"] == "ham"]
    spam = raw_df[raw_df["label"] == "spam"]
    ham = ham.sample(spam.shape[0])
    
    data = ham.append(spam, ignore_index = True)
    print("Distribution of each class:\n {}".format(data["label"].value_counts()))
    
    
    plt.hist(data[data["label"] == "ham"]["length"], bins = 40, alpha = 0.9, color = "red", edgecolor = "black")
    plt.hist(data[data["label"] == "spam"]["length"], bins = 40, alpha = 0.9, color = "purple", edgecolor = "black")
    plt.title("HAM-Red | SPAM-Purple")
    plt.show();

preprocessing("C:\marios\datasets\spam.tsv")
