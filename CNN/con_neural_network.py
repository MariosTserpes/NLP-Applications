import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.models import Model

max_vocab_size = 20000
def CNN_sentiment(input_file):
    global data
    data = pd.read_csv(f"{input_file}", encoding = "ISO-8859-1")
    data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
    data.columns = ["labels", "data"]
    data["labels_2"] = data["labels"].map({"ham": 0, "spam": 1})
    y = data["labels_2"].values
    X = data["data"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
    tokenizer = Tokenizer(num_words = max_vocab_size)
    tokenizer.fit_on_texts(X_train)
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)
    word2index = tokenizer.word_index
    print(f"Total number of unique tokens are: {len(word2index)}")
    data_train = pad_sequences(sequences_train)
    print(f"Shape of data train tensor: {data_train.shape}")
    T = data_train.shape[1]
    data_test = pad_sequences(sequences_test, maxlen = T)
    print(f"Shape of data test tensor: {data_test.shape}")
    D = 20 #Embedding dimensionality
    i = Input(shape = (T, )) #imput layer
    x = Embedding(len(word2index) + 1, D)(i) #embedding layer
    #second layer
    x = Conv1D(32, 3, activation = "relu")(x)
    x = MaxPooling1D(3)(x)
    #third layer
    x = Conv1D(128, 3, activation = "relu")(x)
    x = GlobalMaxPooling1D()(x)
    #Dense layer
    x = Dense(1, activation = "sigmoid")(x)
    #Compile Model
    model = Model(i, x)
    model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    r = model.fit(x = data_train, y = y_train, epochs = 5, validation_data =(data_test, y_test))
    
    plt.plot(r.history["loss"], label = "LOSS")
    plt.plot(r.history["val_loss"], label = "VALIDATION_LOSS")
    plt.legend()
    plt.show()
    
    plt.plot(r.history["accuracy"], label = "ACCURACY")
    plt.plot(r.history["val_accuracy"], label = "VALIDATION_ACCURACY")
    plt.legend()
    plt.show();    
