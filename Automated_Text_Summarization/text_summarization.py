import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

from heapq import nlargest

text = """Argentina's Lionel Messi won the Ballon d'Or award for the best player in the world for a record-stretching seventh time on Monday (Nov 29), beating Robert Lewandowski and Jorginho to lift football's most prestigious trophy yet again.
The forward added to his 2009, 2010, 2011, 2012, 2015 and 2019 trophies after winning the Copa America for the first time with his country in July.Messi, who joined Paris St-Germain on a free transfer from Barcelona during the close season after 
finishing as La Liga's top scorer with the Spanish club, collected 613 points, with Bayern Munich's Lewandowski, named best striker on Monday, getting 580.Jorginho, who won the Champions League with Chelsea and the European Championship with Italy, 
ended up third on 460, ahead of France's Karim Benzema and Ngolo Kante in fourth and fifth places respectively."""

def process_text(text_to_process = None):
    global document, tokens, punctuation
    nlp = spacy.load("en_core_web_sm")
    document = nlp(text_to_process)
    
    tokens = [token.text for token in document]
    punctuation = punctuation + "\n"
    
def cleaning_text(texts):
    global word_freq
    word_freq = {}
    stop_words = list(STOP_WORDS)
    
    for word in document:
        if word.texts.lower() not in stop_words:
            if word.texts.lower() not in punctuation:
                if word.texts not in word_freq.keys():
                    word_freq[word.texts] = 1
                else:
                    word_freq[word.texts] += 1
                    
    max_freq = max(word_freq.values())
    for word in word_freq.keys():
        word_freq[word] = word_freq[word] / max_freq
    return word_freq
  
def sentence_tokenization(texts):
    global sent_score, sent_tokens
    
    sent_score = {}
    sent_tokens = [sent for sent in document.sents]
    for sentence in sent_tokens:
        for word in sentence:
            if word.texts.lower() in word_freq.keys():
                if sentence not in sent_score.keys():
                    sent_score[sentence] = word_freq[word.texts.lower()]
                else:
                    sent_score[sentence] += word_freq[word.texts.lower()]
