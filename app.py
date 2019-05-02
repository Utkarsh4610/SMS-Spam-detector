# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 23:59:17 2019

@author: Utkarsh Kumar
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:26:03 2019

@author: Utkarsh Kumar
"""
import nltk
nltk.download('maxent_treebank_pos_tagger')
from flask import Flask,render_template,request
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
import pickle

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html',template_folder='templates')

@app.route('/predict',methods=['GET','POST'])
def predict():
    p_model = open('model.ml','rb')
    model = joblib.load(p_model)
    with open('aw.pickle', 'rb') as handle:
        alw = pickle.load(handle)
    def find_features(message):
        words = word_tokenize(message)
        features = {}
        for word in alw:
            features[word] = (word in words)
        return features
    
    if request.method == 'POST':
        text = request.form['text']
        processed = re.sub(r'^.+@[^\.].*\.[a-z]{2,}$','emailaddress',text)
        processed = re.sub(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','webaddress',processed)
        processed = re.sub(r'£|\$', 'moneysymb',processed)
        processed = re.sub(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','phonenumbr',processed)
        processed = re.sub(r'\d+(\.\d+)?', 'numbr',processed)
        processed = re.sub(r'[^\w\d\s]', ' ',processed)
        processed = re.sub(r'\s+', ' ',processed)
        processed = re.sub(r'^\s+|\s+?$', '',processed)
        processed = processed.lower()
        word = word_tokenize(processed)
        stop_words = {'a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves'}
        filtered_sentence = []
        filtered_sentence = [w for w in word if not w in stop_words]
        st = []
        ps= PorterStemmer()
        for w in filtered_sentence:
            st.append(ps.stem(w))
        stx=' '.join(st)
        featuresx = find_features(stx)
        prediction = model.classify(featuresx)

    return render_template('result.html',template_folder='templates',prediction = prediction)

if __name__ == '__main__':
    app.run(debug=False)

