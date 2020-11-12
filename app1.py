from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


## Definitions
def remove_pattern(input_txt,pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100


app = Flask(__name__)


data = pd.read_csv("sentiment.tsv",sep = '\t')
data.columns = ["sentiment","text"]
# Features and Labels
data['sentiment'] = data['sentiment'].map({'pos': 0, 'neg': 1})
data['cleanText'] = np.vectorize(remove_pattern)(data['text'],"@[\w]*")
tokenized_tweet = data['cleanText'].apply(lambda x: x.split())
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
data['cleanText'] = tokenized_tweet
data['textlength'] = data['text'].apply(lambda x:len(x) - x.count(" "))
data['punct_rate'] = data['text'].apply(lambda x:count_punct(x))
X = data['cleanText']
y = data['sentiment']
# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data
X = pd.concat([data['textlength'],data['punct_rate'],pd.DataFrame(X.toarray())],axis = 1)
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
## Using Classifier
clf = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                   warm_start=False)
clf.fit(X,y)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = pd.DataFrame(cv.transform(data).toarray())
        textlength= pd.DataFrame([len(data) - data.count(" ")])
        punct = pd.DataFrame([count_punct(data)])
        total_data = pd.concat([textlength,punct,vect],axis = 1)
        my_prediction = clf.predict(total_data)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
