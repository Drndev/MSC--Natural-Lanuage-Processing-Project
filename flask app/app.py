from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from flair.models import TextClassifier
from flair.data import Sentence


app = Flask(__name__)
#model = pickle.load(open('SVM_spam_model.pkl','rb'))
@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		#classifier = TextClassifier.load("best-model.pt")
		#message = request.form['message']
		#sentence=Sentence(message)
		#done = classifier.predict(sentence)



		classifier = TextClassifier.load("best-model.pt")
		message = request.form['message']
		sentence = Sentence(message)
		classifier.predict(sentence)
		print('Sentence: ', sentence.labels)
		label = sentence.labels[0]
    	#response = {'result': label.value, 'polarity':label.score}
		

	return render_template('result.html',prediction = label)



if __name__ == '__main__':
	app.run(debug=True)