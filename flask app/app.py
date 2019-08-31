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
 #Corpus = pd.read_csv(r"Hate_data_S.csv",encoding='latin-1')

	# Step - a : Remove blank rows if any.
# Step - a : Remove blank rows if any.
 #Corpus['tweet'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
 #Corpus['tweet'] = [entry.lower() for entry in Corpus['tweet']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
 #Corpus['tweet']= [word_tokenize(entry) for entry in Corpus['tweet']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
 #tag_map = defaultdict(lambda : wn.NOUN)
 #tag_map['J'] = wn.ADJ
 #tag_map['V'] = wn.VERB
 #tag_map['R'] = wn.ADV
 #for index,entry in enumerate(Corpus['tweet']):
    # Declaring Empty List to store the words that follow the rules for this step
    #Final_words =[]
    # Initializing WordNetLemmatizer()
    #word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    #for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        #if word not in stopwords.words('english') and word.isalpha():
            #word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            #Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'              
 #Corpus.loc[index,'text_final'] = str(Final_words)
 #Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['class'],test_size=0.3)

 #Encoder = LabelEncoder()
 #Train_Y = Encoder.fit_transform(Train_Y)
 #Test_Y = Encoder.fit_transform(Test_Y)

 #Tfidf_vect = TfidfVectorizer(max_features=10000)
 #Tfidf_vect.fit(Corpus['text_final'])
 #Train_X_Tfidf = Tfidf_vect.transform(Train_X)
 #Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# fit the training dataset on the NB classifier
 #clf = naive_bayes.MultinomialNB()
 #clf.fit(Train_X_Tfidf,Train_Y)
 #clf.score(Test_X_Tfidf,Test_Y)
# predict the labels on validation dataset
	#predictions_NB = clf.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
#print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)






    # The final processed set of words for each iteration will be stored in 'text_final'
  
#	df= pd.read_csv("spam.csv", encoding="latin-1")
#	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
#	# Features and Labels
#	df['label'] = df['class'].map({'ham': 0, 'spam': 1})
#	X = df['message']
#	y = df['label']
#	
#	# Extract Feature With CountVectorizer
	#cv = CountVectorizer()
##	X = cv.fit_transform(X) # Fit the Data
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	#Naive Bayes Classifier
#	from sklearn.naive_bayes import MultinomialNB
#
	#clf = MultinomialNB()
	#clf.fit(X_train,y_train)
	#clf.score(X_test,y_test)

#Alternative Usage of Saved Model
	#joblib.dump(clf, 'svm_model.SAV')
	#from flair.models import TextClassifier
	#from flair.data import Sentence
	#classifier = TextClassifier.load("best-model.pt")
	#sentence = Sentence('')
	#classifier.predict(sentence)
	#print(sentence.labels)

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