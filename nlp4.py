import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle

review = pd.read_csv('reviewsnlp.csv')
review = review.rename(columns = {'text': 'review'}, inplace = False)
review.head()
X = review.review
y = review.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)
vector = CountVectorizer(stop_words = 'english',lowercase=False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()
X_test_transformed = vector.transform(X_test)
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)
print(classification_report(naivebayes.predict(X_test_transformed), y_test))
review1 = ['''privacy at least put some option appear offline. i mean for some people like me it's a big pressure to be seen online like you need to response on every message or else you be called seenzone only. if only i wanna do on facebook is to read on my newsfeed and just wanna response on message i want to. pls reconsidered my review. i tried to turn off chat but still can see me as online.''']
vec = vector.transform(review1).toarray()
print('Headline:', review1)
print(str(list(naivebayes.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE'))
saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)
review2 = ['The app is really good']
vec = vector.transform(review2).toarray()

s.predict(vec)[0]