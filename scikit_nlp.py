import nltk
import random
from nltk.corpus import movie_reviews
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

data =  [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(data)

all_word = []
for w in movie_reviews.words():
    all_word.append(w.lower())

all_word = nltk.FreqDist(all_word)

word_feature = list(all_word.keys())[:3000]

def find_feature(doc):
    words = set(doc)
    features = {}
    for w in word_feature:
        features[w] = (w in words)
    return features

#print((find_feature(movie_reviews.words('neg/cv000_29416.txt'))))
test = [(find_feature(rev), category) for (rev,category) in data]

#naive bayes

train_set = test[:1900]
test_set = test[1900:]

cls = nltk.NaiveBayesClassifier.train(train_set)
print('Naive Bayes Algo Accuracy:', (nltk.classify.accuracy(cls,test_set))*100)
#cls.show_most_informative_features(15)

mnb = SklearnClassifier(MultinomialNB())
mnb.train(train_set)
print("MNB algo Accuracy:", (nltk.classify.accuracy(mnb,test_set))*100)


bnb = SklearnClassifier(BernoulliNB())
bnb.train(train_set)
print("bnb:", (nltk.classify.accuracy(bnb,test_set))*100)

lgstc = SklearnClassifier(LogisticRegression())
lgstc.train(train_set)
print('Logistic Regression:', (nltk.classify.accuracy(lgstc,test_set))*100)

lsvc = SklearnClassifier(LinearSVC())
lsvc.train(train_set)
print('LinearSVC:', (nltk.classify.accuracy(lsvc,test_set))*100)

nsvc = SklearnClassifier(NuSVC())
nsvc.train(train_set)
print('NuSVC:', (nltk.classify.accuracy(nsvc,test_set))*100)