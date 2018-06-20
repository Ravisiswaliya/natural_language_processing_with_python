import nltk
import random
from nltk.corpus import movie_reviews
import pickle


data = [(list(movie_reviews.words(fileid)),category)
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


#cls = nltk.NaiveBayesClassifier.train(train_set)

#opening a pickle file
clf = open('data.pickle', 'rb')
cls = pickle.load(clf)
clf.close()
print((nltk.classify.accuracy(cls,test_set))*100)

cls.show_most_informative_features(15)

#Creating a pickle file
'''
sv_clf = open('data.pickle', 'wb')
pickle.dump(cls, sv_clf)
sv_clf.close()

'''

