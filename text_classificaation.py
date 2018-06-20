import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.tokenize import RegexpTokenizer

data = [(list(movie_reviews.words(fileid)),category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(data)
#print(data[1])


all_word = []
for w in movie_reviews.words():
    #all_word.append(w.lower())
    if not w in stop:
        tokenizer = RegexpTokenizer(r'\w+')
        all_word.append(tokenizer.tokenize(w))

#merging multiple list
import itertools
mrg = list(itertools.chain(*all_word))

mrg = nltk.FreqDist(mrg)
#print(mrg.most_common(15))

for i in mrg.most_common(15):
    print(i)