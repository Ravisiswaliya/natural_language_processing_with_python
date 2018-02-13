from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

#define a sentence
sentance = 'I am a python developer.'

#all stop_word of english language
stop_word =  set(stopwords.words('english'))
#print(stop_word)

#tokenizing sentence
tokenize = word_tokenize(sentance)
print(tokenize)

#removing stop words
result = [wrd for wrd in tokenize if not wrd in stop_word]
print(result)
