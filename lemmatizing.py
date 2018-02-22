from nltk.stem import WordNetLemmatizer

lmatzr = WordNetLemmatizer()

print(lmatzr.lemmatize('greatest'))
print(lmatzr.lemmatize('greatest', pos='a'))
print(lmatzr.lemmatize('greater',pos='a'))
print(lmatzr.lemmatize('simpler',pos='a'))
print(lmatzr.lemmatize('simpler'))


