import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
simple_text = state_union.raw('2006-GWBush.txt')

cst = PunktSentenceTokenizer(train_text)

tok = nltk.sent_tokenize(simple_text)
#print(tok)

for t in tok:
    print(t)