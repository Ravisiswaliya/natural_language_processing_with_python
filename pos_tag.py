import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk import pos_tag

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

cst = PunktSentenceTokenizer(train_text)
toknz = cst.tokenize(sample_text)

for c in toknz:
    tok = nltk.word_tokenize(c)
    tag = pos_tag(tok)
    print(tag)


