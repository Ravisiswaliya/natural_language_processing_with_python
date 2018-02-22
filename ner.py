import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import state_union

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

cst = PunktSentenceTokenizer(train_text)
tknd = cst.tokenize(sample_text)

try:
    for i in tknd[5:]:
        word = nltk.word_tokenize(i)
        # print(word)
        tgd = nltk.pos_tag(word)
        nER = nltk.ne_chunk(tgd)
        print(nER)

except Exception as e:
    print(str(e))

