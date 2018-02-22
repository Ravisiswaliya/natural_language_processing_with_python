import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_data = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

token = PunktSentenceTokenizer(train_data)

tokened = token.tokenize(sample_text)

for i in tokened:
    words = nltk.word_tokenize(i)
    tagged = nltk.pos_tag(words)
    # print(tagged)

    cG = r"""Chunk:{<.*>+}
            }<VB.?|IN|NNP|DT|TO>+{
            """
    cP = nltk.RegexpParser(cG)
    cD = cP.parse(tagged)
    print(cD)
    #cD.draw()



