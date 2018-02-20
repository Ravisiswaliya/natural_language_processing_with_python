from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

example_words = ['learning', 'learn', 'learned']

#for w in example_words:
#    print(ps.stem(w))


text = "I am learning python that's why i like to call myself pythonist. Or you can say i am pythoning"

word = word_tokenize(text)

for t in word:
    print(ps.stem(t))

