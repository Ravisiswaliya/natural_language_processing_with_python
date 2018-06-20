from nltk import word_tokenize
from nltk import pos_tag
import nltk

snts = "I'm a python developer"

#word tokenize
tok = word_tokenize(snts)
print(tok)

#part of speech tag
tag = pos_tag(tok)
print(tag)

ent = nltk.chunk.ne_chunk(tag)
print(ent)
