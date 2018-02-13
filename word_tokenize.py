from nltk import word_tokenize
from nltk import pos_tag
import nltk

snts = "I'm a python developer"

tok = word_tokenize(snts)
print(tok)

tag = pos_tag(tok)
print(tag)

ent = nltk.chunk.ne_chunk(tag)
print(ent)
