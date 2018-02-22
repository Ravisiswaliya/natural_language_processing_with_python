from nltk.corpus import wordnet

#synset
sy = wordnet.synsets('work')
#print(syns[0].name())

#defination
query = wordnet.synsets('materialistic')
#print(query[0].name())
#print(query[1].definition())

#examples
#print(sy[0].examples())

#check the synonyms of given word
def find_syno(word):
    for s in wordnet.synsets(word):
        print(s.name()[:-5])
find_syno('tilt')

# check the similarity between two word
def word_similiarity(word,word1):
    word = word+'.n.01'
    word1 = word1+'.n.01'
    w1 = wordnet.synset(word)
    w2 = wordnet.synset(word1)
    print(w1.wup_similarity(w2))

word_similiarity('coding','programming')





