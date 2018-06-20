import urllib.request
response = urllib.request.urlopen('http://php.net/')
html = response.read()
#print(html)
from bs4 import BeautifulSoup
soup = BeautifulSoup(html, "html5lib")
#print(soup)
text = soup.get_text(strip=True)
#print(text)
tokens = [t for t in text.split()]
#print(tokens)

import nltk
freq = nltk.FreqDist(tokens)
for key,val in freq.items():
    print(str(key)+ ":" +str(val))

#freq.plot(20,cumulative=False)

