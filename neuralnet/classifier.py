import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from collections import defaultdict
import nltk
import string
from nltk.stem.porter import *
from sklearn import linear_model

m = []
n = []
a = []
b = []
t = numpy.loadtxt(r'positive.txt', delimiter='\n', dtype=str)
ne = numpy.loadtxt(r'negative.txt', delimiter='\n', dtype=str)
# print(t)
# print(t)
for i in t:
    ll = i.split(", ")
    a.append(ll[0])
    b.append(ll[1])

for i in ne:
    ll = i.split(", ")
    m.append(ll[0])
    n.append(ll[1])

wordCount = defaultdict(int)
for d in a:
    for w in list(d):
        wordCount[w] += 1

for d in m:
    for w in list(d):
        wordCount[w] += 1
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()
counts.reverse()

words = [x[1] for x in counts[:len(wordCount)]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)
def train():
    '''Given the positive and negatvie sample path
    Returns:
        classifier model
    '''


    X1 = [feature(d1, d2) for (d1, d2) in zip(a, b)]
    y1 = [[1] for d in b]

    X2 = [feature(d1, d2) for (d1, d2) in zip(m, n)]
    y2 = [[0] for d in n]

    X = X1 + X2
    y = y1 + y2
    clf = linear_model.Ridge(1.0, fit_intercept=False)  # MSE + 1.0 l2
    clf.fit(X, y)
    theta = clf.coef_
    return clf
    # Regression
    # theta,residuals,rank,s = numpy.linalg.lstsq(X, y)


def feature(datum1,datum2):
  feat = [0]*2*len(words)
  
  for w in list(datum1):
    if w in words:
      feat[wordId[w]] += 1
  for w in list(datum2):
    if w in words:
      feat[len(words)+wordId[w]] += 1
  feat.append(1) #offset
  #print(feat)
  return feat

def test (model,input,output):
    f=[feature(d1, d2) for (d1, d2) in zip(input, output)]
    #f=feature(input,output)
    return model.predict(f)[0]
# Regularized regression
def main():
    model=train()
    print(test(model,["nitm US"], ["nitm"]))
#print(predictions)
#f=feature("+156 6563 324","156") #0.07162274
#f=feature("nitm US","nitm")
#print(len(f))
if __name__ == "__main__":
    main()
