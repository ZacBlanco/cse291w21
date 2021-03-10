import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from collections import defaultdict
import joblib
import string
import os.path
from sklearn import linear_model
a = open('spec.txt','r')

f1 = a.read().split('\n\n')

specs=[]
for i in f1:
    l=i.split('\n')
    specs.append(l)
print(specs)

b = open('program.txt','r')

f2 = b.read().splitlines()

programs=[]
for i in f2:
    l=i.split('\n')
    programs.append(l)
print(programs)
c = open('program.txt','r')

f3 = c.read().splitlines()

programs_f=[]
for i in f2:
    l=i.split('\n')
    programs_f.append(l)
print(programs_f)

def train():
    '''Given the positive and negatvie sample path
    Returns:
        classifier model
    '''
    """
    a = open('spec.txt','r')

    f1 = a.read().split('\n\n')

    specs=[]
    for i in f1:
        l=i.split('\n')
        specs.append(l)
    print(specs)

    b = open('program.txt','r')

    f2 = b.read().splitlines()

    programs=[]
    for i in f2:
        l=i.split('\n')
        programs.append(l)
    print(programs)
    c = open('program.txt','r')

    f3 = c.read().splitlines()

    programs_f=[]
    for i in f2:
        l=i.split('\n')
        programs_f.append(l)
    print(programs_f)

    wordCount = defaultdict(int)
    for d in spec:
        for w in list(d):
            wordCount[w] += 1

    for d in programs:
        for w in list(d):
            wordCount[w] += 1
    counts = [(wordCount[w], w) for w in wordCount]
    counts.sort()
    counts.reverse()

    words = [x[1] for x in counts[:len(wordCount)]]
    wordId = dict(zip(words, range(len(words))))
    wordSet = set(words)
    """
    X1 = [feature(d1, d2) for (d1, d2) in zip(specs, programs)]
    y1 = [[1] for d in programs]
    
    X2 = [feature(d1, d2) for (d1, d2) in zip(specs, programs_f)]
    y2 = [[0] for d in programs_f]
    X=X1+X2
    y=y1+y2
    clf = linear_model.Ridge(1.0, fit_intercept=False)  # MSE + 1.0 l2
    clf.fit(X, y)
    
# Save---format as pkl
    joblib.dump(clf,'train_rank.pkl')
 # Load

    return clf
    
def feature(datum1,datum2):
  
  feat = [0]*3*len(words)
  inputs=[]
  outputs=[]

  for i in datum1:
        inputs.append(i.split("; ")[0])
        outputs.append(i.split("; ")[1])
 
  for w in list(inputs):
    if w in words:
      feat[wordId[w]] += 1
    
  for w in list(outputs):
    if w in words:
      feat[len(words)+wordId[w]] += 1
    
  for w in list(datum2):
    if w in words:
      feat[2*len(words)+wordId[w]] += 1
    
  feat.append(1) #offset
  #print(feat)
  return feat
  
def test (inputs,output):
    if os.path.isfile('train_rank.pkl'):
        model = joblib.load('train_rank.pkl')
        print("load estimator")
    else:
        model=train()
    f=[feature(inputs, output[i]) for i in range(0,len(output))]
    #f=feature(input,output)
    return model.predict(f)[0]
def main():
    a=[]

    inputs=['Jenee Pannell; Dr. Jenee', 'Annalisa Gregori; Dr. Annalisa','Maryann Casler; Dr. Maryann']
    output=['f_1 ((_arg_0 String)) String(str.substr _arg_0 0 (+ (str.indexof _arg_0 " " (str.len (str.++ "ssp." "ssp."))) 0)))']
    print(len(output))

    a.append(output)
    print(test(inputs,a))

if __name__ == "__main__":
    main()



