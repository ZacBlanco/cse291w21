import numpy
from urllib.request import urlopen
from collections import defaultdict
import joblib
import os.path
from sklearn import linear_model
a = open('neuralnet/spec.txt','r')

f1 = a.read().split('\n\n')

specs=[]
for i in f1:
    l=i.split('\n')
    specs.append(l)
#print(specs)

b = open('neuralnet/program.txt','r')

f2 = b.read().splitlines()

programs=[]
for i in f2:
    l=i.split('\n')
    programs.append(l)
#print(programs)
c = open('neuralnet/program_f.txt','r')

f3 = c.read().splitlines()

programs_f=[]
for i in f2:
    l=i.split('\n')
    programs_f.append(l)
#print(programs_f)
wordCount = defaultdict(int)

for d in specs:
        for w in list(d):
            for i in list(w):
                #print(i)
                wordCount[i] += 1

for d in programs:
        for w in list(d):
            for i in list(w):
                #print(i)
                
                wordCount[i] += 1

for d in programs_f:
        for w in list(d):
            for i in list(w):
                wordCount[i] += 1
#print(wordCount)
counts = [(wordCount[w], w) for w in wordCount]
counts.sort()

counts.reverse()

words = [x[1] for x in counts[:len(wordCount)]]
wordId = dict(zip(words, range(len(words))))
wordSet = set(words)
#print(wordSet)
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
    #print(X1)
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
  
    for i in list(datum1):
        inputs.append(i.split("; ")[0])
        outputs.append(i.split("; ")[1])
    
    
    for w in list(inputs):
        
        for i in range(0,len(w)):
            if w[i] in words:
              #print(w[i])
              feat[wordId[w[i]]] += 1
    
    for w in list(outputs):
        for i in range(0,len(w)):
            if w[i] in words:
              #print(w[i])
              
              feat[len(words)+wordId[w[i]]] += 1
    
    
    for w in list(datum2):
        for i in range(0,len(w)):
            if w[i] in words:
              #print(w[i])
              
              feat[2*len(words)+wordId[w[i]]] += 1
    
    feat.append(1) #offset
  #print(feat)
    return feat
  
def test (inputs,output):
    if os.path.isfile('train_rank.pkl'):
        model = joblib.load('train_rank.pkl')
        #print("load estimator")
    else:
        model=train()
    #f=[feature(inputs, output) ]
    #f=feature(input,output)
    #return model.predict(f)[0]
    r=[]
    t=[]
    #print(inputs)
    for o in output:
        #print(o)
        t.append(o)
        f=[feature(inputs,o)]
        #print(f)
        result=model.predict(f)[0]
        t=[]
        r.append(result[0])
    #f=feature(input,output)
    #print(r)
    return r
def main():
    a=[]

    inputs=['Jenee Pannell; Dr. Jenee', 'Annalisa Gregori; Dr. Annalisa','Maryann Casler; Dr. Maryann']
    output=['(str.substr _arg_0 0 String(str.substr _arg_0 0 (+ (str.indexof _arg_0 " " (str.len (str.++ "ssp." "ssp."))) 0)))']
    #print(len(output))

    a.append(output)
    print(test(inputs,output))

if __name__ == "__main__":
    main()
    #train()




