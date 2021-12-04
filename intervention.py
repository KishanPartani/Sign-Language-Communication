import os
#import spacy
#nlp = spacy.load('en_core_web_md')

from nltk.corpus import wordnet

def datalist():
    mylist = os.listdir(os.getcwd() + '/static/dataset')
    ls = []
    for x in mylist:
        ls.append(os.path.splitext(x)[0])
    return ls

"""import csv
filename = 'data.csv'
rows = []
with open (filename, 'r') as fp:
    csvreader = csv.reader(fp)
    for row in csvreader:
        rows.append(row)"""
def similarWords(word):
    max_similarity = 0
    ls = datalist()
    syn1 = wordnet.synsets(word)[0]
    for i in ls:
        syn2 = wordnet.synsets(i)[0]
        similarity = syn1.wup_similarity(syn2)
        print(similarity)
        if similarity > max_similarity:
            max_similarity = similarity
            similarity_name = i
    if(max_similarity > 0.75):
        return similarity_name
    return ''

print(similarWords('Mali'))