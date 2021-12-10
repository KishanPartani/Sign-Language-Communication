import os
import difflib
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
    print(ls)
    #ls.remove('cartoonize')
    return difflib.get_close_matches(word, ls)

print(similarWords('afirca'))