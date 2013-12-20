# ------------------------------------------------------------
# removeNoiseWords.py
# 
# John Cardente
#
# Python script to remove apparently non important words
# from training dictionary
#
#
# ------------------------------------------------------------

import os, os.path
import sys
import argparse
import itertools

from collections import Counter
import cPickle as pickle

def countWords(dictMessages, tagList):

  # Create overall dictionary dictionary
  dictVocabAll    = Counter()
  dictVocabField  = {'body': Counter(), 'title':Counter()}
  dictVocabTag    = {t:{'body': Counter(), 'title':Counter()} for t in tagList}
  countVocabAll   = 0
  countVocabField = {'body':0, 'title':0}
  countVocabTag   = {t:{'body': 0, 'title':0} for t in tagList}

  for sid in dictMessages.iterkeys():
    tags  = [t for t in dictMessages[sid]['tags'] if t in tagList]
    if (len(tags) == 0):
      continue

    for f in ['title', 'body']:
      text = dictMessages[sid][f]

      for w,c in text.iteritems():
        dictVocabAll[w] += c
        countVocabAll   += c
        dictVocabField[f][w] += c
        countVocabField[f]   += c

      for t in tags:
        for w,c in text.iteritems():
          dictVocabTag[t][f][w] += c
          countVocabTag[t][f]   += c


  counts = {}
  counts['vocabAll']   = dictVocabAll  
  counts['vocabField'] = dictVocabField
  counts['vocabTag']   = dictVocabTag
  counts['countAll']   = countVocabAll   
  counts['countField'] = countVocabField
  counts['countTag']   = countVocabTag

  return counts


def chi2SelectWords(counts, numKeep):
  dictVocabAll    = counts['vocabAll']
  dictVocabField  = counts['vocabField']
  dictVocabTag    = counts['vocabTag']
  countVocabAll   = counts['countAll'] 
  countVocabField = counts['countField']
  countVocabTag   = counts['countTag'] 
  tagList         = dictVocabTag.keys()

  if (numKeep is None):
    numKeep = 1000
  else:
    numKeep = int(numKeep)

  dictChi2 = {}
  for f in ['title','body']:
    dictChi2[f] = set()
    for t in tagList:
      chi2Scores = []
      for w,c in dictVocabTag[t][f].iteritems():

        # Calculate number of documents/positions with and without the
        # term both inside and outside the class
        #
        # Nomenclature follows the book: N_term_class
        N   = countVocabAll          
        N11 = c                        
        N01 = countVocabTag[t][f] - N11 
        N10 = dictVocabAll[w] - N11 
        N00 = N - N11 - N01 - N10

        E11 = N * float(N11 + N10)/N * float(N11 + N01)/N
        E01 = N * float(N01 + N11)/N * float(N01 + N00)/N
        E10 = N * float(N10 + N11)/N * float(N10 + N00)/N
        E00 = N * float(N00 + N01)/N * float(N00 + N10)/N

        X2 = 0
        X2 += pow(N11-E11, 2)/E11 if (E11 > 0) else 0
        X2 += pow(N01-E01, 2)/E01 if (E01 > 0) else 0
        X2 += pow(N10-E10, 2)/E10 if (E10 > 0) else 0
        X2 += pow(N00-E00, 2)/E00 if (E00 > 0) else 0

        chi2Scores.append([w,X2])

      # Sort list, highest to lowest    
      chi2Scores.sort(key=lambda x: x[1], reverse=True)
      chi2WordsSorted = [ tt for tt,v in chi2Scores]

      # Save top scoring words
      dictChi2[f] |= set(chi2WordsSorted[0:min(len(chi2WordsSorted), numKeep)])

  return(dictChi2)


def giniSelectWords(counts, thresh):
  dictVocabAll    = counts['vocabAll']
  dictVocabField  = counts['vocabField']
  dictVocabTag    = counts['vocabTag']
  countVocabAll   = counts['countAll'] 
  countVocabField = counts['countField']
  countVocabTag   = counts['countTag'] 
  tagList         = dictVocabTag.keys()

  if (thresh is None):
    thresh = 0.25

  # Calculate within class popularity
  V = len(dictVocabAll)
  n = len(tagList)
  n2 = n * n
  tagList = dictVocabTag.keys()
  dictGiniField = {}
  for f in ['title','body']:
    giniScores = {}
    for w in dictVocabField[f].iterkeys():
      pWordTag = {}
      pWordSum = 0
      for t in tagList:
        if (w in dictVocabTag[t][f]):
          pWordTag[t] = (float(1 + dictVocabTag[t][f][w]) /
                         (V + countVocabTag[t][f]))
        else:
          pWordTag[t] = float(1)/V

        pWordSum   += pWordTag[t]


      # Calculate the within class popularity
      tagWCPs = [pWordTag[t] / pWordSum for t in tagList]
      tagWCPs.sort()
      sumWCPs = sum(tagWCPs)
      avgWCPs = sumWCPs/n

      # Calculate gini 
      gini = sum([(2*(i+1) - n - 1) * tagWCPs[i] 
                   for i in range(n)])
      gini /= (n2 * avgWCPs)
      gini *= float(n)/(n-1)

      giniScores[w] = gini 

    dictGiniField[f]= set([w for w,g in giniScores.iteritems() if g > thresh])
 
  return dictGiniField


def filterMessages(dictMessages, dictField):
  filteredDict = {}
  for sid in dictMessages.iterkeys():
    tags  = dictMessages[sid]['tags']
    title = dictMessages[sid]['title']
    body  = dictMessages[sid]['body']

    newTitle = {w:c for w,c in title.iteritems() if w in dictField['title']}
    newBody  = {w:c for w,c in body.iteritems()  if w in dictField['body']}

    filteredDict[sid] = {}
    filteredDict[sid]['tags']  = tags
    filteredDict[sid]['title'] = newTitle
    filteredDict[sid]['body']  = newBody

  return filteredDict


# ------------------------------------------------------------
# MAIN

methods = {
  'chi2': chi2SelectWords,
  'gini': giniSelectWords
}


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Remove apparently non important words.')

  parser.add_argument('method',
                      help='Method to use')

  parser.add_argument('inFile',
                      help='Input dictionary file')

  parser.add_argument('outFile',
                      help='Output dictionary file')

  parser.add_argument('-k',
                      action="store",
                      type=float,
                      dest="k",
                      default=None,
                      help='Parameter for selection method')

  args = parser.parse_args()

  if (args.method not in methods):
    sys.stderr.write('Unknown method')
    sys.exit()

  # Open input file
  fdDicts = open(args.inFile, 'rb')
  reader = pickle.Unpickler(fdDicts)
  dictMessages  = reader.load()
  dictTagCounts = reader.load()
  dictTagIndex  = reader.load()
  fdDicts.close()

  # Count words per tag and field
  counts = countWords(dictMessages, dictTagCounts.keys())

  # Figure out important words for eac tag/field
  dictSelected = methods[args.method](counts, args.k)

  # filter the data set to only include important words
  dictFiltered = filterMessages(dictMessages, dictSelected)

  # Write dictionaries out in pickle file
  outstream = open(args.outFile, 'wb')
  writer = pickle.Pickler(outstream, pickle.HIGHEST_PROTOCOL)
  writer.dump(dictFiltered)
  writer.dump(dictTagCounts)
  writer.dump(dictTagIndex)
  outstream.close()
