# ------------------------------------------------------------
# nbMultinomial.py
# 
# John Cardente
#
# Python implementation of a binary relevance 
# multinomial Naive Bayes classifier for multi-label
# prediction.
#
# ------------------------------------------------------------

import os, os.path
import sys

from collections import Counter
from math        import log
from math        import exp

def nbMultiTrainSingle(dictMessages, messageKey, trainDocIDs, tagList):

  # Create overall dictionary dictionary
  dictVocabAll  = Counter()
  dictVocabTag  = {}  
  countVocabAll = 0
  countVocabTag = Counter()  
  countTagDocs  = Counter()

  for t in tagList:
    dictVocabTag[t]  = Counter()

  for sid in trainDocIDs:
    tags  = [t for t in dictMessages[sid]['tags'] if t in tagList]
    text = dictMessages[sid][messageKey]

    for w,c in text.iteritems():
      dictVocabAll[w] += c
      countVocabAll   += c

    if (len(tags) == 0):
      continue

    countTagDocs.update(tags)
    for t in tags:
      for w,c in text.iteritems():
        dictVocabTag[t][w] += c
        countVocabTag[t]   += c

  # Calculate log probabilities
  model = {}
  model['tags'] = tagList
  model['messageKey']    = messageKey
  model['lpWordAll']     = {}
  model['lpWordTag']     = {}
  model['lpWordNotTag']  = {}
  model['lpPriorTag']    = Counter(tagList)
  model['lpPriorNotTag'] = Counter(tagList)  

  for t in tagList:
    model['lpWordTag'][t] = {}
    model['lpWordNotTag'][t] = {}

  # Tag priors
  numDocs    = len(trainDocIDs)
  logNumDocs = log(numDocs)
  for t,c in countTagDocs.iteritems():
    model['lpPriorTag'][t] = log(c) - logNumDocs
    if (c < numDocs):
      model['lpPriorNotTag'][t] = log(numDocs - c) - logNumDocs
    else:
      model['lpPriorNotTag'][t] = 0

  # Smoothing parameter for words not in the vocabulary
  smooth = len(dictVocabAll)
  model['lpSmooth'] = log(1) - log(smooth)

  # Calculate per tag positive and negative probabilities
  for w,c in dictVocabAll.iteritems():
    model['lpWordAll'][w] = log(c+1) - log(countVocabAll + smooth)

    for t in tagList:
      if (w in dictVocabTag[t]):
        wTagCount = dictVocabTag[t][w]
        wTagTotal = countVocabTag[t]
        model['lpWordTag'][t][w] = log(wTagCount + 1) - log(wTagTotal + smooth)
      else:
        wTagCount = 0
        wTagTotal = 0

      model['lpWordNotTag'][t][w] = (log(c - wTagCount + 1) - 
                                     log(countVocabAll - wTagTotal + smooth))

  return model



def nbMultiTrain(dictMessages, features, trainDocIDs, tagList):
  models = {}
  for f in features:
    models[f] = nbMultiTrainSingle(dictMessages, f, 
                                   trainDocIDs, tagList)
  return models


def nbMultiPredictLogOdds(models, f, dictText, tagList): 
  model   = models[f]
  pTag    = Counter(tagList)
  pNotTag = Counter(tagList)
  for t in tagList:
    pTag[t]    += model['lpPriorTag'][t]
    pNotTag[t] += model['lpPriorNotTag'][t]

  for w,c in dictText.iteritems():
    for t in tagList:  
      # NB - the unconditional probability of each word is
      #      not calculated or considered since it is a
      #      constant between the in-tag and out-of-tag 
      #      probabilities.
      if (w in model['lpWordTag'][t]):
        pTag[t]    += c * (model['lpWordTag'][t][w])
      else:
        pTag[t] += c * (model['lpSmooth'])

      if (w in model['lpWordNotTag'][t]):
        pNotTag[t] += c * (model['lpWordNotTag'][t][w])
      else:
        pNotTag[t] += c * (model['lpSmooth'])

  tagLogOdds = {t:(pTag[t] - pNotTag[t]) for t in tagList}
  return tagLogOdds


def nbMultiPredict(models, f, dictText, tagList): 
  tagLogOdds     = nbMultiPredictLogOdds(models, f, dictText, tagList)
  tagPredictions = [ t for t,lo in tagLogOdds.iteritems() if lo >=0]
  return tagPredictions 




          
