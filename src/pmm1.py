# ------------------------------------------------------------
# pmm1.py
# 
# John Cardente
#
# Python implementation of parameterized mixture models
# based on Naive Bayes classifiers for multi-label 
# prediction.
#
# Based on:
# http://machinelearning.wustl.edu/mlpapers/paper_files/AA30.pdf
#
# ------------------------------------------------------------

import os, os.path
import sys

from collections import Counter
from math        import log


def pmm1TrainSingle(dictMessages, f, trainIDs, tagList):

  # Compute aggregate per-label word frequencies to get initial
  # probabilities
  dictVocabAll  = Counter()
  dictVocabTag  = Counter()
  countVocabTag = Counter()  
  
  for t in tagList:
    dictVocabTag[t]  = Counter()

  for sid in trainIDs:
    tags  = [t for t in dictMessages[sid]['tags'] if t in tagList]
    text = dictMessages[sid][f]

    for w,c in text.iteritems():
      dictVocabAll[w] += c

    if (len(tags) == 0):
      continue

    for t in tags:
      for w,c in text.iteritems():
        dictVocabTag[t][w] += c
        countVocabTag[t]   += c

  pWordTag = {}
  for t in tagList:
    pWordTag[t] = {w:(float(c)/countVocabTag[t]) for 
                   w,c in dictVocabTag[t].iteritems()}

  # Iterate until convergence
  V = len(dictVocabAll)
  delta     = float("inf")
  lastMaxChange = 0
  iterations    = 0
  while (delta > .01):
    iterations += 1
 
    # Calculate G parameters
    g_nli = {t:{w:{} for w in pWordTag[t].keys()} for t in tagList}
    for sid in trainIDs:
      tags  = [t for t in dictMessages[sid]['tags'] if t in tagList]
      text = dictMessages[sid][f]

      for w in text.iterkeys():      
        sumProb = sum([pWordTag[t][w] for t in tags])
        for t in tags:
          g_nli[t][w][sid] = pWordTag[t][w] / sumProb

    # Calculate updated per-label word probabilities
    maxChange = 0
    pWordTagNew = {t:{} for t in tagList}
    for t in tagList:
      sumProbs = 0
      for w in pWordTag[t].iterkeys(): 
        pWordTagNew[t][w] = 0
        for sid in g_nli[t][w].iterkeys():
          tmp = dictMessages[sid][f][w] * g_nli[t][w][sid] + 1 
          pWordTagNew[t][w] += tmp
          sumProbs += tmp 

      sumProbs += V
      for w in pWordTag[t].iterkeys(): 
        pWordTagNew[t][w] /= sumProbs
        maxChange = max(maxChange, abs(pWordTagNew[t][w] - 
                                       pWordTag[t][w])/pWordTag[t][w])

    pWordTag = pWordTagNew

    delta  = abs(maxChange - lastMaxChange)
    lastMaxChange = maxChange


  # Build model structure with final probabilities
  model = {}
  model['pWordTag']     = pWordTag
  model['pWordSmooth']  = float(1)/V
  model['pTagPrior']    = float(1)/len(pWordTag)
  
  return model


def pmm1Train(dictMessages, features, trainIDs, tagList):
  pmm1Models = {}
  for f in features:
    pmm1Models[f] = pmm1TrainSingle(dictMessages, f, 
                                    trainIDs, tagList)

  return pmm1Models


def pmm1PredictTags(model, tags, dictText, includePriors=True):
 lpTag = 0
 for w,c in dictText.iteritems():
   lpTagWord = 0
   for t in tags:
     if (w in model['pWordTag'][t]):
       lpTagWord += model['pWordTag'][t][w]
     else:
       lpTagWord += model['pWordSmooth']

   lpTag += c*(log(lpTagWord) - log(len(tags)))

 if (includePriors):
   lpTag += len(tags) * log(model['pTagPrior'])
 
 return lpTag


def pmm1Predict(models, f, dictText, tagList):
  model         = models[f]
  currentTags   = set()
  remainingTags = set(tagList)
  lpCurrent     = -1 * sys.float_info.max
  converged     = False
  while ((not converged) and (len(remainingTags) > 0)):
    lpNew = {}
    for t in remainingTags:
      tryTags = currentTags | set([t])
      lpTags = pmm1PredictTags(model, list(tryTags), dictText)
      lpNew[tuple(tryTags)] = lpTags

    bestTag = max(lpNew.iteritems(), key=(lambda v: v[1]))
    if (bestTag[1] > lpCurrent):
      currentTags     = set(list(bestTag[0]))
      remainingTags  -= currentTags
      lpCurrent       = bestTag[1]
    else:
      converged = True

  return list(currentTags)




