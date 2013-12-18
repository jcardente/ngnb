# ------------------------------------------------------------
# runtest.py
# 
# John Cardente
#
# Python script to run and test models for CS229 project.
#
# ------------------------------------------------------------

import os, os.path
import sys
import argparse
import time

from cPickle       import Unpickler
from random        import shuffle
from nbMultinomial import *
from pmm1          import *
from ngnb          import *


# ------------------------------------------------------------
# TEST


def test(fnPredict, models, features, dictMessages, testDocIDs, tagList):
  results = {'tp': 0, 'fp': 0, 'fn': 0, 
             'mean-prec': 0, 'mean-rec': 0, 
             'mean-f1':0}
  for sid in testDocIDs:
    tagsActual  = set([t for t in dictMessages[sid]['tags'] if t in tagList])

    predictions = set()
    for f in features:
      text = dictMessages[sid][f]
      predictions |= set(fnPredict(models, f, text, tagList))

    #print(",".join(tagsActual) + " : " + ",".join(predictions))
    tp = len(tagsActual & predictions)
    fp = len(predictions - tagsActual)
    fn = len(tagsActual - predictions) 

    if ((tp + fp) > 0):
      precision = float(tp) / (tp + fp)
    else: 
      precision = 0

    if ((tp+fn) > 0):
      recall = float(tp) / (tp + fn)   
    else:
      recall = 0

    results['tp'] += tp
    results['fp'] += fp
    results['fn'] += fn
    results['mean-prec'] += precision
    results['mean-rec']  += recall 
    if ((precision + recall) > 0):
      results['mean-f1']   += 2 * (precision * recall) / (precision + recall)

  numTestDocs = len(testDocIDs)
  results['mean-prec'] /= numTestDocs
  results['mean-rec']  /= numTestDocs
  results['mean-f1']   /= numTestDocs

  return results


def resultString(tp, fp, fn, prec, rec, f1):
  return " {tp} | {fp} | {fn} | {prec:.3f} | {rec:.3f} | {f1:.3f} |".format(
    tp=tp, fp=fp, fn=fn, prec=prec, rec=rec, f1=f1) 


# ------------------------------------------------------------
# MAIN


methods = {
  "nbMulti" : {"name": "nbMulti", 
               "train": nbMultiTrain, 
               "predict": nbMultiPredict},
  "pmm1"    : {"name": "pmm1",    
               "train": pmm1Train,    
               "predict": pmm1Predict},
  "ngnb"    : {"name": "ngnb", 
               "train": ngnbTrain,    
               "predict": ngnbPredict}
}


if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Run and test models.')

  parser.add_argument('method',
                      help='Method to use')

  parser.add_argument('trainFile',
                      help='Training file.')

  parser.add_argument('testFile',
                      help='Testing file.')

  parser.add_argument('-nobody',
                      action="store_true",
                      dest="nobody",
                      help='Dont include the body')

  parser.add_argument('-notitle',
                      action="store_true",
                      dest="notitle",
                      help='Dont include the title')


  args = parser.parse_args()

  if (args.nobody and args.notitle) :
    print("Error, can't omit both body and title")
    sys.exit()

  if (not args.method in methods):
    print("Error, unknow method")
    sys.exit()

  features = []
  if (not args.notitle):
    features.append('title')

  if (not args.nobody):
    features.append('body')

  sys.stderr.write("Loading dictionaries......")
  fdDicts = open(args.trainFile, 'rb')
  trainReader = Unpickler(fdDicts)
  trainMessages  = trainReader.load()
  trainTagCounts = trainReader.load()
  trainTagIndex  = trainReader.load()
  fdDicts.close()

  fdDicts = open(args.testFile, 'rb')
  testReader = Unpickler(fdDicts)
  testMessages  = testReader.load()
  testTagCounts = testReader.load()
  testTagIndex  = testReader.load()
  fdDicts.close()

  sys.stderr.write("DONE\n")

  trainIDs = trainMessages.keys()
  testIDs  = testMessages.keys()
  tagList  = trainTagCounts.keys()

  sys.stderr.write("  Training................")    
  models      = methods[args.method]['train'](trainMessages, features, trainIDs, tagList)
  sys.stderr.write("DONE\n")

  sys.stderr.write("  Testing.................")    
  resultsTest = test(methods[args.method]['predict'], models, features, testMessages, 
                     testIDs, tagList)
  sys.stderr.write("DONE\n\n")

  nTags = len(tagList)
  nDocs = len(testIDs)
  outStr  = "| MODEL | NDOC | NTAG | "
  outStr += " TP |  FP | FN | PREC | REC | F1 |"
  outStr +="\n"
  sys.stdout.write(outStr)                    

  outStr  = "| {} | {} | {} |".format(methods[args.method]['name'], nDocs, nTags)
  outStr += resultString(resultsTest['tp'], resultsTest['fp'], resultsTest['fn'], 
                         resultsTest['mean-prec'], resultsTest['mean-rec'], 
                         resultsTest['mean-f1'])

  outStr +="\n"
  sys.stdout.write(outStr)
