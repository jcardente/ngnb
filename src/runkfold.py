# ------------------------------------------------------------
# runmodel.py
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
# KFOLD


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


def runKfold(model, features, 
             dictMessages, docIDs, tagList, 
             kfolds, kstop, 
             doTrainTest=True, doTest=True):

  # Randomize the docIDs
  shuffle(docIDs)

  # Determine cross validation bins
  sys.stderr.write("Computing folds...........")
  if (kfolds > 1):
    binSize  = int(len(docIDs)/kfolds)
    binSizes = [binSize] * kfolds
    binSizes[kfolds -1] += len(docIDs) - sum(binSizes)

    binStarts = [sum(binSizes[:i]) for i in range(0, len(binSizes))]
    binIDs = [docIDs[b:(b+s)] for b,s in zip(binStarts, binSizes)] 
  else:
    binIDs = [docIDs]

  sys.stderr.write("DONE\n")

  nTags = len(tagList)
  nDocs = len(docIDs)
  outStr  = "| MODEL | NDOC | NTAG | FOLD |"
  if (doTrainTest):
    outStr += " TrTP |  TrFP | TrFN | TrPREC | TrREC | TrF1 |"
  if (doTest):           
    outStr += " TP |  FP | FN | PREC | REC | F1 |"
  outStr += " TTRAIN |"
  if (doTest):
    outStr += " TTEST |"

  outLog = [outStr]
  for fold in range(kfolds):
    if ((kstop is not None) and (fold >= kstop)):
      break

    sys.stderr.write("-- Fold {}\n".format(fold+1))
    testIDs  = binIDs[fold]
    if (kfolds > 1):
      trainIDs = list((set(docIDs)).difference(testIDs))
    else:
      trainIDs = binIDs[fold]

    sys.stderr.write("  Training................")    
    trainTime = time.time()
    models  = model['train'](dictMessages, features, trainIDs, tagList)
    trainTime = time.time() - trainTime
    sys.stderr.write("DONE\n")

    if (doTrainTest):
      sys.stderr.write("  Training Test...........")    
      resultsTrain = test(model['predict'], models, features, dictMessages, trainIDs, tagList)
      sys.stderr.write("DONE\n")

    if (doTest):
      sys.stderr.write("  Testing.................")    
      testTime = time.time()
      resultsTest = test(model['predict'], models, features, dictMessages, testIDs, tagList)
      testTime = time.time() - testTime
      sys.stderr.write("DONE\n")

    outStr  = "| {} | {} | {} | {} |".format(model['name'], nDocs, nTags, fold)
    if (doTrainTest):
      outStr += resultString(resultsTrain['tp'], resultsTrain['fp'], resultsTrain['fn'], 
                             resultsTrain['mean-prec'], resultsTrain['mean-rec'], 
                             resultsTrain['mean-f1'])
    if (doTest):
      outStr += resultString(resultsTest['tp'], resultsTest['fp'], resultsTest['fn'], 
                             resultsTest['mean-prec'], resultsTest['mean-rec'], 
                             resultsTest['mean-f1'])
    outStr += " {:.3f} |".format(trainTime)
    if (doTest):
      outStr += " {:.3f} |".format(testTime)

    outLog.append(outStr)

  for l in  outLog:
    print(l)



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
    description='Run and test models with Kfold validation.')

  parser.add_argument('method',
                      help='Method to use')

  parser.add_argument('trainFile',
                      help='Training file.')

  parser.add_argument('-k',
                      action="store",
                      type=int,
                      dest="k",
                      default=4,
                      help='Number of folds')

  parser.add_argument('-n',
                      action="store",
                      type=int,
                      dest="kstop",
                      default=None,
                      help='Number of folds to test')

  parser.add_argument('-ntag',
                      action="store",
                      type=int,
                      dest="ntag",
                      default=None,
                      help='Number of randomly select tags.')

  parser.add_argument('-l',
                      action="store",
                      type=int,
                      dest="docLimit",
                      default=None,
                      help='Number of documents to include.')

  parser.add_argument('-nobody',
                      action="store_true",
                      dest="nobody",
                      help='Dont include the body')

  parser.add_argument('-notitle',
                      action="store_true",
                      dest="notitle",
                      help='Dont include the title')

  parser.add_argument('-notraintest',
                      action="store_false",
                      dest="doTrainTest",
                      help='Dont do training test')

  parser.add_argument('-notest',
                      action="store_false",
                      dest="doTest",
                      help='Dont do test')


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
  reader = Unpickler(fdDicts)
  dictMessages  = reader.load()
  dictTagCounts = reader.load()
  dictTagIndex  = reader.load()
  fdDicts.close()
  sys.stderr.write("DONE\n")

  docIDs   = dictMessages.keys()
  tagList  = dictTagCounts.keys()

  if (args.ntag is not None):
    shuffle(tagList)
    tagList = tagList[0:args.ntag]

    filteredDocIDs = []
    for sid in docIDs:
      docTags = set([t for t in dictMessages[sid]['tags'] if t in tagList])
      if (len(docTags) > 0):
        filteredDocIDs.append(sid)
    docIDs = filteredDocIDs

  if (args.docLimit is not None):
    shuffle(docIDs)
    docIDs = docIDs[0:args.docLimit]
                
  runKfold(methods[args.method], features, 
           dictMessages, docIDs, tagList, 
           args.k, args.kstop, 
           args.doTrainTest, args.doTest)
