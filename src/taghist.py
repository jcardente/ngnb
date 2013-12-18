# ------------------------------------------------------------
# taghist.py
# 
# John Cardente
#
# Python script to calculate the histogram post tag lengths
#
# ------------------------------------------------------------

import os, os.path
import sys
import argparse
import time

from collections import Counter
from cPickle     import Unpickler





# ------------------------------------------------------------
# MAIN



if __name__ == "__main__":

  parser = argparse.ArgumentParser(
    description='Compute tag length histogram.')

  parser.add_argument('trainFile',
                      help='Training file.')

  args = parser.parse_args()


  sys.stderr.write("Loading dictionaries......")
  fdDicts = open(args.trainFile, 'rb')
  dictReader = Unpickler(fdDicts)
  dictMessages  = dictReader.load()
  dictTagCounts = dictReader.load()
  dictTagIndex  = dictReader.load()
  fdDicts.close()
  sys.stderr.write("DONE\n")

  docIDs  = dictMessages.keys()
  tagList = dictTagCounts.keys()
  tagHist = Counter()
  for sid in docIDs:
    tagsActual  = set([t for t in dictMessages[sid]['tags'] if t in tagList])

    l = len(tagsActual)
    if (l == 0):
      continue

    tagHist[l] += 1
    
  lkeys = sorted(tagHist.keys())
  avg   = 0
  for l in lkeys:
    print("{} : {}".format(l, tagHist[l]))
    avg += tagHist[l] * l


  print("Average: {}".format(float(avg)/len(docIDs)))

