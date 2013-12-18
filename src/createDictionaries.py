#!/usr/bin/python
# ------------------------------------------------------------
# createDictionaries.py
# 
# John Cardente
#
# Python script to create dictionaries for naive bayes
# classifiers
#
# ------------------------------------------------------------

import os, os.path
import sys
import argparse
import csv
import itertools
import string
import re 
import cPickle as pickle

from collections import Counter
from HTMLParser import HTMLParser

import nltk


# ------------------------------------------------------------
# MAIN

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Create pre-processed input samples for classifiers.')

  parser.add_argument('msgFile',
                      help='CSV file with input messages.')

  parser.add_argument('tagFile',
                      help='File list of tags to filter on.')

  parser.add_argument('dictFile',
                      help='Dictionary output file')

  args = parser.parse_args()
  sys.stdout.write(" msgFile: {}\n".format(str(args.msgFile)))
  sys.stdout.write(" tagFile: {}\n".format(str(args.tagFile)))
  sys.stdout.write("dictFile: {}\n".format(str(args.dictFile)))

  # Open and process the tag filter list file
  fd = open(args.tagFile)
  tagsString = fd.readline().rstrip('\n')
  fd.close()
  tagList = tagsString.split()

  # Open the training file and
  fd = open(args.msgFile)
  tmp_reader = csv.reader(fd, delimiter=',', quotechar='"')
  headings = tmp_reader.next()

  # Create dictionaries
  dictMessages   = {}
  dictTagCounts  = Counter()    
  dictTagIndex   = {}

  for t in tagList:
    dictTagIndex[t] = []

  for sample in tmp_reader:
    sid   = sample[0]
    title = sample[1].split()
    body  = sample[2].split()
    tags  = [t for t in sample[3].split() if t in tagList]

    if (len(tags)==0):
      continue 

    dictMessages[sid] = {'tags':tags, 
                         'title':Counter(title),
                         'body':Counter(body)}
    dictTagCounts.update(tags)

    for t in tags:
      dictTagIndex[t].append(sid)

  fd.close()

  # Write dictionaries out in pickle file
  outstream = open(args.dictFile, 'wb')
  writer = pickle.Pickler(outstream, pickle.HIGHEST_PROTOCOL)
  writer.dump(dictMessages)
  writer.dump(dictTagCounts)
  writer.dump(dictTagIndex)
  outstream.close()


