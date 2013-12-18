#!/usr/bin/python
# ------------------------------------------------------------
# createFilteredDatast.py
# 
# John Cardente
#
# Python script to a smaller dataset using a list of desired
# tags. 
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
import pdb

# ------------------------------------------------------------
# TEXT CLEANUP

STOP_FILE = "data/english.stop"
stopwords = set()

def load_stopwords():
  global stopwords
  with open(STOP_FILE, 'r') as stopwfile:
    for word in stopwfile:
      stword = word.strip()
      stopwords.add(stword)
  return

def clean_text(s):  
  #s = strip_tags(s)
  s = nltk.clean_html(s)
  s = s.lower()
  s = s.translate(None, string.punctuation)
  NUMRE = re.compile(r"^[0-9]+$")
  s = [x for x in s.split() if not NUMRE.match(x)]
  s = [x for x in s if not x in stopwords]
  return s


# ------------------------------------------------------------
# MAIN

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Facebook classification solution')

  parser.add_argument('trainFile',
                      help='Training file.')

  parser.add_argument('trainFileNew',
                      help='output file')

  parser.add_argument('tagFile',
                      help='tag list file')

  parser.add_argument('-n',
                     action="store",
                     type=int,
                     dest="limit",
                     default=None,
                     help='Limit')


  parser.add_argument('-s',
                     action="store",
                     type=int,
                     dest="skip",
                     default=0,
                     help='skip')

  args = parser.parse_args()
  sys.stdout.write("  Train: {}\n".format(str(args.trainFile)))
  sys.stdout.write(" Output: {}\n".format(str(args.trainFileNew)))
  sys.stdout.write("tagFile: {}\n".format(str(args.tagFile)))
  sys.stdout.write("  Limit: {}\n".format(str(args.limit)))
  sys.stdout.write("   Skip: {}\n".format(str(args.skip)))

  # Open and process the tag filter list file
  fd = open(args.tagFile)
  tagsString = fd.readline().rstrip('\n')
  fd.close()
  tagsList = tagsString.split()

  # Load the stop words list
  load_stopwords()

  # Open the training file
  fd = open(args.trainFile)
  tmp_reader = csv.reader(fd, delimiter=',', quotechar='"')
  headings = tmp_reader.next()

  fdw = open(args.trainFileNew, 'wb')
  tmp_writer = csv.writer(fdw, delimiter=',',
                          quotechar='"', 
                          quoting=csv.QUOTE_ALL)
  tmp_writer.writerow(headings)


  # Loop over training samples and count each pair
  # of tags that appear together. 
  cnt     = 0
  skipcnt = 0
  rawcnt  = 0
  for sample in tmp_reader:
    # NB - sort first to ensure pairs map to same tuple!
    rawcnt += 1
    sid   = sample[0]
    title = sample[1]
    body  = sample[2]
    tags  = [t for t in sample[3].split() if t in tagsList]

    if (len(tags)==0):
      continue 

    skipcnt += 1
    if (skipcnt < args.skip):
      continue

    # Process stuff here
    newTitle = ' '.join(clean_text(title))
    newBody  = ' '.join(clean_text(body))
    newTags  = ' '.join(tags)
    tmp_writer.writerow([sid, newTitle, newBody, newTags])

    cnt += 1
    if (args.limit and cnt > args.limit):
      break 

  fd.close()
  fdw.close()

  sys.stdout.write("\nProcessed {} entries\n".format(str(cnt)))    
  sys.stdout.write("\n      Raw {} entries\n".format(str(rawcnt)))    





