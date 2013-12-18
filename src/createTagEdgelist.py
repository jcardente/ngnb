#!/usr/bin/python
# ------------------------------------------------------------
# creatTagEdgelist.py
# 
# John Cardente
#
# Python script to create an edge list with occurrance counter
# of the tags in the Stackexchange post training set. 
# ------------------------------------------------------------

import os, os.path
import sys
import argparse
import csv
import itertools
from collections import Counter
import cPickle as pickle


# ------------------------------------------------------------
# MAIN

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Facebook classification solution')

  parser.add_argument('trainFile',
                      help='Training file.')

  parser.add_argument('edgeFile',
                      help='Edgelist output file')

  parser.add_argument('-n',
                     action="store",
                     type=int,
                     dest="limit",
                     default=None,
                     help='Limit')

  args = parser.parse_args()
  sys.stdout.write("   Train: {}\n".format(str(args.trainFile)))
  sys.stdout.write("EdgeList: {}\n".format(str(args.edgeFile)))
  sys.stdout.write("   Limit: {}".format(str(args.limit)))

  # Open the training file and get tags. 
  fd = open(args.trainFile)
  tmp_reader = csv.reader(fd, delimiter=',', quotechar='"')

  # Loop over training samples and count each pair
  # of tags that appear together. 
  headings = tmp_reader.next()
  tagEdges = Counter()
  tagLengths = Counter()
  cnt = 1
  for sample in tmp_reader:
    # NB - sort first to ensure pairs map to same tuple!
    tags = sorted(sample[3].split())
    tagLengths[len(tags)] += 1     
    tagEdges.update([tp for tp in itertools.combinations(tags,2)])
    cnt += 1
    if (args.limit and cnt > args.limit):
      break 

  sys.stdout.write("\nProcessed {} entries\n".format(str(cnt)))    
  sys.stdout.write("Found: {} tag pairs\n".format(str(len(tagEdges))))

  # Write edgelist as csv file
  sys.stdout.write("Writing edgelist csv....\n")
  with open(args.edgeFile+'.csv', 'wb') as fdEdge:
     writer = csv.writer(fdEdge, delimiter=',',
                         quotechar='"', quoting=csv.QUOTE_MINIMAL)
     for edge, count  in tagEdges.items():
       writer.writerow([edge[0], edge[1], count])

  #sys.stdout.write("Writing tagEdges binary dump....\n")
  #with open(args.edgeFile+'.pk', 'wb') as edgestream:
  #  writer = pickle.Pickler(edgestream, pickle.HIGHEST_PROTOCOL)
  #  writer.dump(tagEdges)

  # Dump the tag length histogram
  sys.stdout.write("\n\nTag Lengths\n ")
  for k in sorted(tagLengths.keys()) :
    sys.stdout.write("{}\t{}\n".format(str(k),str(tagLengths[k])))





