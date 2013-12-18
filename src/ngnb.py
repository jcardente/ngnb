# ------------------------------------------------------------
# ngnb.py
# 
# John Cardente
#
# Python implementation of multinomial Naive Bayes classifiers 
# guided by an undirected graph of training set labels for multi-label 
# prediction.
#
# ------------------------------------------------------------

import itertools
import networkx as nx
import numpy    as np 
import pdb

from collections   import Counter
from nbMultinomial import *


# From Python itertools documentation
def powerset(iterable, max=None):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    if (max is None):
      max = len(s)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,max+1))


def ngnbGetAbovePercentile(g, fn, q=90, **kwargs):
   scores = fn(g, **kwargs)
   values = np.array([v for v in scores.itervalues()])      
   thresh = np.percentile(values, q)
   nodes = [n for n,v in scores.iteritems() if v >= thresh]   
   return nodes


def ngnbCreateNetwork(dictMessages, trainIDs, tagList):
  nodeCounts   = Counter()
  edgeCounts   = Counter()
  maxNodeCount = 0
  maxEdgeCount = 0

  g = nx.Graph()
  g.add_nodes_from(tagList)
  for sid in trainIDs:
    tags  = sorted([t for t in dictMessages[sid]['tags'] if t in tagList])

    g.add_nodes_from(tags)
    nodeCounts.update(tags)

    if (len(tags) > 1):
      for n1,n2 in itertools.combinations(tags,2):
        g.add_edge(n1,n2)
        edgeCounts[(n1,n2)] += 1
    else:
      # Add a self loop to track how many times node appears alone
      g.add_edge(tags[0],tags[0])
      edgeCounts[(tags[0],tags[0])] += 1

  for n in g.nodes_iter():
    maxNodeCount = max(maxNodeCount, nodeCounts[n])
    g.node[n]['count'] = nodeCounts[n]

  for e in [sorted([e1,e2]) for e1,e2 in g.edges_iter()]:
    maxEdgeCount = max(maxEdgeCount, edgeCounts[(e[0],e[1])])
    g.edge[e[0]][e[1]]['count'] = edgeCounts[(e[0],e[1])]

  # Add a distance attribute to edges
  for e in [sorted([e1,e2]) for e1,e2 in g.edges_iter()]:
    g.edge[e[0]][e[1]]['distance'] = (maxEdgeCount - 
                                      (float(edgeCounts[(e[0],e[1])])/maxEdgeCount)*maxEdgeCount + 
                                      1)
  return g



def ngnbTrain(dictMessages, f, trainIDs, tagList):
  # Create graph from tags and save top nodes based on centrality
  # scores for possible use during prediction.
  g = ngnbCreateNetwork(dictMessages, trainIDs, tagList)
  topNodes = set()
  topNodes |= set(ngnbGetAbovePercentile(g, 
                                         nx.degree, q=90))
  topNodes |= set(ngnbGetAbovePercentile(g, 
                                         nx.closeness_centrality, 
                                         q=90, distance='distance'))
  topNodes |= set(ngnbGetAbovePercentile(g, nx.betweenness_centrality, 
                                         q=90, weight='distance'))

  if (len(topNodes) == 0):
    pdb.set_trace()

  countTagTotal = 0
  for n in g.nodes():
    countTagTotal += g.node[n]['count']

  countEdgeTotal = 0
  for e in g.edges():
    countEdgeTotal += g.edge[e[0]][e[1]]['count']

  # Train binary relevance multinomial model
  models = nbMultiTrain(dictMessages, f, trainIDs, tagList)
  models['network'] = {'g':g, 
                       'topNodes': topNodes,
                       'countTrainDocs': len(trainIDs),
                       'countTagTotal' : countTagTotal,
                       'countEdgeTotal': countEdgeTotal }

  return models


def ngnbPredict(models, f, dictText, tagList): 

  # TUNABLES
  SEARCH_LIMIT   = 5
  DAMPING_FACTOR = 0.7

  model   = models[f]
  network = models['network']
  g       = models['network']['g']
  topTags = models['network']['topNodes']

  # NB
  #
  # For better scalability, can use the top centrality nodes
  # to start a search for the best initial tag.
  #
  # remainingTags = set(topTags)
  # startTag    = None
  # lpStart     = -1 * sys.float_info.max
  # while(len(remainingTags) > 0):
  #   tagLogOdds = nbMultiPredictLogOdds(models, f, dictText, remainingTags)
  #   bestTag = max(tagLogOdds.iteritems(), key=(lambda v: v[1]))
  #   if (bestTag[1] > lpStart):
  #     startTag = bestTag[0]
  #     lpStart  = bestTag[1]
  #     remainingTags = set([n for n in nx.all_neighbors(g,startTag)])
  #   else: 
  #     remainingTags = set()
  #  
  tagLogOdds = nbMultiPredictLogOdds(models, f, dictText, tagList)
  bestTag = max(tagLogOdds.iteritems(), key=(lambda v: v[1]))
  startTag = bestTag[0]
  lpStart  = bestTag[1]

  # Examine neighbors around starting tag to find best
  # candidate set. Set the starting log odd to the weighted
  # ratio based on the number of selfloops. This approximates
  # the likelihood that the starting tag is by itself.
  currentTags   = set([startTag])
  if (g.has_edge(startTag,startTag)):
    lpCurrent = lpStart * (float(g.edge[startTag][startTag]['count']) /
                           g.node[startTag]['count'])
  else:
    lpCurrent =  -1 * sys.float_info.max

  neighborTags  = set([n for n in nx.all_neighbors(g,startTag)])
  neighborTags -= currentTags
  goodTags      = [(t,tagLogOdds[t]) for t in neighborTags if tagLogOdds[t] >=0]

  if (len(goodTags) > 0):

    goodTags.sort(key=(lambda v: v[1]), reverse=True)
    goodTags = goodTags[0:min( SEARCH_LIMIT, len(goodTags))]
    goodTags = set([v[0] for v in goodTags])
    tagsPowerset = powerset(list(goodTags))

    lpNew = {}
    for tagsCombo in tagsPowerset:
      # get the sub-graph and remove all self-loops
      tryTags = currentTags | set(tagsCombo)
      sg = nx.subgraph(g,list(tryTags))
      sg.remove_edges_from(sg.selfloop_edges())

      # NB
      #
      # Determine the smallest edgecount in the graph. This 
      # is basically the negative feedback that avoids the
      # subgraph expanding indefinitely. The rational is
      # that the tags are a set and therefore their joint
      # occurrence is, at most, the smallest edge weight.
      #
      # Scale the damping coefficient by a tunable paramter.
      #
      smallestCount = min([sg.edge[e[0]][e[1]]['count'] for e in sg.edges_iter()])
      dampingCoef   = DAMPING_FACTOR * (float(smallestCount)/
                                        g.node[startTag]['count'])

      # Starting with the tags farthest from the startTag, accumulate
      # weighted log odds forward
      tagDist = [(t,nx.shortest_path_length(sg,t,startTag)) for t in tryTags]
      tagDist.sort(key=(lambda v: v[1]), reverse=True)
      tagDistDict = {t:d for t,d in tagDist}
      accruals = Counter()
      for sgTag,d in tagDist:

        # Weight the per-tag log odd by the damping coefficient
        lpThisTag   = tagLogOdds[sgTag] * dampingCoef
        #if (g.has_edge(sgTag,sgTag)):
        #    # only take log odds for non-single occurrences
        #    lpThisTag *= (float(g.node[sgTag]['count'] - 
        #                        g.edge[sgTag][sgTag]['count']) /
        #                  g.node[sgTag]['count'])
        accruals[sgTag] += lpThisTag

        if (d > 0):
          # Propogate accrued log odds to next closest
          # tags
          nextTags = [nt for nt in nx.all_neighbors(sg,sgTag) if tagDistDict[nt] < d]
          for nt in nextTags:
            accruals[nt] += accruals[sgTag] /len(nextTags)

        else:
          # This had better be the start node
          if (not (sgTag == startTag)):
            pdb.set_trace()

      lpNew[tuple(tagsCombo)] = accruals[startTag]

    # Anything beat the single tag case? 
    bestTag = max(lpNew.iteritems(), key=(lambda v: v[1]))
    if (bestTag[1] > lpCurrent):
      currentTags    = currentTags | set(bestTag[0])
      lpCurrent      = bestTag[1]

  return list(currentTags)


   



  
