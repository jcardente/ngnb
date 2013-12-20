Introduction
============

The Network Guided Naive Bayes (NGNB) classifier is a multi-label
classifier that uses an undirected graph built from the training set
labels to to help with prediction. I developed NGNB as a class project
for Stanford's [CS229 Machine Learning](http://cs229.stanford.edu) class.

Also included in this repository are implementations of a Binary
Relevance Multinomial Naive Bayes classifier and Parameteric Mixture
Models Naive Bayes classifier. These models served as baselines for
evaluating effectiveness and efficiency of NGNB.

Overall, NGNB did well when evaluated using the data set described
in the next session. It achieved F1 scores similar to PMM1 with 
efficiencies similar to the binary Multinomial models. See the 
[associated paper](http://dsgeek.com/docs/jcardente_cs229_project.pdf) for more details. 


Example
=======

The following example illustrates how to use the code in this
repository. The code and supporting scripts was written for use with
the StackExchange dataset provided in the 
[Kaggle Facebook Recruiting Challenge III](http://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction)
competition. 

Since the original data set was very, very large, a smaller training set was created
using the following steps. 

1. Create an edge list for the first 200K training examples. 

   ```python src/createTagEdgelist.py data/Train.csv data/edgelist200K  -n 200000```

2. Extract a subset of labels and create a network diagram of their relationships.

   ```Rscript src/analyzeTagNetwork.R data/edgelist200K.csv data/selectedtags.txt tags_netdiag.pdf```

3. Create a reduced training set containing only the selected tags.

   ```python src/createFilteredDataset.py data/Train.csv data/TrainReduced.csv data/selectedtags.txt -n 100000```

4. Process the training examples and save as pickled Python dictionaries. 

   ```python src/createDictionaries.py data/TrainReduced.csv data/selectedtags.txt data/trainDicts.pk```

5. Remove uninformative words using within-class-popularity and GINI criteria.

   ```python src/removeNoiseWords.py gini data/trainDicts.pk data/trainGiniDicts.pk -k 0.25```


A test data set was also created as follows,

1. Run createFilteredDataset to make smaller training set

   ```python src/createFilteredDataset.py data/Train.csv data/TestReduced.csv data/selectedtags.txt -n 15000 -s 110000```

2. Run createDictionaries to get words associated with each tag

   ```python src/createDictionaries.py data/TestReduced.csv data/selectedtags.txt data/testDicts.pk```


The three models were evaluated using 10-fold cross validation by running the 
following commands

```
python src/runkfold.py nbMulti data/trainGiniDicts.pk -k 10 -notraintest 
python src/runkfold.py pmm1    data/trainGiniDicts.pk -k 10 -notraintest 
python src/runkfold.py ngnb    data/trainGiniDicts.pk -k 10 -notraintest 
```

Sensitivity to the number of tags was evaluated by running 10-fold validation
with randomly selected subsets of tags using the simple shell script,

```
#!/bin/bash

for model in nbMulti pmm1 ngnb
do
for tsize in 5 10 20 50
do
  python src/runkfold.py ${model} data/trainGiniDicts.pk -k 10 -notraintest -ntag ${tsize} -l 50000
done
done
```

The following commands were used to evaluate the models using the 
test data set,

```
python src/runtest.py nbMulti  data/trainGiniDicts.pk data/testDicts.pk
python src/runtest.py pmm1     data/trainGiniDicts.pk data/testDicts.pk
python src/runtest.py ngnb     data/trainGiniDicts.pk data/testDicts.pk
```