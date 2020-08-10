# doc_class_algos

**decision_tree.py**
- From-scratch implementation of [decision tree](https://en.wikipedia.org/wiki/Decision_tree) using python used as a document classifier
- To invoke: ```python decision_tree.py [.txt training data] [.txt test data] [(int) max depth] [(float) min gain] [model file path] [output path]```
- Each line in training and testing represents a document and must be formatted as below:
```
label1 word1:count1 word2:count2 ...
label2 word1:count1 word2:count2 ...
...
```
For example, if a politics document only contained the words "an", "about", "absurd", and "again", each with their respective counts, then it would be respresented on one line as: 
```politics a:11 about:2 absurd:1 again:1```
- Max depth specifies the maximum depth of the tree
- Min gain specifies the minimum information gain needed to make another split in the tree
- Model file specifies path for file to show how many documents end up at each leaf node and the percentage of documents at the leaf node
- Output path specifies where test results will be stored

**knn.py**
- [K nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) implementation for document classification.
- To invoke: ```python knn.py [.txt training data] [.txt test data] [(int) k] [(int) similarity] [output path]```
- Train and test files have same format as decision tree files above.
- k specifies the number of neighbors
- Similarity specifies the measure of distance between points in the vector space (1=euclidean, 2=cosine)
- Output path specifies where test results will be stored


**naive_bayes.py**
- From-scratch implementation of [naive bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) using python used as a document classifier
- To invoke: ```python naive_bayes.py [.txt training data] [.txt test data] [(float) class prior delta] [(float) cond prob delta] [model file path] [output path]```
- Train and test data same format as above
- Class prior delta is the δ used in add-δ smoothing when calculating the class prior P(c). Cond prob delta is the δ used in add-δ smoothing when calculating the conditional probability P(f | c).
- Model file specifies path for file to show various class probabilities
- Output path specifies where test results will be stored

**chi_2.py**
- Implementation of [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test) for feature selection
- To invoke: ```python chi_2.py [.txt training data]```
