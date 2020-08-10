# Author: Devin Johnson

import sklearn.metrics
import sys
import time
import math
import numpy as np


# Files
train_file = sys.argv[1]
test_file = sys.argv[2]
k = int(sys.argv[3])
sim_func = int(sys.argv[4])
s_file = sys.argv[5]

# Relevant problem info (vocab, classes, etc)
# Forms columns labels of matrix
word_to_index = {}
index_to_word = {}
class_to_num = {}
num_to_class = {}
n_train_docs = 0
n_test_docs = 0


# Get vocab, word indices, classes, etc.
# Set up all the necessary info for the problem
def setup_problem(train_data):
    with open(train_data) as t:
        vocab = set()
        for line in t:
            s = line.split()
            # Add class to classes
            if s[0] not in class_to_num:
                class_to_num[s[0]] = len(class_to_num)
                num_to_class[len(num_to_class)] = s[0]

            # Add words to vocab
            for i in range(1, len(s)):
                vocab.add(s[i].split(":")[0])
        
        # Sort the vocab, get index nums
        vocab = sorted(list(vocab))
        for i in range(0, len(vocab)):
            word_to_index[vocab[i]] = i
            index_to_word[i] = vocab[i]


# Build a numpy matrix from train and test data
# Use one large matrix
def build_matrix(train, test):
    matrix = []
    global n_train_docs
    global n_test_docs
    with open(train) as f:
        # For each document
        for line in f:
            # Build a new row
            curr_row = []
            word_counts = {}
            s = line.split(" ")
            # Sentences have format: label word1:count word2:count
            doc_words = set()
            for i in range(1, len(s)):
                word_and_count = s[i].split(":")
                if len(word_and_count) > 1:
                    curr_word = word_and_count[0]
                    curr_word_count = float(word_and_count[1].strip())
                    word_counts[curr_word] = curr_word_count
                    doc_words.add(curr_word)
            
            # For word in vocab, see if doc has or does not have it
            for i in range(0, len(index_to_word)):
                if index_to_word[i] in doc_words:
                    curr_row.append(word_counts[index_to_word[i]])
                else:
                    curr_row.append(0)
            
            # Append label to end
            curr_row.append(class_to_num[s[0]])

            # Append current row to matrix
            matrix.append(curr_row)
            n_train_docs += 1

    with open(test) as f:
        # For each document
        for line in f:
            # Build a new row
            curr_row = []
            word_counts = {}
            s = line.split(" ")
            # Sentences have format: label word1:count word2:count
            doc_words = set()
            for i in range(1, len(s)):
                word_and_count = s[i].split(":")
                if len(word_and_count) > 1:
                    curr_word = word_and_count[0]
                    curr_word_count = float(word_and_count[1].strip())
                    word_counts[curr_word] = curr_word_count
                    doc_words.add(curr_word)
            
            # For word in vocab, see if doc has or does not have it
            for i in range(0, len(index_to_word)):
                if index_to_word[i] in doc_words:
                    curr_row.append(word_counts[index_to_word[i]])
                else:
                    curr_row.append(0)
            
            # Append label to end
            curr_row.append(class_to_num[s[0]])

            # Append current row to matrix
            matrix.append(curr_row)
            n_test_docs += 1
    
    return np.array(matrix)


# Classify data using the model
def classify(matrix, distances, k, m, datatype):
    
    # Rows = docs
    # Cols = classes (+1 for true class)
    classifications = np.zeros((len(distances), len(class_to_num) + 1))

    # Go over each test doc in distances matrix
    for i in range(0, len(distances)):
        
        # Current test doc
        row = distances[i]

        # Indices of k neigbours
        indices = None
        if m == "cosine":
            for j in range(0, len(row)):
                row[j] = 1 - row[j]
            # Find indices of biggest similarities
            indices = np.argpartition(row, -k)[-k:]
        elif m == "euclidean":
            # Find indices of smallest distances
            indices = np.argpartition(row, k)[:k]

        # Get probs for each class
        probs = {}
        for j in indices:
            if matrix[j][-1] in probs:
                probs[matrix[j][-1]] += 1/k
            else:
                probs[matrix[j][-1]] = 1/k

        # Fill in row of classifications matrix
        for j in range(0, len(num_to_class)):
            if j in probs:
                classifications[i][j] = probs[j]
            else:
                classifications[i][j] = 0

        # Fill in true label    
        if datatype == "test":
            classifications[i][-1] = matrix[i + n_train_docs][-1]
        elif datatype == "train":  
            classifications[i][-1] = matrix[i][-1]

    return classifications
    

# Output classification results given classification matrix
def output_classifications(data, mode):

    confusion_matrix = np.zeros((len(num_to_class), len(num_to_class)))

    with open(s_file, "a") as s:
        s.write("%%%%% " + mode + " data:\n")

        # For each document (go over classification matrix)
        for i in range(0, len(data)):
            curr_row = data[i]
            best_class = np.where(curr_row[0:-1] == np.amax(curr_row[0:-1]))[0][0]
            true_class = int(curr_row[-1])
            s.write("array:" + str(i) + " " + num_to_class[true_class] + " ")

            # List of tuples
            probs_to_sort = []
            for j in range(0, len(data[i]) - 1):
                probs_to_sort.append( (num_to_class[j], data[i][j]) )
            probs_to_sort = sorted(probs_to_sort, key=lambda x: -x[1])
            for p in probs_to_sort:
                s.write(" " + str(p[0]) + " " + str(round(p[1], 5)))
            s.write("\n")

            # Acc file using best_class
            confusion_matrix[best_class][true_class] += 1

    # Print confusion matrix
    print("Confusion matrix for the training data:\nrow is the truth, column is the system output")
    print("\t\t\t", end="")
    for i in range(0, len(confusion_matrix)):
        print(num_to_class[i] + " ", end="")
    print()
    correct = 0
    total = 0
    for i in range(0, len(confusion_matrix)):
        print(num_to_class[i] + "\t\t", end="")
        for j in range(0, len(confusion_matrix)):
            print(str(confusion_matrix[i][j]) + "\t\t\t", end="")
            if i == j:
                correct += confusion_matrix[i][j]
            total += confusion_matrix[i][j]
        print()
    print(mode + " accuracy=" + str(correct/total))
    print()


# Build matrix of training/test data
t0 = time.time()
setup_problem(train_file)
matrix = build_matrix(train_file, test_file)

# Get pairwise distances between each all docs |D|x|D| matrix
m = None
if sim_func == 1:
    m = "euclidean"
elif sim_func == 2:
    m = "cosine"

# Slice the distances matrix
distances = sklearn.metrics.pairwise_distances(matrix[:,:-1], metric=m)
d_test_train = distances[len(matrix[:,:-1]) - n_test_docs:,:n_train_docs]
d_train_train = distances[:n_train_docs, :n_train_docs]

# Classify testing documents
classifications_train = classify(matrix, d_train_train, k, m, "train")
classifications_test = classify(matrix, d_test_train, k, m, "test")

# Output
open(s_file, "w").close()
output_classifications(classifications_train, "Training")
output_classifications(classifications_test, "Test")
print("Runtime in minutes: " + str((time.time() - t0) / 60))