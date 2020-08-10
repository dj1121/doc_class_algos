# Author: Devin Johnson

import sys
import numpy as np
import math
import time

# Files
train_file = sys.argv[1]

# Relevant problem info (vocab, classes, etc)
word_to_index = {}
index_to_word = {}
class_to_num = {}
num_to_class = {}

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


# Build a numpy matrix from the data in a given file
def build_matrix(file):
    matrix = []
    with open(file) as f:
        # For each document
        for line in f:
            # Build a new row
            curr_row = []
            s = line.split(" ")
            # Sentences have format: label word1:count word2:count
            doc_words = set()
            for i in range(1, len(s)):
                curr_word = s[i].split(":")[0]
                doc_words.add(curr_word)
            
            # For word in vocab, see if doc has or does not have it
            for i in range(0, len(index_to_word)):
                if index_to_word[i] in doc_words:
                    curr_row.append(1)
                else:
                    curr_row.append(0)
            
            # Append label to end
            curr_row.append(class_to_num[s[0]])

            # Append current row to matrix
            matrix.append(curr_row)
    
    return np.array(matrix)


def calc_chi(matrix):

    chi_results = np.zeros((len(word_to_index), 3))

    # For each feature
    for j in range(0, len(word_to_index)):
        # Rows = f_k and !f_k
        # Columns = classes
        contingency = np.zeros((2, len(num_to_class)))
        curr = matrix[:,j]
        
        for i in range(0, len(curr)):            
            if curr[i] == 1:
                contingency[1][matrix[i][-1]] += 1
            else:
                contingency[0][matrix[i][-1]] += 1
        
        # Loop over all cells in contingency table
        chi_square = 0
        total = len(matrix)
        for i in range(0, len(contingency)):
            row_total = np.sum(contingency[i])
            for k in range(0, len(contingency[i])):
                observed = contingency[i][k]
                col_total = np.sum(contingency[:,k])
                expected = (row_total * col_total) / total
                chi_square += math.pow((observed - expected), 2)/expected
        
        chi_results[j][0] = j
        chi_results[j][1] = chi_square
        chi_results[j][2] = np.sum(contingency[1])
    
    # Sort the matrix descending by chi score
    chi_results = chi_results[chi_results[:,1].argsort()[::-1]]
    return chi_results


def output_results(chi_results):
    for row in chi_results:
        print(index_to_word[row[0]] + " " + str(round(row[1], 5)) + " " + str(row[2]))



t0 = time.time()
setup_problem(train_file)
matrix = build_matrix(train_file)
chi_results = calc_chi(matrix)
output_results(chi_results)
print("Runtime in minutes: " + str((time.time() - t0) / 60))