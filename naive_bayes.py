# Author: Devin Johnson

import sys
import time
import math
import numpy as np


# Files
train_file = sys.argv[1]
test_file = sys.argv[2]
prior_d = float(sys.argv[3])
cond_d = float(sys.argv[4])
m_file = sys.argv[5]
s_file = sys.argv[6]

# Relevant problem info (vocab, classes, etc)
word_to_index = {}
index_to_word = {}
class_to_num = {}
num_to_class = {}

# Parameters of model
prior_probs = []
cond_probs = []
one_min_cond_probs = []


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
            

# Separate data by class
def separate_by_class(data):
	separated = dict()
	for i in range(len(data)):
		vector = data[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated


# Learn parameters for NB from training data
def train(data):
    ## Get P(c) for each class ##
    global prior_probs
    n_docs = np.sum(data[:,-1])
    class_counts = np.unique(data[:,-1], return_counts = True)
    for i in range(0, len(class_counts[0])):
        # (prior_d + # docs class c) / (# classes + total # docs)
        prior_probs.append((prior_d + class_counts[1][i]) / (len(class_counts[0]) + n_docs))
    prior_probs = np.array(prior_probs)

    ## Get cond probs for each class (for each word, each class) ##
    # Dictionary where key = class, val = data matrix
    global cond_probs
    global one_min_cond_probs
    separated_data = separate_by_class(data)
    # For each subset of the data (by class)
    for i in range(0, len(separated_data)):
         # Get sums of columns (counts of each word per this class)
        subset = separated_data[i]
        counts = np.sum(subset, axis=0)
        cond_probs.append([])
        one_min_cond_probs.append([])

        # For each feature count
        for j in range(0, len(counts) - 1):
            curr_feat_count = counts[j]
            cond_probs[i].append((cond_d + curr_feat_count) / ((2*cond_d + len(subset))))
            one_min_cond_probs[i].append(1 - ((cond_d + curr_feat_count) / ((2*cond_d + len(subset)))))

    cond_probs = np.array(cond_probs)
    one_min_cond_probs = np.array(one_min_cond_probs)

    # Print model information to model file
    with open(m_file, "a") as model:
        # Printout to model file
        model.write("%%%%% prior prob P(c) %%%%%\n")
        for i in range(0, len(prior_probs)):
            model.write(num_to_class[i] + " " + str(prior_probs[i]) + "\t" + str(math.log(prior_probs[i], 10)) + "\n")
        model.write("%%%%% conditional prob P(f|c) %%%%%\n")
        for i in range(0, len(cond_probs)):
            model.write("%%%%% conditional prob P(f|c) c=" + num_to_class[i] + "%%%%%\n")
            for j in range(0, len(cond_probs[i])):
                model.write(index_to_word[j] + " " + num_to_class[i] + " " + str(cond_probs[i][j]) + " " + str(math.log(cond_probs[i][j], 10)) + "\n")



# Classify data using the model
def classify(data):

    # Rows = docs
    # Cols = classes (+1 for true class)
    classifications = np.zeros((len(data), len(prior_probs) + 1))

    # For each doc, calculate P(d|c)
    for i in range(0, len(data)):
        for j in range(0, len(prior_probs)):
            # Term 1 P(c)
            term1 = math.log(prior_probs[j], 10)
            
            # Term 2 Sum log(P(w|c)) for w in doc
            doc_vector = data[i][0:-1]
            cond_prob_vector = cond_probs[j]
            one_min_vector = one_min_cond_probs[j]
            prob_vector = np.log10(np.divide(cond_prob_vector, one_min_vector))
            term2 = np.dot(doc_vector, prob_vector)

            # Term 3 Sum log(1-P(w|c))
            # term3 = np.sum(np.log10(one_min_cond_probs[j]))
            # Seems to result in less accuracy?

            # P(d|c)
            p = term1 + term2
            classifications[i][j] = p
        
        classifications[i][-1] = data[i][-1]

    return classifications


# Output classification results given classification matrix
def output_classifications(data, mode):

    confusion_matrix = np.zeros((len(prior_probs), len(prior_probs)))

    with open(s_file, "a") as s:
        s.write("%%%%% " + mode + " data:\n")

        # For each document (go over classification matrix)
        for i in range(0, len(data)):
            curr_row = data[i]
            best_class = np.where(curr_row[0:-1] == np.amax(curr_row[0:-1]))[0][0]
            true_class = int(curr_row[-1])
            s.write("array:" + str(i) + " " + num_to_class[best_class] + " ")

            for j in range(0, len(data[i]) - 1):
                s.write(num_to_class[j] + " " + str(data[i][j]) + " ")
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



# Train
t0 = time.time()
setup_problem(train_file)
training_data = build_matrix(train_file)
open(m_file, "w").close()
train(training_data)

# Classify
test_data = build_matrix(test_file)
training_classifications = classify(training_data)
testing_classifications = classify(test_data)

# Output
open(s_file, "w").close()
output_classifications(training_classifications, "training")
output_classifications(testing_classifications, "testing")
print("Runtime in minutes: " + str((time.time() - t0) / 60))
