# Author: Devin Johnson

import sys
import time
import math
import numpy as np

# Used to build our tree
class Node(object):
    def __init__(self, data, parent):
        self.data = data
        self.left = None
        self.right = None
        self.parent = parent
        self.split_attribute = None
        self.label = None
        self.label_counts = {}

    def is_left_child(self):
        return self is self.parent.left

    def is_right_child(self):
        return self is self.parent.right

    def __str__(self):
        return "NODE \n" + "Data: " + str(self.data) + "\n" + "Split attribute: " + str(self.split_attribute) + "\nExists Parent: " + str(self.parent != None) + \
                "\nExists Left Child: " + str(self.left !=None) + "\nExists Right Child: " + str(self.right != None) + "\nLabel (if leaf): " + str(self.label)


# Global parameters of model
train_vocab = {}
vocab = set()
labels = {}
labels_inverse = {}
leaf_nodes = []
num_train_docs = 0
num_test_docs = 0
max_depth = int(sys.argv[3])
min_gain = float(sys.argv[4])


# Build a vocabulary that stores all positions in alphabetical order and corresponding words
# Position will be used later for making vectors of documents
def build_training_vocab():

    global labels
    global train_vocab
    global num_train_docs
    global num_test_docs
    global train_line_labels
    global vocab
    
    vocab = set([])

    # Make the vocab a set
    with open(sys.argv[1]) as file:
        for line in file:
            s = line.split()
            if s[0] not in labels:
                labels[s[0]] = len(labels)
                labels_inverse[len(labels_inverse)] = s[0]
            for i in range(1, len(s)):
                vocab.add(s[i].split(":")[0])
            num_train_docs += 1

        sort = sorted(list(vocab))
        for i in range(0, len(sort)):
            if sort[i] not in train_vocab:
                train_vocab[sort[i]] = i
                vocab.add(sort[i])

        
    # As an extra step, get number of test docs
    with open(sys.argv[2]) as test:
        for line in test:
            num_test_docs += 1


# Build document training vectors (vectors are dictionaries here)
def build_matrix(vocab, phase):
    
    global num_train_docs
    global num_test_docs

    num_docs = -1
    arg = -1
    if phase == "train":
        arg = 1
        num_docs = num_train_docs
    elif phase == "test":
        arg = 2
        num_docs = num_test_docs

    # Create a matrix (num_docs * num_features (+1 for class label))
    data = np.zeros((num_docs, len(train_vocab) + 1))

    # Make a vector for each document, add as row to matrix
    with open(sys.argv[arg]) as file:
        line_count = 0
        for line in file:
            # Add 1s where necessary and class label at end
            s = line.split()
            data[line_count][-1] = labels[s[0]]
            for i in range(1,len(s)):
                curr_word = s[i].split(":")[0]
                # If word in vocab (sometimes test words may not be in vocab)
                if curr_word in vocab:
                    data[line_count][vocab[curr_word]] = 1
            line_count += 1
        
    return data


# Get the entropy of some data
def entropy(data):

    # Get the counts of each label
    label_counts = np.unique(data, return_counts=True)[1]
    
    # Get probabilities of labels (how many of label x / total in column)
    label_probs = np.array([count / len(data) for count in label_counts])
    
    # Dot the probs (a column vector x) with log base 2 of the probs (this is entropy)
    # -(x1log2x1 + x2log2x2 + x3log2x3)....
    return -label_probs.dot(np.log2(label_probs))


# Calculate info gain for an attribute given a set
def info_gain(attribute, data):
    index = train_vocab[attribute]

    x = data[:, index]
    y = data[:, -1]
    z = np.sum([x,y * 2], axis = 0)


    return entropy(x) + entropy(y) - entropy(z)


# Given a set (data at a node), find which attribute to split on and split
def split_data(node):
    # print("called find best split - " + "num docs: " + str(len(node.data)))
    max_info_gain = 0
    best = None

    # Test each attribute (word) i (len(training_data[0]) - 1 = |V|)
    t0 = time.time()
    for attribute in vocab:
        curr_info_gain = info_gain(attribute, node.data)
        if curr_info_gain > max_info_gain:
            max_info_gain = curr_info_gain
            best = attribute

    if max_info_gain < min_gain:
        return None
    
    # print("time to find best attribute: " +  str(time.time() - t0))

    # Remove attribute
    if best != None:
        vocab.remove(best)

    return best


# Set the label of leaf nodes after built tree
def set_leaf_labels():
    for leaf in leaf_nodes:
        # Get counts of labels at leaf
        label_counts = {}
        if len(leaf.data) != 0:
            for i in range(0, len(leaf.data)):
                if leaf.data[i][-1] in label_counts:
                    label_counts[leaf.data[i][-1]] += 1
                else:
                    label_counts[leaf.data[i][-1]] = 1
    
        # Make sure to include counts for labels with 0 docs
        for label_num in labels_inverse:
            if label_num not in label_counts:
                label_counts[label_num] = 0

        # Set the label for the leaf node
        leaf_label = None
        max_count = -1
        for key in label_counts:
            if label_counts[key] > max_count:
                leaf_label = key
                max_count = label_counts[key]
        
        

        leaf.label = labels_inverse[leaf_label]
        leaf.label_counts = label_counts


# Build a decision tree
def build_tree(curr_node, max_depth, min_gain, curr_depth):

    # print("depth: " + str(curr_depth))

    global leaf_nodes

    if curr_depth == max_depth or len(curr_node.data) == 1:
        leaf_nodes.append(curr_node)
        return
    
    # Find best split, if none then stop
    best_attribute = split_data(curr_node)
    if best_attribute == None:
        leaf_nodes.append(curr_node)
        return
    index = train_vocab[best_attribute]


    # Build left and right children
    curr_node.split_attribute = best_attribute
    left = curr_node.data[curr_node.data[:, index] == 1]
    right = curr_node.data[curr_node.data[:, index] == 0]


    # Construct children
    if len(left) != 0:
        curr_node.left = Node(left, curr_node)
        build_tree(curr_node.left, max_depth, min_gain, curr_depth + 1)
    if len(right) != 0:
        curr_node.right = Node(right, curr_node)
        build_tree(curr_node.right, max_depth, min_gain, curr_depth + 1)

    # Clear the non-leaf nodes of data (to avoid duplication)
    curr_node.data = None

    # Non leaf nodes
    return curr_node


# Clear out the leaf nodes of the tree (clear all data, but keep labels learned from training)
def clear_tree_data():
    global leaf_nodes
    for leaf in leaf_nodes:
        leaf.data = None


# Output training stats to files
def output_training():
    
    # Clear previous files
    open(sys.argv[5], "w").close()
    open(sys.argv[6], "w").close()

    

    # Write to model file
    with open(sys.argv[5], "a") as model_file, open(sys.argv[6], "a") as sys_file:
        
        # Headers
        sys_file.write("%%%%% training data:\n")
        print("Confusion matrix for the training data:")
        print("row is the truth, column is the system output")
        print("\t\t\t", end = "")
        for label in sorted(labels):
            print(label + " ", end = "")
        print()

        confusion = {}
        doc_counter = 0
        
        # Go through all leaf nodes
        for leaf in leaf_nodes:
            curr = leaf

            # Write path from leaf to root in model file (based on features split)
            while curr.parent != None:
                # If this is the left child
                if curr.is_left_child():
                    model_file.write(str(curr.parent.split_attribute))
                # If this is the right child
                elif curr.is_right_child():
                    model_file.write("!" + str(curr.parent.split_attribute))
                # Update curr
                curr = curr.parent
                if curr.parent != None:
                    model_file.write("&")

            # Write number of examples at leaf node and distribution at leaf
            model_file.write(" " + str(len(leaf.data)) + " ")
            for label in leaf.label_counts:
                model_file.write(str(labels_inverse[label]) + " " + str(leaf.label_counts[label] / len(leaf.data)) + " ")

            # Write to sys file
            for doc in leaf.data:
                sys_file.write("array:" + str(doc_counter) + " ")
                doc_counter += 1
                for label in leaf.label_counts:
                    sys_file.write(str(labels_inverse[label]) + " " + str(leaf.label_counts[label] / len(leaf.data)) + " ")
                sys_file.write("\n")
                

            # Get confusion data
            for i in range(0, len(leaf.data)):
                my_label = leaf.label
                system_label = labels_inverse[leaf.data[i][-1]]

                if (my_label, system_label) in confusion:
                    confusion[(my_label, system_label)] += 1
                else:
                    confusion[(my_label, system_label)] = 1
                    
            # New line for each leaf
            model_file.write("\n")

    
    # Fill in rest = 0 for confusion matrix
    for label_x in labels:
        for label_y in labels:
            if (label_x, label_y) not in confusion:
                confusion[(label_x, label_y)] = 0
    
    
    # Print confusion matrix
    accurate_count = 0
    total_count = 0
    for label_x in sorted(labels):
            print(label_x, end = "")
            for label_y in sorted(labels):
                if label_x == label_y:
                    accurate_count += confusion[(label_x, label_y)]
                total_count += confusion[(label_x, label_y)]
                print("\t" + str(confusion[(label_x, label_y)]), end = "")
            print()
    
    print("\nTraining accuracy: " + str(accurate_count / total_count))
      

# Output testing stats
def output_testing():
    with open(sys.argv[6], "a") as sys_file:

        # Header
        sys_file.write("\n\n %%%%% test data:\n")
        print("\n\nConfusion matrix for the training data:")
        print("row is the truth, column is the system output")
        print("\t\t\t", end = "")
        for label in sorted(labels):
            print(label + " ", end = "")
        print()

        confusion = {}
        doc_counter = 0

        for leaf in leaf_nodes:
             # Get confusion data
            if leaf.data != None:
                for i in range(0, len(leaf.data)):
                    my_label = leaf.label
                    system_label = labels_inverse[leaf.data[i][-1]]

                    if (my_label, system_label) in confusion:
                        confusion[(my_label, system_label)] += 1
                    else:
                        confusion[(my_label, system_label)] = 1

            # Write to sys file
            if leaf.data != None:
                for doc in leaf.data:
                    sys_file.write("array:" + str(doc_counter) + " ")
                    doc_counter += 1
                    for label in leaf.label_counts:
                        sys_file.write(str(labels_inverse[label]) + " " + str(leaf.label_counts[label] / len(leaf.data)) + " ")
                    sys_file.write("\n")
        
        
        # Fill in rest = 0 for confusion matrix
        for label_x in labels:
            for label_y in labels:
                if (label_x, label_y) not in confusion:
                    confusion[(label_x, label_y)] = 0
        

        # Print confusion matrix
        accurate_count = 0
        total_count = 0
        for label_x in sorted(labels):
                print(label_x, end = "")
                for label_y in sorted(labels):
                    if label_x == label_y:
                        accurate_count += confusion[(label_x, label_y)]
                    total_count += confusion[(label_x, label_y)]
                    print("\t" + str(confusion[(label_x, label_y)]), end = "")
                print()
        
        print("\nTesting accuracy: " + str(accurate_count / total_count))


# Given test data, classify it using the tree which has been built
def classify(documents, root):
    

    # For each test vector (document)
    for document in documents:
        curr_node = root
        while curr_node.split_attribute != None:
            split_on = curr_node.split_attribute
            if document[train_vocab[split_on]] == 1:
                curr_node = curr_node.left
            else:
                curr_node = curr_node.right

        if curr_node.data == None:
            curr_node.data = [document]
        else:
            curr_node.data.append(document)

        if document[-1] in curr_node.label_counts:
            curr_node.label_counts[document[-1]] += 1
        else:
            curr_node.label_counts[document[-1]] = 1


# Time stats
t0 = time.time()

# Training
build_training_vocab()
root = Node(build_matrix(train_vocab, "train"), None)

build_tree(root, max_depth, min_gain, 0)
set_leaf_labels()
output_training()

# Testing
clear_tree_data()
test_documents = build_matrix(train_vocab, "test")
classify(test_documents, root)
output_testing()


# Time stats
t1 = time.time()
total = (t1-t0) / 60
print("\n\nruntime: " + str(total))