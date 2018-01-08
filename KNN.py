# neha_bhagwat_knn.py
# Author: Neha Bhagwat (SJSU ID: 012412140)
# Last updated: September 9th, 2017
# Purpose: To implement the k-nearest neighbours algorithm
#
#

import csv
import math
import re
import os
try:
    import pygraph
    import matplotlib
    import matplotlib.pyplot as pplot
    import numpy as np
except:
    print("The graphs will not be created as the required libraries could not be imported.")


# Method to calculate the Euclidean distance between the percept and the training data
def calculate_distance(percept, training_data):
    distance = 0
    for i in range(0,len(percept)-1):
        distance = distance + math.pow((percept[i] - training_data[i]),2)
    return distance


# Class: Agent
class agent:
    # Constructor function for class agent
    # Sensor function of the agent class
    # Accepts input arguments from the environment and passes them on to the instance of agent
    def __init__(self, training_data, num_of_attributes, k, class_list):
        self.training_data = training_data
        self.num_of_attributes = num_of_attributes
        self.k = k
        self.class_list = class_list

    # Function: Predict class
    # Arguments: closest_classes (list of the k class types closest to the percept), class_list (list
    #            of classes detected from the @attribute line of the input file that contains a list
    #            of possible classes
    # Functions counts the number of occurences of each class type in the list closest_classes. The class
    # which occurs maximum number of times in the list is returned
    # This function acts as the ACTUATOR component of the class agent. 
    def predict_class(self, closest_classes, class_list):
        class_count = []
        for i in range(0,len(class_list)):
            class_count.append(0)
        for i in range(0,len(closest_classes)):
            for j in range(0,len(class_list)):
                if closest_classes[i] == class_list[j]:
                    class_count[j] = class_count[j] + 1
        max_count = max(class_count)
        for i in range(0,len(class_list)):
            if class_count[i] == max_count:
                return(class_list[i])


    # Function: test_percept
    # Argument: percept_data
    # The percept is compared with every entry in the training set. The distance between each entry and
    # the percept is calculated using the calculate_distance function.
    # This function acts as the FUNCTION component of the class agent. It finds the k closest neighbours
    # for each percept.
    def test_percept(self, percept_data):
        self.percept_data = percept_data

        closest_samples = []
        closest_index = []
        
        max_distance = 0
        distances = []

        # For loop to calculate the distance between percept and the 'i' entry in the training set
        # All the distances are stored in a list named distances 
        for i in range(0,len(self.training_data)):
            distance = calculate_distance(self.percept_data, self.training_data[i])
            distances.append(distance)

        # Initializing the lists closest_samples and closest_index to find the k smallest values
        # in the list distances later
        for i in range(0, self.k):
            closest_samples.append(0)
            closest_index.append(0)

        closest_classes = []

        # Save the list of k closest samples (samples with minimum distance) in the training data
        # to the list closest_classes
        for i in range(0, self.k):
            closest_samples[i] = min(distances)
            closest_index[i] = distances.index(min(distances))
            closest_classes.append(self.training_data[closest_index[i]][self.num_of_attributes-1])
            distances.remove(min(distances))
        
        return(self.predict_class(closest_classes, self.class_list))
    
# Class: Environment
class environment:

    # Function: initialize_files
    # Takes as input all the data in the training or test file, removes the header contents in the files
    # Separates the rest of the data to create a list, returns the list
    def initialize_files(train_or_test_data):
        new_training_data = []
        header_count = 0
        for line in train_or_test_data:
            if line[0]=='@':
                header_count = header_count + 1
            else:
                # print(line)
                new_training_data.append(line)
        num_of_attributes = header_count - 4

        # Read the lines as comma separated data
        training_data = []
        for line in range(0,len(new_training_data)):
            training_data.append(new_training_data[line].split(','))

        # Convert each element of the data from strings to float values
        for element in training_data:
            for i in range(0,num_of_attributes):
                element[i] = float(element[i].rstrip(' ').rstrip('\n'))
        # print(training_data[0])
        return training_data

    # Function: count_num_of_attributes
    # Takes as input arguments all the data in the training or test files, extracts the header contents,
    # Counts the number of attributes based on the number of lines in the header contents
    # Returns the number of attributes
    def count_num_of_attributes(train_or_test_data):
        header_count = 0
        for line in train_or_test_data:
            if line[0]=='@':
                header_count = header_count + 1
            num_of_attributes = header_count - 4
        return num_of_attributes

    # Function: find_class_list
    # Takes as input the training data and the number of attributes
    # Finds the header line with the class list in the header content
    # Splits the classes into a list and returns the list
    def find_class_list(training_file_data, num_of_attributes):
        class_list_line = training_file_data[num_of_attributes]
        # print(class_list_line)
        class_list = (class_list_line[class_list_line.find('{')+1:class_list_line.find('}')]).split(',')
        for i in range (0, len(class_list)):
            class_list[i] = float(class_list[i])
        return class_list
    
    # Set the value of k to an odd number
    k = 1
    max_k = 10
    # Input the name of the folder that contains the training and test data
    print("Enter the name of the folder which contains the training and test data.")
    folder_path = raw_input()
    
    list_of_files = os.listdir(folder_path)

    # print(list_of_files)
    training_file_path = ""
    testing_file_path = ""
    
    
    accuracy_list = []
    file_number_list = []
    accuracy_comprehensive = []
    files_comprehensive = []
    k_list = []
    for k in range(1,max_k,2):
        colours_list = ['purple', 'blue', 'green', 'red', 'black']
        color_count = 0
        net_accuracy = 0
        print("\nk = " + str(k) + "\n")
        for file in list_of_files:
            found = re.search(".*(?=-10-[0-9]+tra)",file,re.M|re.I)
            if found:
                if(file[0] != '~'):
                    # training_file_path = folder_path + "\\" + file
                    training_file_path = os.path.join(os.sep, folder_path, file)
                    testing_file_path = training_file_path[0:(len(training_file_path)-7)] + "tst.dat"

                    file_number_list.append(int(training_file_path[len(training_file_path)-8])+1)
                    training_file = open(training_file_path)
                    # print(training_file_path)
                    training_file_data = training_file.readlines()
                    num_of_attributes = count_num_of_attributes(training_file_data)
                    class_list = find_class_list(training_file_data, num_of_attributes)
                    # print(class_list)
                    training_data = initialize_files(training_file_data)
                        
                    # Prepare testing data for agent
                    testing_file = open(testing_file_path)
                    testing_file_data = testing_file.readlines()
                    testing_data = initialize_files(testing_file_data)
                    correct_prediction = 0
                    x = agent(training_data, num_of_attributes, k, class_list)
                        
                    for i in range(0,len(testing_data)):
                        # Finding the predicted class for a test percept from the agent class
                        predicted_class = x.test_percept(testing_data[i])
                        # Extracting the actual class of a test percept from the test data
                        actual_class = testing_data[i][num_of_attributes-1]

                        # If predicted class and actual class are the same, increment the number of correct predictions
                        if predicted_class == actual_class:
                            correct_prediction = correct_prediction + 1
                                
                    # Calculating accuracy based on the number of correct predictions of the test data
                    accuracy = (correct_prediction * 100)/len(testing_data)
                    accuracy_list.append(accuracy)
                    net_accuracy = net_accuracy + accuracy
                    print("Accuracy for file " + str(file) + ": " + str(accuracy))
                    training_file.close()
                    testing_file.close()
        try:
            pplot.plot(file_number_list, accuracy_list, color = 'grey' , marker = 'o', markersize = '2')
            pplot.axis([0,10,0,100])
            pplot.xlabel('File Numbers')
            pplot.ylabel('Accuracies (in %)')
            pplot.title("Plot of Accuracy for different files for the same value of k, where k = " + str(k))
            pplot.savefig(str(k)+'_graph')
            pplot.cla()
            pplot.clf()
                
        except Exception as e:
            print("Graphs not created.")
            print(e)
        
        # Calculating net accuracy for 10 training and test files
        net_accuracy = net_accuracy/10
        print("\nNet Accuracy for the data set with k = " + str(k) + ": " + str(net_accuracy) + "\n")
        accuracy_comprehensive.append(accuracy_list)
        files_comprehensive = file_number_list
        accuracy_list = []
        file_number_list = []
    try:
        list_count = 0
        legend_list = []
        k_count = 1
        for list_count in range (0,5):
            print(len(accuracy_comprehensive[list_count]))
            print(len(files_comprehensive))
            print(len(colours_list))
            pplot.plot(files_comprehensive, accuracy_comprehensive[list_count], color = colours_list[list_count], marker = 'o', markersize = '2')
            pplot.axis([0,10,0,100])
            pplot.xlabel('File Numbers')
            pplot.ylabel('Accuracies (in %)')
            pplot.title("Comprehensive plot of accuracies for different files for each value of k")
            legend_list.append('k = ' + str(k_count))
            k_count = k_count + 2
            print(legend_list)
            pplot.legend(legend_list)
            pplot.savefig('net_accuracies')
    except Exception as e:
        print("Comprehensive graph not created.")
        print(e)
            
