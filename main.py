import pandas as pd
import numpy as np
import math
import random 
import os 

directory_path = "output"
quality = 130 

# Check if the directory exists
if not os.path.exists(directory_path):
    # Create the directory
    os.makedirs(directory_path)

if __name__=="__main__": 
    
    # part a
    training_path = os.path.join(directory_path, "Corona_train.csv")
    validation_path = os.path.join(directory_path, "Corona_validation.csv")

    # part a
    print(50*'-',"Part A", 50*'-')
    with open("output/q1/a.txt", 'w') as file:
        pass

    training_alogrithm_accuracy, training_confusion_matrix = part_a(training_path, training_path, "Training", "a", True)
    validation_alogrithm_accuracy, validation_confusion_matrix = part_a(training_path, validation_path, "Validation", "a", True)
