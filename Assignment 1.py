# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 18:36:30 2021

@author: 20183112
"""
''' Assignment 1 '''
import os
import random
from PIL import Image

directory_0 = r"C:\Users\20183112\OneDrive - TU Eindhoven\TU Eindhoven\Jaar 3\Q3 Project Imaging (8P361)\train\0" # file directory 
directory_1 = r"C:\Users\20183112\OneDrive - TU Eindhoven\TU Eindhoven\Jaar 3\Q3 Project Imaging (8P361)\train\1"

random_train_0 = random.choice(os.listdir(directory_0))
random_train_1 = random.choice(os.listdir(directory_1))

random_train_0 = random_train_0.replace("'", '') # remove ' from string
random_train_1 = random_train_1.replace("'", '') # remove ' from string 

image_0 = Image.open(directory_0 + '/' + random_train_0)
image_0.show()

image_1 = Image.open(directory_1 + '/' + random_train_1)
image_1.show()