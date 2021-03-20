import matplotlib.pyplot as plt
from PIL import Image
from time import sleep
from IPython import display
import os
import random

def display_images_two_classes(TRAIN_PATH):
    """ 
    DESCRIPTION: This function reads in the training data
    and displays two images, one of each class. 
    
    Parameters:
    ----------
    TRAIN_PATH: the path to where the training data is stored
    
    Returns:
    ----------
    None
    """
    
    # Read in the images per class
    imgs_neg = os.listdir(TRAIN_PATH + "0/");
    imgs_pos = os.listdir(TRAIN_PATH + "1/");

    # Make the figure and the subplots
    fig, (ax0,ax1) = plt.subplots(1, 2) 
    ax0.set_title("No metastasis")
    ax1.set_title("With metastasis")

    # Choose a random image per class
    i = random.randint(0,len(imgs_pos))
    img0 = Image.open(TRAIN_PATH + '0/' + imgs_neg[i])
    img1 = Image.open(TRAIN_PATH + '1/' + imgs_pos[i])

    # Visualize these images in the figure
    ax0.imshow(img0)
    ax1.imshow(img1)

display_images_two_classes()