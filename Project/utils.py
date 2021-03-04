"""
8P361 - Project Imaging (BIA)
Code for project assignment
Group 8
Developed by:
    - Derikx, Pien
    - Donkers, Kim
    - Mulder, Eva
    - Vousten, Vincent

utils.py contains utility functions, for example, reading the data and saving
results.
"""

# Reduce TensorFlow verbose (disabled)
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# Load Tensorflow and Keras libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load Numpy, Matplotlib and XlsxWriter libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import xlsxwriter

# Constant definitions
IMAGE_SIZE = 96

################################################################################

def get_generators(base_dir, train_batch_size=32, val_batch_size=32):
    """
    DESCRIPTION: This function imports the data and divides it into batches.
    """
    # dataset parameters
    train_path = os.path.join(base_dir, 'train')
    valid_path = os.path.join(base_dir, 'valid')
    RESCALING_FACTOR = 1./255

    # instantiate data generators
    datagen = ImageDataGenerator(rescale=RESCALING_FACTOR)

    train_gen = datagen.flow_from_directory(train_path,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=train_batch_size,
                                         class_mode='binary')

    val_gen = datagen.flow_from_directory(valid_path,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=val_batch_size,
                                         class_mode='binary')

    val_gen_no_shuffle = datagen.flow_from_directory(valid_path,
                                         target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         batch_size=val_batch_size,
                                         shuffle=False,
                                         class_mode='binary')

    return train_gen, val_gen, val_gen_no_shuffle

################################################################################

def plot_ROC_curve(fpr, tpr, folder, model_name):
    """
    DESCRIPTION: This function plots the ROC curves of the different runs
    and saves it to a png file.

    Parameters:
    ----------
    fpr: the list with the false positive rates for every run
    tpr: the list with the true positive rates for every run
    folder: the folder to save to
    model_name: the name to be included in the filename.

    Returns
    -------
    None
    """
    # Make the figure
    fig,ax = plt.subplots()

    # Plot the ROC curves for every run and every fold
    ax.plot(fpr, tpr, color='#ffaa75',LineWidth=1)

    # Style plot and save to file
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    fig.savefig(os.path.join(folder, model_name + "_ROC_curve"))

################################################################################

def plot_history(history, folder, model_name):
    """
    DESCRIPTION: This function plots the accuracies and losses of the model and
    saves this to two images files.

    Parameters:
    ----------
    history: the history dictionary resulting from model.fit
    folder: the folder to save to
    model_name: the name to be included in the filename.

    Returns
    -------
    None
    """
    # Make the first figure
    fig1, ax1 = plt.subplots()

    # Plot the accuracy curves
    ax1.plot(history['val_accuracy'], color='orange', linewidth=1, label='Validation accuracy')
    ax1.plot(history['accuracy'], color='royalblue', linewidth=1, label='Training accuracy')

    # Style plot and save to file
    ax1.set_xlim([0, len(history['val_accuracy'])-1])
    ax1.set_ylim([0,1])
    ax1.set_xlabel('Nr of epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig1.savefig(os.path.join(folder, model_name + "_accuracy_history"), bbox_inches='tight')

    # Make the second figure
    fig2, ax2 = plt.subplots()

    # Plot the accuracy curves
    ax2.plot(history['val_loss'], color='orange', linewidth=1, label='Validation loss')
    ax2.plot(history['loss'], color='royalblue', linewidth=1, label='Training loss')

    # Style plot and save to file
    ax2.set_xlim([0, len(history['val_loss'])-1])
    ax2.set_ylim([0, 1])
    ax2.set_xlabel('Nr of epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig2.savefig(os.path.join(folder, model_name + "_loss_history"), bbox_inches='tight')

################################################################################

def save_history(history, model_name, save_folder='./'):
    """
    DESCRIPTION:
    -----------
    Write the accuracies and losses after every epoch to an xlsx-file.

    Parameters
    ----------
    history:     The history dictionary, resulting from model.fit
    save_folder: The folder in which the Xlsx-file is saved.

    Returns
    -------
    None.

    """
    # Get the datetime for an overall filename
    filename = os.path.join(save_folder, model_name + '_history.xlsx');


    # Save the loss/accuracy history to a xlsx file
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    worksheet.write('A1', 'epoch')
    worksheet.write('B1', 'val_loss')
    worksheet.write('C1', 'val_acc')
    worksheet.write('D1', 'train_loss')
    worksheet.write('E1', 'train_acc')
    line = 1;
    for i in range(len(history['val_loss'])):
        worksheet.write(line, 0, i+1)
        worksheet.write(line, 1, history['val_loss'][i])
        worksheet.write(line, 2, history['val_accuracy'][i])
        worksheet.write(line, 3, history['loss'][i])
        worksheet.write(line, 4, history['accuracy'][i])
        line += 1;
    workbook.close()

################################################################################
