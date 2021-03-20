'''
TU/e BME Project Imaging 2021
Convolutional neural network for PCAM
Author: Mitko Veta
'''

# disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

# Constant definitions
DATA_PATH = '../../data/';
SAVE_PATH = '../results/';
TENSORBOARD_PATH = '../TB_logs';
IMAGE_SIZE = 96;
input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3);


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

def get_pcam_generators(base_dir, train_batch_size=32, val_batch_size=32):

     # dataset parameters
     train_path = os.path.join(base_dir, 'train')
     valid_path = os.path.join(base_dir, 'valid')

     # instantiate data generators
     datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

     train_gen = datagen.flow_from_directory(train_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=train_batch_size,
                                             class_mode='binary')

     val_gen = datagen.flow_from_directory(valid_path,
                                             target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                             batch_size=val_batch_size,
                                             shuffle=False,
                                             class_mode='binary')

     return train_gen, val_gen


# MODEL DEFINITION
input = Input(input_shape)

# get the pretrained model, cut out the top layer
pretrained = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')

# if the pretrained model it to be used as a feature extractor, and not for
# fine-tuning, the weights of the model can be frozen in the following way
# for layer in pretrained.layers:
#    layer.trainable = False

output = pretrained(input)
output = GlobalAveragePooling2D()(output)
#output = Dropout(0.5)(output) # NO DROPOUT
output = Dense(1, activation='sigmoid')(output)
model = Model(input, output)

# Compile the model
model.compile(SGD(lr=0.001, momentum=0.95), loss = 'binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Get data generators
train_gen, val_gen = get_pcam_generators(DATA_PATH)

# Save the model and define model weights save path
model_name = 'Transfer_learning_model_3'
model_filepath = os.path.join(SAVE_PATH, model_name + '.json')
weights_filepath = os.path.join(SAVE_PATH, model_name + '_weights.hdf5')

model_json = model.to_json() # serialize model to JSON
with open(model_filepath, 'w') as json_file:
    json_file.write(model_json)


# define the model checkpoint and Tensorboard callbacks
checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(os.path.join(TENSORBOARD_PATH, model_name))
callbacks_list = [checkpoint, tensorboard]


# train the model, note that we define "mini-epochs"
train_steps = train_gen.n//train_gen.batch_size#//20
val_steps = val_gen.n//val_gen.batch_size#//20

# Train the model
history = model.fit(train_gen, steps_per_epoch=train_steps,
                    validation_data=val_gen,
                    validation_steps=val_steps,
                    epochs=10,
                    callbacks=callbacks_list)

# Evaluate model and calculate score metrics
y_pred = model.predict(val_gen,verbose = 0)
y_pred_bin = (y_pred > 0.5).astype(int)

fpr, tpr, _ = roc_curve(val_gen.labels, y_pred)
auc_score = roc_auc_score(val_gen.labels, y_pred)
f1 = f1_score(val_gen.labels, y_pred_bin)
acc = accuracy_score(val_gen.labels, y_pred_bin)

# Save results
plot_ROC_curve(fpr,tpr, SAVE_PATH, model_name)
with open(os.path.join(SAVE_PATH, model_name + '_scores.txt'), 'w') as result_file:
    result_file.write('AUC score      = {}\n'.format(auc_score))
    result_file.write('F1 score       = {}\n'.format(f1))
    result_file.write('Accuracy score = {}\n'.format(acc))
