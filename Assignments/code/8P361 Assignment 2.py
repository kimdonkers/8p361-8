"""
TU/e BME Project Imaging 2021
Simple multiLayer perceptron code for MNIST
Author: Suzanne Wetstein
"""

# Disable overly verbose tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf


# Import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard

def mnist_train_and_evaluate(model, batch_size=32, epochs=10, model_name=''):
    # Load the dataset using the builtin Keras method
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Derive a validation set from the training set
    # the original training set is split into 
    # new training set (90%) and a validation set (10%)
    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

    # The shape of the data matrix is NxHxW, where
    # N is the number of images,
    # H and W are the height and width of the images
    # keras expect the data to have shape NxHxWxC, where
    # C is the channel dimension
    X_train = np.reshape(X_train, (-1,28,28,1)) 
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))

    # Convert the datatype to float32
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize our data values to the range [0,1]
    X_train /= 255
    X_val /= 255
    X_test /= 255

    # Convert 1D class arrays to 10D class matrices
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # Use this variable to name your model
    model_name="my_first_model"

    # Create a way to monitor our model in Tensorboard
    tensorboard = TensorBoard("logs/" + model_name)

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_val, y_val), callbacks=[tensorboard])

    score = model.evaluate(X_test, y_test, verbose=0)
    return score


# Exercise 1

model1 = Sequential()
model1.add(Flatten(input_shape=(28,28,1))) 
model1.add(Dense(64, activation='relu'))
model1.add(Dense(10, activation='softmax'))
accuracy1 = mnist_train_and_evaluate(model1, model_name='Example')

model2 = Sequential()
model2.add(Flatten(input_shape=(28,28,1))) 
model2.add(Dense(64, activation='relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(10, activation='softmax'))
accuracy2 = mnist_train_and_evaluate(model2, model_name='More layers')

model3 = Sequential()
model3.add(Flatten(input_shape=(28,28,1)))
model3.add(Dense(96, activation='relu'))
model3.add(Dense(10, activation='softmax'))
accuracy3 = mnist_train_and_evaluate(model3, model_name='More neurons')

model4 = Sequential()
model4.add(Flatten(input_shape=(28,28,1)))
model4.add(Dense(96, activation='relu'))
model4.add(Dense(96, activation='relu'))
model4.add(Dense(10, activation='softmax'))
accuracy4 = mnist_train_and_evaluate(model4, model_name='Combination')

print('{:50}: {}'.format('The accuracy of the example model 1 is',accuracy1))
print('{:50}: {}'.format('The accuracy of a model with more layers is',accuracy2))
print('{:50}: {}'.format('The accuracy of a model with more neurons is',accuracy3))
print('{:50}: {}'.format('The accuracy of a model with a combination of both is',accuracy4))


# Exercise 2

model5 = Sequential()
model5.add(Flatten(input_shape=(28,28,1)))
model5.add(Dense(10, activation='softmax'))
accuracy5 = mnist_train_and_evaluate(model5, model_name='Without any hidden layers')

model6 = Sequential()
model6.add(Flatten(input_shape=(28,28,1)))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(64, activation='relu'))
model6.add(Dense(10, activation='softmax'))
accuracy6 = mnist_train_and_evaluate(model6, model_name='3 hidden layers with ReLU activations')

model7 = Sequential()
model7.add(Flatten(input_shape=(28,28,1)))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(64, activation='linear'))
model7.add(Dense(10, activation='softmax'))
accuracy7 = mnist_train_and_evaluate(model7, model_name='3 hidden layers with linear activations')

print('{:50}: {}'.format('The accuracy without any hidden layers is',accuracy5))
print('{:50}: {}'.format('The accuracy with 3 hidden layers with ReLU activations is',accuracy6))
print('{:50}: {}'.format('The accuracy with 3 hidden layers with linear activations is',accuracy7))

# Exercise 3

def mnist_train_and_evaluate_4_class(model, batch_size=32, epochs=10, model_name=''):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    y_train_new = y_train.copy()
    y_train_new[((y_train == 1) | (y_train == 7))] = 0
    y_train_new[((y_train == 0) | (y_train == 6) | (y_train == 8) | (y_train == 9))] = 1
    y_train_new[((y_train == 2) | (y_train == 5))] = 2
    y_train_new[((y_train == 3) | (y_train == 4))] = 3
    y_test_new = y_test.copy()
    y_test_new[((y_test == 1) | (y_test == 7))] = 0
    y_test_new[((y_test == 0) | (y_test == 6) | (y_test == 8) | (y_test == 9))] = 1
    y_test_new[((y_test == 2) | (y_test == 5))] = 2
    y_test_new[((y_test == 3) | (y_test == 4))] = 3
    
    y_test = y_test_new;
    y_train = y_train_new;

    X_train, X_val = train_test_split(X_train, test_size=0.10, random_state=101)
    y_train, y_val = train_test_split(y_train, test_size=0.10, random_state=101)

    X_train = np.reshape(X_train, (-1,28,28,1))
    X_val = np.reshape(X_val, (-1,28,28,1))
    X_test = np.reshape(X_test, (-1,28,28,1))

    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_val /= 255
    X_test /= 255

    y_train = to_categorical(y_train, 4)
    y_val = to_categorical(y_val, 4)
    y_test = to_categorical(y_test, 4)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(X_val, y_val))
    score = model.evaluate(X_test, y_test, verbose=0)
    return score[1]


model8 = Sequential()
model8.add(Flatten(input_shape=(28,28,1)))
model8.add(Dense(64, activation='relu'))
model8.add(Dense(4, activation='softmax'))
accuracy8 = mnist_train_and_evaluate_4_class(model8, model_name='8')

print('{:50}: {}'.format('The accuracy of a four class classification model is',accuracy8))