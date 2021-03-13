import models
import main

from tensorflow.keras.optimizers import SGD, Adam


SAVE_PATH   = '../../results/'

######################## STATIC LEARNING RATE ##################################
#model1 = models.CNN_01(32, 64, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model1, 'model_01', save_folder = SAVE_PATH)

#model2 = models.CNN_02(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model2, 'model_02', save_folder = SAVE_PATH)

#model3 = models.CNN_03(32, 64, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model3, 'model_03', save_folder = SAVE_PATH)

#model4 = models.CNN_04(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model4, 'model_04', save_folder = SAVE_PATH)

#FCN = models.FCN_05(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(FCN, 'model_05', save_folder = SAVE_PATH)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model6, 'model_06', save_folder = SAVE_PATH)

####################### LEARNING RATE DECAY ####################################
#model1 = models.CNN_01(32, 64, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model1, 'model_01_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model2 = models.CNN_02(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model2, 'model_02_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model3 = models.CNN_03(32, 64, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model3, 'model_03_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model4 = models.CNN_04(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model4, 'model_04_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#FCN = models.FCN_05(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(FCN, 'model_05_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95)
#main.train_and_evaluate(model6, 'model_06_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

######################## ADAM OPTIMIZER ########################################
# NOTE: Adam has a lower (starting) learning rate than SGD, this was based on
# the default value for the Adam optimizer.
model1 = models.CNN_01(32, 64, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(model1, 'model_01_Adam', save_folder = SAVE_PATH)

model2 = models.CNN_02(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(model2, 'model_02_Adam', save_folder = SAVE_PATH)

model3 = models.CNN_03(32, 64, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(model3, 'model_03_Adam', save_folder = SAVE_PATH)

model4 = models.CNN_04(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(model4, 'model_04_Adam', save_folder = SAVE_PATH)

FCN = models.FCN_05(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(FCN, 'model_05_Adam', save_folder = SAVE_PATH)

model6 = models.CNN_06(32, 64, 128, 256, optimizer=Adam, lr=0.001, momentum=None)
main.train_and_evaluate(model6, 'model_06_Adam', save_folder = SAVE_PATH)
