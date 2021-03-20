import models
import main
import os
from tensorflow.keras.optimizers import SGD, Adam


SAVE_PATH   = '../../results/'

######################## STATIC LEARNING RATE ##################################
#model1 = models.CNN_01(32, 64, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model1, 'model_01', save_folder = SAVE_PATH)

#model2 = models.CNN_02(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model2, 'model_02', save_folder = SAVE_PATH)

#model3 = models.CNN_03(32, 64, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model3, 'model_03', save_folder = SAVE_PATH)

#model4 = models.CNN_04(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model4, 'model_04', save_folder = SAVE_PATH)

#FCN = models.FCN_05(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(FCN, 'model_05', save_folder = SAVE_PATH)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model6, 'model_06', save_folder = SAVE_PATH)

####################### LEARNING RATE DECAY ####################################
#model1 = models.CNN_01(32, 64, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model1, 'model_01_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model2 = models.CNN_02(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model2, 'model_02_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model3 = models.CNN_03(32, 64, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model3, 'model_03_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model4 = models.CNN_04(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model4, 'model_04_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#FCN = models.FCN_05(32, 64, 128, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(FCN, 'model_05_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0)
#main.train_and_evaluate(model6, 'model_06_LRDecay', save_folder = SAVE_PATH, adaptive_LR=True)

######################## ADAM OPTIMIZER ########################################
# NOTE: Adam has a lower (starting) learning rate than SGD, this was based on
# the default value for the Adam optimizer.
#model1 = models.CNN_01(32, 64, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(model1, 'model_01_Adam', save_folder = SAVE_PATH)

#model2 = models.CNN_02(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(model2, 'model_02_Adam', save_folder = SAVE_PATH)

#model3 = models.CNN_03(32, 64, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(model3, 'model_03_Adam', save_folder = SAVE_PATH)

#model4 = models.CNN_04(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(model4, 'model_04_Adam', save_folder = SAVE_PATH)

#FCN = models.FCN_05(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(FCN, 'model_05_Adam', save_folder = SAVE_PATH)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0)
#main.train_and_evaluate(model6, 'model_06_Adam', save_folder = SAVE_PATH)


###################### TRANSFER LEARNING ########################################
# Note: no further experiments were done concerning learning rate, batch size etc.
# Only freezing and using a pretrained model was evaluated


#TL_MBN_01 = models.transfer_MobileNetV2(pretrained_weights='imagenet')
#main.train_and_evaluate(TL_MBN_01, 'TL_MBN_01', save_folder = SAVE_PATH)

#TL_MBN_02 = models.transfer_MobileNetV2(pretrained_weights=None)
#main.train_and_evaluate(TL_MBN_02, 'TL_MBN_02', save_folder = SAVE_PATH)

#TL_MBN_03 = models.transfer_MobileNetV2(pretrained_weights='imagenet', freeze=True)
#main.train_and_evaluate(TL_MBN_03, 'TL_MBN_03', save_folder = SAVE_PATH)

#TL_INC_01 = models.transfer_InceptionV3(pretrained_weights='imagenet')
#main.train_and_evaluate(TL_INC_01, 'TL_INC_01', save_folder = SAVE_PATH)

#TL_INC_02 = models.transfer_InceptionV3(pretrained_weights=None)
#main.train_and_evaluate(TL_INC_02, 'TL_INC_02', save_folder = SAVE_PATH)

#TL_INC_03 = models.transfer_InceptionV3(pretrained_weights='imagenet', freeze=True)
#main.train_and_evaluate(TL_INC_03, 'TL_INC_03', save_folder = SAVE_PATH)




######################## NOW WITH DROPOUT ###########################################
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
#model1 = models.CNN_01(32, 64, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(model1, 'model_01_Adam', save_folder = SAVE_PATH)

#model2 = models.CNN_02(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(model2, 'model_02_Adam', save_folder = SAVE_PATH)

#model3 = models.CNN_03(32, 64, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(model3, 'model_03_Adam', save_folder = SAVE_PATH)

#model4 = models.CNN_04(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(model4, 'model_04_Adam', save_folder = SAVE_PATH)

#FCN = models.FCN_05(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(FCN, 'model_05_Adam', save_folder = SAVE_PATH)

#model6 = models.CNN_06(32, 64, 128, 256, optimizer=Adam, lr=0.001, momentum=None)
#main.train_and_evaluate(model6, 'model_06_Adam', save_folder = SAVE_PATH)









################ Hyperparameter search model 3 #################################
#hps_save_path = os.path.join(SAVE_PATH, 'HPS_model3');

#filters = [(32,64) ,   (16,32)  ,    (64,128)]
#conv_ks = [(3,3) , (5,5)]
#mp_ks   = [(2,2) , (4,4)]

#i=0
#for filt in filters:
#    for conv in conv_ks:
#        for mp in mp_ks:
#            model3_HPS = models.CNN_03(filt[0], filt[1], optimizer=SGD, lr=0.01, momentum=None)
#            main.train_and_evaluate(model3_HPS, f"model_03_HPS_{i}", save_folder = hps_save_path)
#            i += 1;


# HPS EXPERIMENTS:
#i=0:   f: 32, 64  c: (3, 3), m: (2, 2)
#i=1:   f: 32, 64  c: (3, 3), m: (4, 4)
#i=2:   f: 32, 64  c: (5, 5), m: (2, 2)
#i=3:   f: 32, 64  c: (5, 5), m: (4, 4)
#i=4:   f: 16, 32  c: (3, 3), m: (2, 2)
#i=5:   f: 16, 32  c: (3, 3), m: (4, 4)
#i=6:   f: 16, 32  c: (5, 5), m: (2, 2)
#i=7:   f: 16, 32  c: (5, 5), m: (4, 4)
#i=8:   f: 64, 128 c: (3, 3), m: (2, 2)
#i=9:   f: 64, 128 c: (3, 3), m: (4, 4)
#i=10:  f: 64, 128 c: (5, 5), m: (2, 2)
#i=11:  f: 64, 128 c: (5, 5), m: (4, 4)




###### ALTERNATIVES: DROPOUT = 0.2##############################################
model4_dp = models.CNN_04(32, 64, 128, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0.2)
main.train_and_evaluate(model4_dp, 'model_04_Adam_dropout_20', save_folder = SAVE_PATH)

model6_dp1 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0.2)
main.train_and_evaluate(model6_dp1, 'model_06_SGD_dropout_20', save_folder = SAVE_PATH)

model6_dp2 = models.CNN_06(32, 64, 128, 256, optimizer=SGD, lr=0.01, momentum=0.95, dropout_frac=0.2)
main.train_and_evaluate(model6_dp2, 'model_06_LR_Decay_dropout_20', save_folder = SAVE_PATH, adaptive_LR=True)

model6_dp3 = models.CNN_06(32, 64, 128, 256, optimizer=Adam, lr=0.001, momentum=None, dropout_frac=0.2)
main.train_and_evaluate(model6_dp3, 'model_06_Adam_dropout_20', save_folder = SAVE_PATH)
