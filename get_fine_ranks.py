import sys
sys.path.append('../AQA-framework')
from datasets import AVA_generators

import os
from argparse import ArgumentParser
# esta linea es para usar CPU como backend
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# esto es para ejecutar en la GTX
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications
import tensorflow.keras.backend as K

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, mobilenet, inception_v3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import CustomObjectScope, GeneratorEnqueuer
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#applications.set_keras_submodules(backend=keras.backend, 
#                                        layers=keras.layers, 
#                                        models=keras.models, 
#                                        utils=keras.utils)

# Argument parser
p = ArgumentParser('AQA train')

# OBJECTIVE VARIABLE PARAMETERS    
# Objective variable
p.add_argument('-obj', type = str, default = 'mean',  choices=['mean', 'ARV', 'WARV', 'distribution', 'LDA', 'Gauss', 'K-means'],
               help = 'network used for training (default: inception)')
# Modifications to the objetive variable.
p.add_argument('-mod', type = str, default = 'none', choices=['none', 'binaryClasses', 'binaryWeights', 'cumulative', 'rank', 'pairwise'])

# Modifications to the AVA dataset which are applied before using any method
p.add_argument('-pre', type = str, default = 'none', choices = ['none', 'RF-IRF', 'RF-IRF_soft'],
               help = 'transformation applied to AVA before computing any objective')

# Cache parameters
p.add_argument('-use_cache', type = bool, default = False, help = "Load cached AVA transformation (if it exists), and write the scores to cache once computed")

# TRAINING PARAMETERS
p.add_argument('-tsize', type = float, default = 0.08,
               help = 'test set size (default: 0.08)')
p.add_argument('-vsize', type = float, default = 0.2,
               help = 'validation set size (default: 0.2)')

# NETWORK PARAMETERS
# Finetuning network
p.add_argument('-net', type = str, default = 'inception',  choices=['vgg16', 'inception', 'mobilenet', 'resnet','naive'],
               help = 'network used for training (default: inception)')
# Activation function
p.add_argument('-act', type = str, default = 'linear',  choices=['linear', 'sigmoid', 'softmax'],
               help = 'network used for training (default: inception)')
# Loss
p.add_argument('-loss', type = str, default = 'mse', choices=['mse','msle','emd','kl','cross','pairwise','Wmse','bhatta','bce','Wbce'],
               help = 'network loss function (default: mse)')
# Optimizer
p.add_argument('-opt', type = str, default = 'Adam', choices=['Adam','SGD'],
               help = 'network optimizer (default: Adam)')
# Learning Rate
p.add_argument('-lr', type = float, default = 3e-6,
               help = 'learning rate for the optimizer (default: 3e-6)')
# Batch size
p.add_argument('-bsize', type = int, default = 64,
               help = 'batch size (default: 64)')
# Epochs
p.add_argument('-epochs', type = int, default = 20,
               help = 'number of epochs (default: 20)')

parser = p.parse_args()
                   
OBJECTIVE = parser.obj
MODIFIER = parser.mod
PRETRANSFORM = parser.pre

USE_CACHE = parser.use_cache

VALSIZE = parser.vsize
TESTSIZE = parser.tsize

NETWORK = parser.net
ACTIVATION = parser.act
LOSS = parser.loss
OPTIMIZER = parser.opt
LR = parser.lr
BATCHSIZE = parser.bsize
EPOCHS = parser.epochs

BATCH_SIZE = 1

AVA_path = '/home/frubio/AVA/keras_partition/'
main_path = '/home/lgleznah/Doctorado'
model_path = 'AQA_OBJ-{}_MOD-{}_PRE-{}_NET-{}_ACT-{}_LOSS-{}_OPT-{}_LR-{}_BS-{}_E-{}'.format(OBJECTIVE,
                                                                                             MODIFIER,
                                                                                             PRETRANSFORM,
                                                                                             NETWORK,
                                                                                             ACTIVATION,
                                                                                             LOSS,
                                                                                             OPTIMIZER,
                                                                                             LR,
                                                                                             BATCHSIZE,
                                                                                             EPOCHS)

with open(os.path.join(main_path,"AQA-models/{}_model.pkl".format(model_path)), 'rb') as f:
    model_config = pickle.load(f)
    
def relu6(x):
    return K.relu(x, max_value=6)

with CustomObjectScope({'relu6': relu6}):
    model = Model.from_config(model_config)
    
model.summary()
    
# NUMPY
#model_weights = np.load(os.path.join(main_path,"FinetuningModels/{}_weights1.npy".format(model_path)))
#model.set_weights(model_weights)

# CHECKPOINTS
model.load_weights(os.path.join(main_path,"AQA-checkpoints/{}_bestmodel.h5".format(model_path)))

from datasets import AVA_generators

############## CAMBIOS #######################
datagen = ImageDataGenerator(preprocessing_function = applications.mobilenet.preprocess_input)

ava_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                       pre_transform=PRETRANSFORM, use_cache=USE_CACHE)

frame = pd.DataFrame(ava_generator.test_scores)
y_columns = np.array(frame.columns)
frame['files'] = ava_generator.test_image_paths
test_iter = datagen.flow_from_dataframe(frame, 
                                         x_col='files', 
                                         y_col = list(y_columns),
                                         target_size=model.input_shape[1:3], 
                                         class_mode='multi_output',
                                         batch_size=1)

'''
ava_generator = AVA_generators(class_type="rank")
train_frame = pd.DataFrame(9 - (ava_generator.train_scores * 45))
train_frame = train_frame.astype(np.int64).astype(str)
train_y_columns = np.array(train_frame.columns)
for i in range(0,10):
    enc = OneHotEncoder(sparse = False, categories=[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']])
    aux_frame = enc.fit_transform(np.array(train_frame[i]).reshape(len(train_frame[i]), 1))
    train_frame[i] = aux_frame.tolist()
train_frame['files'] = ava_generator.train_image_paths
train_iter = datagen.flow_from_dataframe(train_frame, 
                                         x_col='files', 
                                         y_col = list(train_y_columns),
                                         target_size=model.input_shape[1:3], 
                                         class_mode='multi_output',
                                         batch_size=128)
'''
##############################################

predictions = model.predict_generator(test_iter,
                                      steps=(ava_generator.TEST_CASES))
np.save(os.path.join(main_path,"AQA-predictions/{}_predictions.npy".format(model_path)),predictions)

