# coding: utf-8

# Code for applying cleanlab to the AVA dataset. The specific data being cleaned will depend on the network and ground-truth being used

import os
import numpy as np
import cleanlab as clb
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from cleanlab.pruning import get_noise_indices
from datasets import AVA_generators
from models import keras_models
from losses import earth_mover_loss, mean_pairwise_squared_loss, bhattacharyya_loss
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Get preprocessing function for a given model
def get_preprocess(model_name):
        if model_name == 'vgg16':
            return tf.keras.applications.vgg16.preprocess_input
        
        if model_name == 'inception':
            return tf.keras.applications.inception_v3.preprocess_input
        
        if model_name == 'mobilenet':
            return tf.keras.applications.mobilenet.preprocess_input
        
        if model_name == 'resnet':
            return tf.keras.applications.resnet.preprocess_input
    
# Convert an array of binary class probabilities to discrete classes
def weights_to_class(binary_weights):
    return binary_weights > 0.5


# Load neural network model and its corresponding generator

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

p.add_argument('-clean_tech', type=str, default='both', choices=['prune_by_class', 'prune_by_noise_rate', 'both'],
               help='technique used by cleanlab to clean the dataset')

p.add_argument('-retrain_mode', type=str, default='from_scratch', choices=['from_scratch', 'fine_tune'],
               help='how to re-train the model after cleaning')

parser = p.parse_args()

# Ground-truth params
OBJECTIVE = parser.obj
MODIFIER = parser.mod
PRETRANSFORM = parser.pre

USE_CACHE = parser.use_cache

# Data split params
VALSIZE = parser.vsize
TESTSIZE = parser.tsize

# Network params
NETWORK = parser.net
ACTIVATION = parser.act
LOSS = parser.loss
OPTIMIZER = parser.opt
LR = parser.lr
BATCHSIZE = parser.bsize
EPOCHS = parser.epochs

# Data-cleaning params
CLEAN_TECH = parser.clean_tech
RETRAIN = parser.retrain_mode

checkpoints_dir = '../AQA-checkpoints'
models_dir = '../AQA-models'
predictions_dir = '../AQA-predictions'

# Load the model and its weights
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

model = tf.keras.models.load_model(os.path.join(checkpoints_dir,"{}_bestmodel.h5".format(model_path)))

# Load data generator
dataset_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                   pre_transform=PRETRANSFORM, use_cache=USE_CACHE)

# Predict train set
predictions = model.predict_generator(dataset_generator.get_train(prep = tf.keras.applications.mobilenet.preprocess_input,
                                                                 bsize = 16,
                                                                 tsize = model.input_shape[1:3],
                                                                 shuf=False))

# Detect and remove noisy labels with the specified technique
labels = weights_to_class(dataset_generator.train_scores[:,1])

ordered_label_errors = get_noise_indices(
    s=labels,
    psx=predictions,
    prune_method=CLEAN_TECH
 )

# Retrain the network from scratch or fine-tune it, according to the specific parameter
clean_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, pre_transform=PRETRANSFORM, remove_train_idxs = ordered_label_errors)
model_path_clean = f"{model_path}_CLEAN-{CLEAN_TECH}_RETRAIN-{RETRAIN}"

# Fine-tune the current model using 1/10th of the current learning rate
if (RETRAIN == 'fine_tune'):
    
    K.set_value(model.optimizer.learning_rate, LR / 10)
    prep_func = get_preprocess(NETWORK)
    
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,"{}_bestmodel.h5".format(model_path_clean)),
                                      monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csvlogger = CSVLogger(os.path.join(models_dir,"{}_history.csv".format(model_path_clean)),
                          separator=',', append=False)
    
    history = model.fit_generator(dataset_generator.get_train(prep = prep_func,
                                                                    bsize = BATCHSIZE,
                                                                    tsize = model.input_shape[1:3]),
                                        steps_per_epoch=(dataset_generator.TRAIN_CASES // BATCHSIZE + 1),
                                        epochs=EPOCHS, 
                                        callbacks=[best_checkpoint,csvlogger],
                                        validation_data = dataset_generator.get_val(prep = prep_func,
                                                                                    bsize = BATCHSIZE,
                                                                                    tsize = model.input_shape[1:3]),
                                        validation_steps=(dataset_generator.VAL_CASES // BATCHSIZE + 1))

    np.save(os.path.join(models_dir,"{}_weights.npy".format(model_path_clean)), model.get_weights())
    f = open(os.path.join(models_dir,"{}_history.pkl".format(model_path_clean)), "wb")
    pickle.dump(history.history,f)
    f.close()

# Re-generate the current model with ImageNet weights
if (RETRAIN == 'from_scratch'):
    model = keras_models(num_classes=clean_generator.NUM_CLASSES, network=NETWORK, act=ACTIVATION)
                   
    # Store models
    dict_model = model.model.get_config()
    f = open(os.path.join(models_dir,"{}_model.pkl".format(model_path_clean)),"wb")
    pickle.dump(dict_model,f)
    f.close()

    json_string = model.model.to_json()
    f = open(os.path.join(models_dir,"{}_model.json".format(model_path_clean)),"w")
    f.write(json_string)
    f.close()
    
    # Configure models
    if LOSS == 'mse':
        aux_loss = 'mean_squared_error'
        aux_metric = ['mae']
    elif LOSS == 'msle':
        aux_loss = 'mean_squared_logarithmic_error'
        aux_metric = ['mse','mae']
    elif LOSS == 'emd':
        aux_loss = earth_mover_loss
        aux_metric = ['mse','mae']
    elif LOSS == 'kl':
        aux_loss = 'kullback_leibler_divergence'
        aux_metric = ['mse','mae']
    elif LOSS == 'cross':
        aux_loss = 'categorical_crossentropy'
        aux_metric = ['mse', 'acc']
    elif LOSS == 'pairwise':
        aux_loss = mean_pairwise_squared_loss
        aux_metric = ['mse']
    elif LOSS == 'bhatta':
        aux_loss = bhattacharyya_loss
        aux_metric = ['mse','mae']
    elif LOSS == 'bce':
        aux_loss = 'binary_crossentropy'
        aux_metric = ['mse','mae']
                           
    if OPTIMIZER == 'Adam':
        optimizer = Adam(lr=LR, decay=1e-8)        

    if OPTIMIZER == 'SGD':
        optimizer = SGD(lr=LR, decay=0)
            
    model.model.compile(loss=aux_loss, optimizer=optimizer, metrics=aux_metric)

    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,"{}_bestmodel.h5".format(model_path_clean)),
                                      monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    csvlogger = CSVLogger(os.path.join(models_dir,"{}_history.csv".format(model_path_clean)),
                          separator=',', append=False)
    
    history = model.model.fit_generator(clean_generator.get_train(prep = model.prep_func,
                                                                    bsize = BATCHSIZE,
                                                                    tsize = model.model.input_shape[1:3]),
                                        steps_per_epoch=(clean_generator.TRAIN_CASES // BATCHSIZE + 1),
                                        epochs=EPOCHS, 
                                        callbacks=[best_checkpoint,csvlogger],
                                        validation_data = clean_generator.get_val(prep = model.prep_func,
                                                                                    bsize = BATCHSIZE,
                                                                                    tsize = model.model.input_shape[1:3]),
                                        validation_steps=(clean_generator.VAL_CASES // BATCHSIZE + 1))

    np.save(os.path.join(models_dir,"{}_weights.npy".format(model_path_clean)),model.model.get_weights())
    f = open(os.path.join(models_dir,"{}_history.pkl".format(model_path_clean)),"wb")
    pickle.dump(history.history,f)
    f.close()