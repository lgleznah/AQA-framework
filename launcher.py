# coding: utf-8

# # Finetuning para resolver AVA mediante clasificación binaria
# En este notebook se desarrolla el proceso para hacer finetuning en nuestro problema con las redes más utilizadas actualmente en problemas de tratamiento de imágenes.

import os
import sys
from argument_parser import Parser

import pickle
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import AVA_generators
from models import keras_models
from losses import earth_mover_loss, mean_pairwise_squared_loss, bhattacharyya_loss
    
# This launcher is an example of the use of the AQA Framework. It is divided in three steps:
# 1. Dataset preparation
# 2. Network preparation
# 3. Training

if __name__ == '__main__':
    
    # Create parser and parse arguments
    p = Parser()
    (params, model_path, _) = p.parse_and_generate_path(sys.argv[1:])    
                   
    # check/create the folders
    checkpoints_dir = os.environ['AQA_checkpoints']
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    models_dir = os.environ['AQA_models']
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
                   
    OBJECTIVE = params.obj
    MODIFIER = params.mod
    PRETRANSFORM = params.pre
    
    USE_CACHE = params.use_cache
    
    VALSIZE = params.vsize
    TESTSIZE = params.tsize
    
    NETWORK = params.net
    ACTIVATION = params.act
    LOSS = params.loss
    OPTIMIZER = params.opt
    LR = params.lr
    BATCHSIZE = params.bsize
    EPOCHS = params.epochs
    DROPOUT = params.dropout
    
    #MULTIPATCH = parser.mp
                   
                   
    max_epochs = 10
    lr_factor = 0.33
    
    # 1. Dataset preparation
    dataset_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                       pre_transform=PRETRANSFORM, use_cache=USE_CACHE)
    # 2. Network preparation
    model = keras_models(num_classes=dataset_generator.NUM_CLASSES, network=NETWORK, act=ACTIVATION, drop_value=DROPOUT)
    print(DROPOUT)
                   
    # Almacenamos los modelos
    dict_model = model.model.get_config()
    f = open(os.path.join(models_dir,"{}_model.pkl".format(model_path)),"wb")
    pickle.dump(dict_model,f)
    f.close()

    json_string = model.model.to_json()
    f = open(os.path.join(models_dir,"{}_model.json".format(model_path)),"w")
    f.write(json_string)
    f.close()
    
    # configuramos el modelo
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

    best_checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoints_dir,"{}_bestmodel.h5".format(model_path)),
                                      monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    csvlogger = CSVLogger(os.path.join(models_dir,"{}_history.csv".format(model_path)),
                          separator=',', append=False)
    
    history = model.model.fit_generator(dataset_generator.get_train(prep = model.prep_func,
                                                                    bsize = BATCHSIZE,
                                                                    tsize = model.model.input_shape[1:3]),
                                        steps_per_epoch=(dataset_generator.TRAIN_CASES // BATCHSIZE + 1),
                                        epochs=EPOCHS, 
                                        callbacks=[best_checkpoint,csvlogger],
                                        validation_data = dataset_generator.get_val(prep = model.prep_func,
                                                                                    bsize = BATCHSIZE,
                                                                                    tsize = model.model.input_shape[1:3]),
                                        validation_steps=(dataset_generator.VAL_CASES // BATCHSIZE + 1))

    np.save(os.path.join(models_dir,"{}_weights.npy".format(model_path)),model.model.get_weights())
    f = open(os.path.join(models_dir,"{}_history.pkl".format(model_path)),"wb")
    pickle.dump(history.history,f)
    f.close()

    sys.exit(0)