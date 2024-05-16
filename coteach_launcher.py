import os
import sys
import math

import pickle
import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import Progbar

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets import AVA_generators
from models import keras_models
from argument_parser import Parser

from coteach_model import CoteachingModel

# This script launches the training process for coteaching models. Since Keras
# doesn't have a built-in coteaching model, the training loop has been created
# from the ground-up. 
if __name__ == '__main__':
    
    # Create parser and parse arguments
    p = Parser()
    (params, model_path, _) = p.parse_and_generate_path(sys.argv[1:])    
                   
    # check/create the folders
    checkpoints_dir = '../AQA-checkpoints'
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)
    models_dir = '../AQA-models'
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

    
    # 1. Dataset preparation
    dataset_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                       pre_transform=PRETRANSFORM, use_cache=USE_CACHE)

    # 2. Network preparation. Two models must be built
    model_f = keras_models(num_classes=dataset_generator.NUM_CLASSES, network=NETWORK, act=ACTIVATION)
    model_g = keras_models(num_classes=dataset_generator.NUM_CLASSES, network=NETWORK, act=ACTIVATION)
                   
    # Store models. For now, the same model is being used, so only one config can be saved (E.G., model_f)
    dict_model = model_f.model.get_config()
    f = open(os.path.join(models_dir,"{}_model.pkl".format(model_path)),"wb")
    pickle.dump(dict_model,f)
    f.close()

    json_string = model_f.model.to_json()
    f = open(os.path.join(models_dir,"{}_model.json".format(model_path)),"w")
    f.write(json_string)
    f.close()
    
    # Loss configuration
    if LOSS == 'mse':
        aux_loss = tf.keras.losses.MeanSquaredError
        aux_metric = [tf.keras.metrics.MeanAbsoluteError]
    elif LOSS == 'msle':
        aux_loss = tf.keras.losses.MeanSquaredLogarithmicError
        aux_metric = [tf.keras.metrics.MeanSquaredError,tf.keras.metrics.MeanAbsoluteError]
    elif LOSS == 'kl':
        aux_loss = tf.keras.losses.KLDivergence
        aux_metric = [tf.keras.metrics.MeanSquaredError,tf.keras.metrics.MeanAbsoluteError]
    elif LOSS == 'cross':
        aux_loss = tf.keras.losses.CategoricalCrossentropy
        aux_metric = [tf.keras.metrics.MeanSquaredError, tf.keras.metrics.CategoricalAccuracy]
    elif LOSS == 'bce':
        aux_loss = tf.keras.losses.BinaryCrossentropy
        aux_metric = [tf.keras.metrics.MeanSquaredError,tf.keras.metrics.MeanAbsoluteError]
    else:
        raise ValueError('Unsupported loss function!')
                           
    if OPTIMIZER == 'Adam':
        optimizer_f = Adam(lr=LR, decay=1e-8)    
        optimizer_g = Adam(lr=LR, decay=1e-8)    

    if OPTIMIZER == 'SGD':
        optimizer_f = SGD(lr=LR, decay=0)
        optimizer_g = SGD(lr=LR, decay=0)

    # Build coteaching model object
    coteaching = CoteachingModel((model_f.model, model_g.model), aux_loss, (optimizer_f, optimizer_g), aux_metric)

    # Create image generators for train and validation
    train_generator = dataset_generator.get_train(
        prep = model_f.prep_func,
        bsize = BATCHSIZE,
        tsize = model_f.model.input_shape[1:3]
    )

    val_generator = dataset_generator.get_val(
        prep = model_f.prep_func,
        bsize = BATCHSIZE,
        tsize = model_f.model.input_shape[1:3]
    )

    ################################################################################
    # TRAINING LOOP
    ################################################################################

    # Assume a noise rate of 0.2
    tau = 0.2

    # t_k=10 seems to work well on the original article
    t_k = 10

    select_rate = 1
    best_val_loss = 100000000

    # Training loop. Run for each epoch
    for epoch in range(EPOCHS):
        
        # Reset training and validation metrics.
        coteaching.reset_metrics()
        coteaching.reset_metrics_val()
        
        # Print epoch number and progress bar
        print("\nEpoch {}/{}".format(epoch+1, EPOCHS))
        pb_i = Progbar(len(train_generator))
        
        # Shuffle dataset and run training
        for i in range(len(train_generator)):
            inputs_batch, targets_batch = train_generator[i]
            logs = coteaching.train_step(inputs_batch, targets_batch, select_rate)
            pb_i.add(1)
        
        # Update selection rate
        select_rate = 1 - min((epoch/t_k)*tau, tau)

        # Print training results
        print(f"Training results at the end of epoch {epoch}")
        for key, value in logs.items():
            print(f"...{key}: {value:.4f}")
            
        # Run over validation dataset and print results
        for i in range(len(val_generator)):
            inputs_batch_val, targets_batch_val = val_generator[i]
            val_logs = coteaching.test_step(inputs_batch_val, targets_batch_val)

        print("Test evaluation results:") 
        for key, value in val_logs.items(): 
            print(f"...{key}: {value:.4f}")
            
        if (val_logs["loss_f_val"] < best_val_loss):
            print(f"Validation loss for model F has improved from {best_val_loss} to {val_logs['loss_f_val']}. Checkpointing model F...")
            best_val_loss = val_logs["loss_f_val"]       
            model_f.model.save(os.path.join(checkpoints_dir, f"{model_path}_coteach.h5"))
        
        if (val_logs["loss_g_val"] < best_val_loss):
            print(f"Validation loss for model G has improved from {best_val_loss} to {val_logs['loss_g_val']}. Checkpointing model G...")
            best_val_loss = val_logs["loss_g_val"]       
            model_g.model.save(os.path.join(checkpoints_dir, f"{model_path}_coteach.h5"))

        # Shuffle training set
        train_generator.on_epoch_end()
