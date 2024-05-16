import os
import sys
from argument_parser import Parser
from datasets import AVA_generators
from models import get_preprocess

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Argument parser
p = Parser()
(params, model_name, base_model_name) = p.parse_and_generate_path(sys.argv[1:])
                   
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

RETRAIN = params.retrain_mode

checkpoints_dir = os.environ['AQA_checkpoints']
models_dir = os.environ['AQA_models']
predictions_dir = os.environ['AQA_predictions']

# Load base model if cleaning and fine-tuning were performed. Otherwise, load regular model
if (RETRAIN != 'none'):
    with open(os.path.join(models_dir,"{}_model.pkl".format(base_model_name)), 'rb') as f:
        model_config = pickle.load(f)

else:
    with open(os.path.join(models_dir,"{}_model.pkl".format(model_name)), 'rb') as f:
        model_config = pickle.load(f)
    
model = tf.keras.models.Model.from_config(model_config)

# CHECKPOINTS
model.load_weights(os.path.join(checkpoints_dir,"{}_bestmodel.h5".format(model_name)))


############## CAMBIOS #######################
dataset_generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                   pre_transform=PRETRANSFORM, use_cache=USE_CACHE)

predictions = model.predict_generator(dataset_generator.get_test(prep = get_preprocess(NETWORK),
                                                                 bsize = 16,
                                                                 tsize = model.input_shape[1:3]))

np.save(os.path.join(predictions_dir,"{}_predictions.npy".format(model_name)), predictions)

