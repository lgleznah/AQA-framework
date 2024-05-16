import os
from argparse import ArgumentParser

from datasets import AVA_generators

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import wasserstein_distance

# Parse common arguments
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

# Pipeline steps to run
p.add_argument('-flags', type = str, default = "111", help = 'Train-predict-metrics pipeline steps to run (formatted like xxx, where each x can be either 0 or 1. 0 -> don\'t run step. 1 -> run step)')

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

FLAGS = parser.flags

# Launch the training process in launcher.py
args = f"-obj {OBJECTIVE} -mod {MODIFIER} -pre {PRETRANSFORM} " \
        + f"-tsize {TESTSIZE} -vsize {VALSIZE} " \
        + f"-net {NETWORK} -act {ACTIVATION} -loss {LOSS} -opt {OPTIMIZER} -lr {LR} -bsize {BATCHSIZE} -epochs {EPOCHS}"

if USE_CACHE:
    args += f' -use_cache True'

if (FLAGS[0] == "1"):
    os.system(f"python launcher.py {args}")

# Run the trained network over the test set in get_fine_ranks.py
if (FLAGS[1] == "1"):
    os.system(f"python predictions.py {args}")

# Compute prediction metrics over the prediction file
if (FLAGS[2] == "1"):
    os.system(f"python metrics.py {args}")