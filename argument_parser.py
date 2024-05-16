from argparse import ArgumentParser

def build_path_from_list_of_args(arg_list):
    """
    This function builds a model path from a list of arguments.

    Arguments:
        - arg_list: a list of strings. Each string is a specific argument. The arguments contained within must be in this order:
                    obj,mod,pre,tsize,vsize,net,act,loss,opt,lr,bsize,epochs,dropout,clean_tech,retrain_mode

    Returns:
        - model_name: the name of the model. Afterwards, this name can be employed to load/save models, load predictions, etc.
    """
    return 'AQA_OBJ-{}_MOD-{}_PRE-{}_NET-{}_ACT-{}_LOSS-{}_OPT-{}_LR-{}_BS-{}_E-{}_DP-{}_CLEAN-{}_RETRAIN-{}'.format(arg_list[0],
                                                                                                                arg_list[1],
                                                                                                                arg_list[2],
                                                                                                                arg_list[5],
                                                                                                                arg_list[6],
                                                                                                                arg_list[7],
                                                                                                                arg_list[8],
                                                                                                                arg_list[9],
                                                                                                                arg_list[10],
                                                                                                                arg_list[11],
                                                                                                                arg_list[12],
                                                                                                                arg_list[13],
                                                                                                                arg_list[14])

# This class handles paramenter parsing and model path naming in a centralised way, 
# for both regular models and cleaned models
class Parser():

    # Class constructor. Initializes the argument parser object
    def __init__(self):

        self.p = ArgumentParser('AQA train')

        # OBJECTIVE VARIABLE PARAMETERS    
        # Objective variable
        self.p.add_argument('-obj', type = str, default = 'mean',  choices=['mean', 'ARV', 'WARV', 'distribution', 'LDA', 'Gauss', 'K-means'],
                    help = 'network used for training (default: inception)')
        # Modifications to the objetive variable.
        self.p.add_argument('-mod', type = str, default = 'none', choices=['none', 'binaryClasses', 'binaryWeights', 'cumulative', 'rank', 'pairwise'])

        # Modifications to the AVA dataset which are applied before using any method
        self.p.add_argument('-pre', type = str, default = 'none', choices = ['none', 'RF-IRF', 'RF-IRF_soft'],
                    help = 'transformation applied to AVA before computing any objective')

        # Cache parameters
        self.p.add_argument('-use_cache', action='store_true', help = "Load cached AVA transformation (if it exists), and write the scores to cache once computed")

        # TRAINING PARAMETERS
        self.p.add_argument('-tsize', type = float, default = 0.08,
                    help = 'test set size (default: 0.08)')
        self.p.add_argument('-vsize', type = float, default = 0.2,
                    help = 'validation set size (default: 0.2)')

        # NETWORK PARAMETERS
        # Finetuning network
        self.p.add_argument('-net', type = str, default = 'inception',  choices=['vgg16', 'inception', 'mobilenet', 'resnet','naive'],
                    help = 'network used for training (default: inception)')
        # Activation function
        self.p.add_argument('-act', type = str, default = 'linear',  choices=['linear', 'sigmoid', 'softmax'],
                    help = 'network used for training (default: inception)')
        # Loss
        self.p.add_argument('-loss', type = str, default = 'mse', choices=['mse','msle','emd','kl','cross','pairwise','Wmse','bhatta','bce','Wbce'],
                    help = 'network loss function (default: mse)')
        # Optimizer
        self.p.add_argument('-opt', type = str, default = 'Adam', choices=['Adam','SGD'],
                    help = 'network optimizer (default: Adam)')
        # Learning Rate
        self.p.add_argument('-lr', type = float, default = 3e-6,
                    help = 'learning rate for the optimizer (default: 3e-6)')

        # Dropout
        self.p.add_argument('-dropout', type = float, default = 0.75,
                    help = 'dropout rate for the CNN output (default: 0.75)')

        # Batch size
        self.p.add_argument('-bsize', type = int, default = 64,
                    help = 'batch size (default: 64)')
        # Epochs
        self.p.add_argument('-epochs', type = int, default = 20,
                    help = 'number of epochs (default: 20)')

        self.p.add_argument('-clean_tech', type=str, default='none', choices=['prune_by_class', 'prune_by_noise_rate', 'both', 'none'],
                    help='technique used by cleanlab to clean the dataset')

        self.p.add_argument('-retrain_mode', type=str, default='none', choices=['from_scratch', 'fine_tune', 'none'],
                    help='how to re-train the model after cleaning')


    # Parse a list of arguments, and return both the Namespace object
    # generated by calling parse_args, and a string representing the
    # base file path for the model specified in the arguments
    def parse_and_generate_path(self, args):

        parser = self.p.parse_args(args)

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
        DROPOUT = parser.dropout

        CLEAN_TECH = parser.clean_tech
        RETRAIN = parser.retrain_mode

        # Regular model path, including cleaning information
        self.model_path = 'AQA_OBJ-{}_MOD-{}_PRE-{}_NET-{}_ACT-{}_LOSS-{}_OPT-{}_LR-{}_BS-{}_E-{}_DP-{}_CLEAN-{}_RETRAIN-{}'.format(OBJECTIVE,
                                                                                                                            MODIFIER,
                                                                                                                            PRETRANSFORM,
                                                                                                                            NETWORK,
                                                                                                                            ACTIVATION,
                                                                                                                            LOSS,
                                                                                                                            OPTIMIZER,
                                                                                                                            LR,
                                                                                                                            BATCHSIZE,
                                                                                                                            EPOCHS,
                                                                                                                            DROPOUT,
                                                                                                                            CLEAN_TECH,
                                                                                                                            RETRAIN)

        # Model path without cleaning information. Used for loading base models when performing cleaning and fine-tuning
        self.base_model_path = 'AQA_OBJ-{}_MOD-{}_PRE-{}_NET-{}_ACT-{}_LOSS-{}_OPT-{}_LR-{}_BS-{}_E-{}_DP-{}_CLEAN-none_RETRAIN-none'.format(OBJECTIVE,
                                                                                                                            MODIFIER,
                                                                                                                            PRETRANSFORM,
                                                                                                                            NETWORK,
                                                                                                                            ACTIVATION,
                                                                                                                            LOSS,
                                                                                                                            OPTIMIZER,
                                                                                                                            LR,
                                                                                                                            BATCHSIZE,
                                                                                                                            EPOCHS,
                                                                                                                            DROPOUT)

        return (parser, self.model_path, self.base_model_path)

