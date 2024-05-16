import os
import warnings

from argument_parser import Parser
from datasets import AVA_generators
from losses import earth_mover_loss

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys

from sklearn.metrics import balanced_accuracy_score, accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from scipy.stats import rankdata
from scipy.stats import entropy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_confusion_matrix_and_fp_fn_means(ground, pred, means):
    """
    Compute confusion matrix of (ground) and (pred). Afterwards, compute average rating scores of false positives and false negatives.

    Assumes ground and pred to be binary ground-truths, and means to be an array of floats.

    Returns the amount of true positives, true negatives, false positives and false negatives, and the average scores of all.
    """
    unq = np.array([x + 2*y for x, y in zip(pred, ground)])

    tp = np.array(np.where(unq == 3)).tolist()[0]
    fp = np.array(np.where(unq == 1)).tolist()[0]
    tn = np.array(np.where(unq == 0)).tolist()[0]
    fn = np.array(np.where(unq == 2)).tolist()[0]

    fn_means = np.mean(means[fn])
    fp_means = np.mean(means[fp])
    tn_means = np.mean(means[tn])
    tp_means = np.mean(means[tp])

    # Obtain mean of 10% largest false negatives and 10% of smallest false positives
    fn_scores = means[fn]
    fn_means_largest_fns = np.mean(fn_scores[fn_scores > np.percentile(fn_scores, 90)])
    fp_scores = means[fp]
    fp_means_smallest_fps = np.mean(fp_scores[fp_scores < np.percentile(fp_scores, 10)])

    #min_fn = np.max(means[fn])
    #max_fp = np.min(means[fp])

    return len(tn), len(fp), len(fn), len(tp), tn_means, fp_means, fn_means, tp_means, fn_means_largest_fns, fp_means_smallest_fps

def bal_accuracy_thirds(ground, pred, model_name):
    """
    Compute balanced accuracy of (pred) w.r.t. (ground) on each tercile of the ground truth (ground)
    
    Assumes both predictions and ground-truth to be a single probability value, in the range (0,1)

    Prints the results in a file called model_name, located in the directory specified by the environment variable "AQA_results"
    """
    ground_terciles = np.percentile(ground, [i*100/3 for i in range(1,4)])
    
    index_bad = np.argwhere(ground < ground_terciles[0])
    index_normal = np.argwhere((ground >= ground_terciles[0]) & (ground < ground_terciles[1]))
    index_good = np.argwhere(ground >= ground_terciles[1])
    
    if (index_bad.size != 0):
        bal_acc_bad = balanced_accuracy_score(ground[index_bad] > 0.5, pred[index_bad] > 0.5)
    else:
        bal_acc_bad = 0
    bal_acc_normal = balanced_accuracy_score(ground[index_normal] > 0.5, pred[index_normal] > 0.5)
    bal_acc_good = balanced_accuracy_score(ground[index_good] > 0.5, pred[index_good]> 0.5)
    
    with open(f'{os.environ["AQA_results"]}/{model_name}_results.txt', 'a') as f:
        print(f'Balanced accuracy results by ground-truth quality terciles: ', file=f)
        print(f'Balanced accuracy, 1st tercile: {bal_acc_bad}', file=f)
        print(f'Balanced accuracy, 2nd tercile: {bal_acc_normal}', file=f)
        print(f'Balanced accuracy, 3rd tercile: {bal_acc_good}', file=f)
        print(f"\n{'#'*80}\n", file=f)

def get_distribution_metrics_and_plot(ground, pred, means, ground_name, plot_dir, plot_name, model_name):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be normalized vote distributions, from 1 to 10
    """
    pred_means = np.sum(pred * np.arange(0.1,1.1,0.1), axis=1) / np.sum(pred, axis=1)
    ground_means = np.sum(ground * np.arange(0.1,1.1,0.1), axis=1) / np.sum(ground, axis=1)
                
    bal_accuracy = balanced_accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mean_emd = np.mean([earth_mover_loss(tf.constant(pred[i], dtype=tf.float64), tf.constant(ground[i], dtype=tf.float64)) for i in range(len(pred))])
    accuracy = accuracy_score(ground_means > 0.5, pred_means > 0.5)
    mse = mean_squared_error(ground, pred)
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in np.vstack((1-pred_means,pred_means)).T])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in np.vstack((1-ground_means,ground_means)).T])
    tn, fp, fn, tp, tn_means, fp_means, fn_means, tp_means, min_fn, max_fp = get_confusion_matrix_and_fp_fn_means(ground_means > 0.5, pred_means > 0.5, means)
    
    with open(f'{os.environ["AQA_results"]}/{model_name}_results.txt', 'a') as f:
        print(f"RESULTS W.R.T. {ground_name}: ", file=f)
        print("Balanced accuracy: " + str(bal_accuracy), file=f)
        print("Accuracy: " + str(accuracy), file=f)
        print("Mean EMD distance: " + str(mean_emd), file=f)
        print("Mean squared error: " + str(mse), file=f)
        print("Average entropy (predictions): " + str(avg_entropy_pred), file=f)
        print(f"Average entropy ({ground_name}): " + str(avg_entropy_grnd), file=f)
        print(f"Confusion matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}", file=f)
        print(f"Avg. score of TN: {tn_means}, Avg. score of FP: {fp_means}, Avg. score of FN: {fn_means}, Avg. score of TP: {tp_means}", file=f)
        print(f"Avg. of 10% smallest FN: {min_fn}, Avg. of 10% largest FP: {max_fp}", file=f)
        print(f"{bal_accuracy:.4f} & {accuracy:.4f} & {mean_emd:.4f} & {mse:.4f} & {avg_entropy_grnd:.4f} & {fp_means:.3f} & {fn_means:.3f} \\\\ \\hline", file=f)
        print("--------------------------------------------------------", file=f)
    
    bal_accuracy_thirds(ground_means, pred_means, model_name)
    plot_prediction_and_gt(ground_means, pred_means, ground_name, plot_dir, plot_name)
    
def get_binary_metrics_and_plot(ground, pred, means, ground_name, plot_dir, plot_name, model_name):
    """
    Compute balanced accuracy, mean EMD distance, and accuracy of a set of predictions (pred)
    WRT to another ground-truth (ground), and print results, naming the other ground truth as (ground-name).
    
    After printing such metrics, plot predictions and ground-truth together.
    
    Assumes both predictions and ground-truths to be a 2-component probability distribution, in the range (0,1). 
    """
        
    bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
    mse = mean_squared_error(ground, pred)
    tn, fp, fn, tp, tn_means, fp_means, fn_means, tp_means, min_fn, max_fp = get_confusion_matrix_and_fp_fn_means(ground[:,1] > 0.5, pred[:,1] > 0.5, means)
    avg_entropy_pred = np.mean([entropy(p, base=2) for p in pred])
    avg_entropy_grnd = np.mean([entropy(g, base=2) for g in ground])

    # Component swapping fix
    if (accuracy < 0.5):
        pred = 1 - pred
        bal_accuracy = balanced_accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        accuracy = accuracy_score(ground[:,1] > 0.5, pred[:,1] > 0.5)
        mse = mean_squared_error(ground, pred)
        tn, fp, fn, tp, tn_means, fp_means, fn_means, tp_means, min_fn, max_fp = get_confusion_matrix_and_fp_fn_means(ground[:,1] > 0.5, pred[:,1] > 0.5, means)

        
    with open(f'{os.environ["AQA_results"]}/{model_name}_results.txt', 'a') as f:
        print(f"RESULTS W.R.T. {ground_name}: ", file=f)
        print("Balanced accuracy: " + str(bal_accuracy), file=f)
        print("Accuracy: " + str(accuracy), file=f)
        print("Mean squared error: " + str(mse), file=f)
        print("Average entropy (predictions): " + str(avg_entropy_pred), file=f)
        print(f"Average entropy ({ground_name}): " + str(avg_entropy_grnd), file=f)
        print(f"Confusion matrix: TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}", file=f)
        print(f"Avg. score of TN: {tn_means}, Avg. score of FP: {fp_means}, Avg. score of FN: {fn_means}, Avg. score of TP: {tp_means}", file=f)
        print(f"Avg. of 10% smallest FN: {min_fn}, Avg. of 10% largest FP: {max_fp}", file=f)
        print(f"{bal_accuracy:.4f} & {accuracy:.4f} & {mse:.4f} & {avg_entropy_grnd:.4f} & {fp_means:.3f} & {fn_means:.3f} \\\\ \\hline", file=f)
        print("--------------------------------------------------------", file=f)
        
    bal_accuracy_thirds(ground[:,1], pred[:,1], model_name)
    plot_prediction_and_gt(ground[:,1], pred[:,1], ground_name, plot_dir, plot_name)
    
def plot_prediction_and_gt(ground, pred, ground_name, plot_dir, plot_name):
    """
    Plot predictions and ground-truth against each other, and save the plot in (plot_dir).
    
    Assumes both (pred) and (ground) to be in the range (0,1), either coming from a probability or a normalized
    mean value.
    """
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    plt.hist(ground*10, bins = np.linspace(0,10,100), alpha = 0.5, color = 'r', label=ground_name)
    plt.hist(pred*10, bins = np.linspace(0,10,100), alpha = 0.5, color = 'b', label='Predictions')
    plt.legend()
    plt.savefig(f'{plot_dir}/{plot_name}.svg', format = 'svg', dpi = 1200)
    plt.clf()
    
    # Plot and save decile-hit heatmap
    ax = sns.heatmap(compute_nthile_hits(ground, pred, 10))
    plt.xlabel('predicted',fontsize=15)
    plt.ylabel('true',fontsize=15)
    plt.title(f'Against {plot_name.split("_")[1]}')
    plt.savefig(f'{plot_dir}/{plot_name}_heatmap_decile.svg', format = 'svg', dpi = 1200)
    plt.clf()

    # Plot and save quintile-hit heatmap
    ax = sns.heatmap(compute_nthile_hits(ground, pred, 5))
    plt.xlabel('predicted',fontsize=15)
    plt.ylabel('true',fontsize=15)
    plt.title(f'Against {plot_name.split("_")[1]}')
    plt.savefig(f'{plot_dir}/{plot_name}_heatmap_quintile.svg', format = 'svg', dpi = 1200)
    plt.clf()

    # Plot and save prediction boxplot
    sns.set_theme(style="whitegrid")
    sns.boxplot(y=pred * 9 + 1)
    plt.ylim(0,10)
    plt.savefig(f'{plot_dir}/predictions_boxplot.svg', format = 'svg', dpi = 1200)
    plt.clf()
    
def compute_nthile_hits(ground, pred, n):
    """
    Compute the nth-iles for each value in both ground-truth and predictions, and then compute the number of hits
    for each ground-truth/prediction nth-ile pair. This should give an estimate of where is the model giving good/bad
    predictions.
    """
    ground_decs = pd.cut((rankdata(ground)-1) / len(ground), n, labels=False)
    pred_decs = pd.cut((rankdata(pred)-1) / len(pred), n, labels=False)
    
    return np.matrix([[np.sum(pred_decs[ground_decs == i] == j) / np.sum(ground_decs == i) for j in range(n)] for i in range(n)])
    
    
######################################################################################################################################
def main():
    # Create parser and parse arguments
    p = Parser()
    (params, model_name, _) = p.parse_and_generate_path(sys.argv[1:])

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

    # Load both trained model, baseline NIMA model and real new ground-truth scores
    prediction_path = os.environ['AQA_predictions']
    
    generator = AVA_generators(obj_class=OBJECTIVE, mod_class=MODIFIER, test_split=TESTSIZE, val_split=VALSIZE, 
                                       pre_transform=PRETRANSFORM, use_cache=USE_CACHE)
        
    predictions = np.load(f"{prediction_path}/{model_name}_predictions.npy")
    new_gt = generator.test_scores
    #nima_gt = np.load(f"{prediction_path}/nima_baseline_mobilenet.npy")[:,1:11]
    real_votes = AVA_generators(obj_class='distribution', test_split=TESTSIZE, val_split=VALSIZE, use_cache=False).test_scores
    real_votes_binary = AVA_generators(obj_class='mean', mod_class='binaryWeights', test_split=TESTSIZE, val_split=VALSIZE, use_cache=False).test_scores
    real_votes_means = AVA_generators(obj_class='mean_unnormalized', test_split=TESTSIZE, val_split=VALSIZE, use_cache=False).test_scores
    
    plot_dir = f'{prediction_path}/{model_name}_graphs'

    # Each form of ground-truth (distribution, binary weights/classes) requires
    # a different way of obtaining metrics.
              
    warnings.filterwarnings("ignore", category=UserWarning)

    # Reset results file
    with open(f'{os.environ["AQA_results"]}/{model_name}_results.txt', 'w') as f:
        print(f"\n{'#'*80}\n")
    
    # Distribution-like ground-truths
    if (OBJECTIVE in ['distribution']):
        get_distribution_metrics_and_plot(new_gt, predictions, real_votes_means, "new ground-truth", plot_dir, "against_new-gt", model_name)
        get_distribution_metrics_and_plot(real_votes, predictions, real_votes_means, "real votes", plot_dir, "against_real", model_name)

    # Binary-like ground-truths
    if (OBJECTIVE in ['LDA', 'Gauss', 'K-means', 'mean']):
        get_binary_metrics_and_plot(new_gt, predictions, real_votes_means, "new ground-truth", plot_dir, "against_new-gt", model_name)
        get_binary_metrics_and_plot(real_votes_binary, predictions, real_votes_means, "real votes", plot_dir, "against_real", model_name)

if __name__ == '__main__':
    main()