import tensorflow as tf
import tensorflow.keras.backend as K

# EMD
def earth_mover_loss(y_true, y_pred):
    cdf_ytrue = K.cumsum(y_true, axis=-1)
    cdf_ypred = K.cumsum(y_pred, axis=-1)
    samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
    return K.mean(samplewise_emd)

def bhattacharyya_loss(y_true, y_pred):
    return K.mean(-K.log(K.sum(K.sqrt(y_true * y_pred + K.epsilon()), axis=-1)))

def mean_pairwise_squared_loss(y_true, y_pred):
    terms = K.int_shape(y_pred)[-1]
    
    diffs = y_pred - y_true
    sum_squares_diff_per_batch = K.sum(K.square(diffs),axis=-1)
    term1 = 2.0 * sum_squares_diff_per_batch / (terms - 1)
    sum_diff = K.sum(diffs, axis=-1)
    term2 = 2.0 * K.square(sum_diff) / (terms * (terms - 1))
    
    return K.sum(term1 - term2)

