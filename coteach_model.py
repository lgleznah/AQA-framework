import tensorflow as tf

class CoteachingModel():
    '''
    This class implements a coteaching model for TensorFlow 2.0, as explained in:
    https://proceedings.neurips.cc/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html
    '''

    def __init__(self, networks, loss, optimizers, metrics):
        '''
        CoteachingModel constructor.

        Parameters:
            networks: pair of DNNs which will be trained
            loss: loss function to apply during training and sample index selection
            optimizers: pair of optimizers, one for each model
            metric: list of metrics to be tracked for all the models.
        '''
        # Init networks
        self.model_f, self.model_g = networks

        # Init loss functions. For index selection, no reduction must be applied
        self.loss_fn_coteach = loss(reduction=tf.keras.losses.Reduction.NONE)
        self.loss_fn = loss()

        # Init optimizers
        self.optimizer_f, self.optimizer_g = optimizers

        # Init metrics
        self.metrics_f = [metric() for metric in metrics]
        self.metrics_g = [metric() for metric in metrics]
        self.val_metrics_f = [metric() for metric in metrics]
        self.val_metrics_g = [metric() for metric in metrics]

        # Init loss-tracking metrics
        self.loss_tracking_metric_f = tf.keras.metrics.Mean()
        self.loss_tracking_metric_g = tf.keras.metrics.Mean()
        self.val_loss_tracking_metric_f = tf.keras.metrics.Mean()
        self.val_loss_tracking_metric_g = tf.keras.metrics.Mean()

    # Compute R(T)% small-loss samples, and return their indices
    def select_coteaching(self, y_1, y_2, t, select_rate):
        '''
        Computes which sample indices should be sent to the models to perform coteaching.

        Parameters:
            y_1: batch predictions for model F
            y_2: batch predictions for model G
            y: ground-truth labels
            select_rate: rate of indices to keep

        Returns:
            A pair of lists. Each lists contains the sample indices which should be
            sent to the OTHER model to compute the gradients.
        '''
        loss_1 = self.loss_fn_coteach(t, y_1)
        ind_1_sorted = tf.argsort(loss_1)

        loss_2 = self.loss_fn_coteach(t, y_2)
        ind_2_sorted = tf.argsort(loss_2)

        num_remember = int(select_rate * len(ind_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]
        
        return ind_1_update, ind_2_update

    @tf.function
    def train_step(self, inputs, targets, select_rate):
        '''
        Training process for coteaching.

        Parameters:
            inputs: samples to be processed by the models
            targets: ground-truth labels for the samples
            select_rate: rate of indices to keep

        Returns:
            A dictionary with logs for the metrics and losses for both models
        '''
        # Run models and get predictions
        predictions_f = self.model_f(inputs, training=False)
        predictions_g = self.model_g(inputs, training=False)
        
        # Compute smallest-loss predictions
        best_f, best_g = self.select_coteaching(predictions_f, predictions_g, targets, select_rate)
        
        # Exchange inputs to perform co-teaching
        inputs_f = tf.gather(inputs, indices=best_g)
        inputs_g = tf.gather(inputs, indices=best_f)
        targets_f = tf.gather(targets, indices=best_g)
        targets_g = tf.gather(targets, indices=best_f)
        
        # Run these inputs over the models
        with tf.GradientTape() as f_tape, tf.GradientTape() as g_tape:                  
            predictions_f = self.model_f(inputs_f, training=True)
            predictions_g = self.model_g(inputs_g, training=True)
            loss_f = self.loss_fn(targets_f, predictions_f)
            loss_g = self.loss_fn(targets_g, predictions_g)

        # Compute gradients and apply optimizers  
        gradients_f = f_tape.gradient(loss_f, self.model_f.trainable_weights)             
        gradients_g = g_tape.gradient(loss_g, self.model_g.trainable_weights)
        self.optimizer_f.apply_gradients(zip(gradients_f, self.model_f.trainable_weights)) 
        self.optimizer_g.apply_gradients(zip(gradients_g, self.model_g.trainable_weights)) 
        
        # Compute metrics
        logs = {}                                       
        for metric_f in self.metrics_f:                           
            metric_f.update_state(targets_f, predictions_f)   
            logs[f"{metric_f.name}_f"] = metric_f.result()
            
        for metric_g in self.metrics_g:                           
            metric_g.update_state(targets_g, predictions_g)   
            logs[f"{metric_g.name}_g"] = metric_g.result()
            
        self.loss_tracking_metric_f.update_state(loss_f)       
        logs["loss_f"] = self.loss_tracking_metric_f.result()
        self.loss_tracking_metric_g.update_state(loss_g)       
        logs["loss_g"] = self.loss_tracking_metric_g.result() 
        return logs

    @tf.function
    def test_step(self, inputs, targets):
        '''
        Testing process for coteaching.

        Parameters:
            inputs: samples to be processed by the models
            targets: ground-truth labels for the samples

        Returns:
            A dictionary with logs for the metrics and losses for both models
            over the test dataset
        '''      
        # Run test inputs over the models and compute losses
        predictions_f = self.model_f(inputs, training=False)
        predictions_g = self.model_g(inputs, training=False)
        loss_f = self.loss_fn(targets, predictions_f)
        loss_g = self.loss_fn(targets, predictions_g)

        # Compute validation metrics
        logs = {}                                       
        for metric_f in self.val_metrics_f:                           
            metric_f.update_state(targets, predictions_f)   
            logs[f"{metric_f.name}_f_val"] = metric_f.result()
            
        for metric_g in self.val_metrics_g:                           
            metric_g.update_state(targets, predictions_g)   
            logs[f"{metric_g.name}_g_val"] = metric_g.result()
            
        self.val_loss_tracking_metric_f.update_state(loss_f)       
        logs["loss_f_val"] = self.val_loss_tracking_metric_f.result()
        self.val_loss_tracking_metric_g.update_state(loss_g)       
        logs["loss_g_val"] = self.val_loss_tracking_metric_g.result()
        return logs

    def reset_metrics(self):
        '''
        Reset training metrics for each epoch.
        '''
        for metric_f in self.metrics_f:
            metric_f.reset_states()
        for metric_g in self.metrics_g:
            metric_g.reset_states()
        self.loss_tracking_metric_f.reset_states()    
        self.loss_tracking_metric_g.reset_states()

    def reset_metrics_val(self):
        '''
        Reset validation metrics for each validation pass.
        '''
        for metric_f in self.val_metrics_f:
            metric_f.reset_states()
        for metric_g in self.val_metrics_g:
            metric_g.reset_states()
        self.val_loss_tracking_metric_f.reset_states()    
        self.val_loss_tracking_metric_g.reset_states()