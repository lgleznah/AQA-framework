import numpy as np
import os
import pandas as pd
import gzip
import pickle
from skimage.io import imread

from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import mixture
from sklearn.cluster import KMeans

from scipy.stats import beta, norm

import tensorflow as tf

class AVA_generators:
    
    def __init__(self, obj_class='mean', mod_class='none', pre_transform='none', test_split=0.08, val_split=0.2, random_seed=1000, use_cache = False, remove_train_idxs = []):

        self.objective = obj_class
        self.modifier = mod_class
        self.val_set = val_split > 0
        
        # path to the images and the text file which holds the scores and ids
        ava_images = os.environ['AVA_images_folder']
        cache_path = os.environ['AVA_cache']
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        ava_root = os.environ['AVA_info_folder']

        #file_csv = gzip.open(f'{ava_root}/info.csv','rb',2)
        self.data = pd.read_csv(f'{ava_root}/info.csv')
        self.data.loc[:,'id'] = self.data['id'].apply(str)
        self.data.sort_values(['id'],inplace=True)
        self.data.reset_index(inplace=True,drop=True)
        file_list = np.array([ava_images + '/{:}'.format(i) for i in np.array(self.data.loc[:,'id'])])
        
        # This is done first, because there are objectives that require the partition
        scores = np.array(self.data.iloc[:,2:12])
        self.train_image_paths, self.test_image_paths, self.train_scores, self.test_scores = train_test_split(file_list, 
                                                                                                  scores, 
                                                                                                  test_size = test_split,
                                                                                                  random_state = random_seed)
        
        if self.val_set:
            self.train_image_paths, self.val_image_paths, self.train_scores, self.val_scores = train_test_split(
                                                                                                self.train_image_paths, 
                                                                                                self.train_scores,
                                                                                                test_size = val_split,
                                                                                                random_state = random_seed)

        # Remove noisy indices from the training set, if an array of noisy indices was specified
        if len(remove_train_idxs) > 0:
            self.train_image_paths = self.train_image_paths[~remove_train_idxs]
            self.train_scores = self.train_scores[~remove_train_idxs]
        
        # Tras haber particionado, si se ha activado el flag use_cache y hay una cache almacenada, cargar el fichero
        # de scores desde la cache
        partial_filename = f"cache_{obj_class}_{mod_class}_{pre_transform}_{test_split}_{val_split}_{random_seed}_"
        cache_partial_path = cache_path + partial_filename
                
        if (use_cache and os.path.isfile(cache_partial_path + "train.pkl")):
            
            with open(cache_partial_path + "train.pkl", "rb") as f:
                self.train_scores = pickle.load(f)
                
            with open(cache_partial_path + "test.pkl", "rb") as f:
                self.test_scores = pickle.load(f)

            if (self.val_set):
                with open(cache_partial_path + "val.pkl", "rb") as f:
                    self.val_scores = pickle.load(f)
        
        
        # Todo el proceso de transformación, recalculado y modificación de la clase sólo debería hacerse si no se ha
        # cacheado la clase   
        else:
            # Antes de aplicar cualquier metodo, es posible que se quiera utilizar la variante RF-IRF o RF-IRF suavizada
            # del conjunto de votos. Aqui se aplican dichas transformaciones previas
            self.pretransform_scores(pre_transform)

            # esto es necesario ya que requerimos del calculo de los pesos en el train.
            # se podria generalizar facilmente para que se puedan aplicar mas funciones que requieran calculos en el train    
            if self.objective == 'WARV':
                self.aux_train = np.mean(self.get_pairwise_from_votes(self.train_scores), axis=0)
                aux_indexes = self.aux_train > 0.5
                self.aux_train[aux_indexes] = 1 - self.aux_train[aux_indexes]

            if self.objective == 'LDA':
                self.lda_model = LatentDirichletAllocation(
                                   n_components=2, 
                                   max_iter=10,
                                   random_state=0)

                self.lda_model.fit(self.train_scores)

            if self.objective == 'Gauss':
                self.clf = mixture.GaussianMixture(n_components=2,covariance_type='full')
                self.clf.fit(self.train_scores)

            if self.objective == 'K-means':
                self.clf = KMeans(n_clusters=2)
                self.clf.fit(self.train_scores)

            # Definimos la variable objetivo para las tres particiones a partir de los votos
            self.train_scores = self.get_objective_class(self.train_scores)
            self.test_scores = self.get_objective_class(self.test_scores)
            if self.val_set:
                self.val_scores = self.get_objective_class(self.val_scores)
                
            # Terminamos de modificar la clase si hay que alterar los valores objetivos
            self.train_scores = self.apply_modifier(self.train_scores)
            self.test_scores = self.apply_modifier(self.test_scores)
            if self.val_set:
                self.val_scores = self.apply_modifier(self.val_scores)

        print('Datasets ready !')
        print('Train set size : ', self.train_image_paths.shape, self.train_scores.shape)
        if self.val_set:
            print('Validation set size : ', self.val_image_paths.shape, self.val_scores.shape)
        print('Test set size : ', self.test_image_paths.shape, self.test_scores.shape)
        
        self.NUM_CLASSES = self.train_scores.shape[1]
        
        self.TRAIN_CASES = self.train_scores.shape[0]
        self.TEST_CASES = self.test_scores.shape[0]
        if self.val_set:
            self.VAL_CASES = self.val_scores.shape[0]
        
        # Antes de finalizar, guardar la clase en un fichero si se ha activado la flag de use_cache y si
        # las scores calculadas no están en cache
        if (use_cache and not os.path.isfile(cache_partial_path + "train.pkl")):
            
            # it turns into a pickle. funniest stuff ever
            with open(cache_partial_path + "train.pkl", "wb+") as f:
                pickle.dump(self.train_scores, f)
                
            with open(cache_partial_path + "test.pkl", "wb+") as f:
                pickle.dump(self.test_scores, f)

            if (self.val_set):
                with open(cache_partial_path + "val.pkl", "wb+") as f:
                    pickle.dump(self.val_scores, f)
            
    def get_objective_class (self, actual_scores):
        
        if self.objective == 'mean':        
            return np.sum(actual_scores * np.arange(0.1,1.1,0.1), axis=1) / np.sum(actual_scores, axis=1)

        if self.objective == 'mean_unnormalized':
            return np.sum(actual_scores * np.arange(1,11,1), axis=1) / np.sum(actual_scores, axis=1)
    
        if self.objective == 'distribution':
            return (actual_scores.T / np.sum(actual_scores, axis=1).T).T
        
        if self.objective == 'ARV':
            pairwise_scores = self.get_pairwise_from_votes(actual_scores)
            return 1 - (np.sum(pairwise_scores, axis=1) / pairwise_scores.shape[1])
                        
        if self.objective == 'WARV':
            pairwise_scores = self.get_pairwise_from_votes(actual_scores)
            pairwise_scores_bool = pairwise_scores.astype(bool)
            w_scores = np.array(pairwise_scores)
            for i in range(0,w_scores.shape[1]):       
                w_scores[pairwise_scores_bool[:,i],i] = self.aux_train[i]
            return 1 - np.sum(w_scores, axis=1) / np.sum(self.aux_train)
        
        if self.objective == 'LDA':
            return self.lda_model.transform(actual_scores)[:,1]
        
        if self.objective == 'Gauss':
            return self.clf.predict_proba(actual_scores)[:,1]
        
        if self.objective == 'K-means':
            return self.clf.predict(actual_scores)
        
        if self.objective == 'unmodified':
            return actual_scores
        
                    

    def apply_modifier(self, actual_scores):
        if self.modifier == 'binaryClasses':
            classes = np.array(actual_scores >= 0.5).astype(int)
            return np.vstack(([1-classes],[classes])).T
        
        if self.modifier == 'binaryWeights':
            return np.vstack(([1-actual_scores],[actual_scores])).T
        
        if self.modifier == 'cumulative':
            return np.cumsum(actual_scores)

        if self.modifier == 'rank':
            votes = np.array(self.data.iloc[:,2:12])
            # obtenemos el ranking
            rank_scores = np.argsort(np.argsort(actual_scores,axis=1), axis=1)
            # y lo normalizamos para que sume 1 y se pueda usar softmax
            n_values = actual_scores.shape[1]
            return rank_scores / (n_values * (n_values - 1) / 2)
        
        if self.modifier == 'pairwise':
            return self.get_pairwise_from_votes(actual_scores)
        
        # comprobamos si tiene 2 dimensiones
        if len(actual_scores.shape) < 2:
            return np.array([actual_scores]).T
        
        return actual_scores
        
    def pretransform_scores(self, pretransformer):
        # Transformacion RF-IRF
        if pretransformer == 'none':
            pass
        
        elif pretransformer == 'RF-IRF':
            idf_vector = np.sum(self.train_scores,axis=0)
            idf_vector = 1/idf_vector
            
            X = self.train_scores
            Xt = self.test_scores
            
            self.train_scores = [((X[i]*idf_vector)/np.sum(X[i]*idf_vector))*np.sum(X[i]) for i in range(X.shape[0])]
            self.train_scores = np.asarray(self.train_scores)
            self.test_scores = [((Xt[i]*idf_vector)/np.sum(Xt[i]*idf_vector))*np.sum(Xt[i]) for i in range(Xt.shape[0])]
            self.test_scores = np.asarray(self.test_scores)
            
            if self.val_set:
                Xv = self.val_scores
                self.val_scores = [((Xv[i]*idf_vector)/np.sum(Xv[i]*idf_vector))*np.sum(Xv[i]) for i in range(Xv.shape[0])]
                self.val_scores = np.asarray(self.val_scores)
                
        elif pretransformer == 'RF-IRF_soft':
            X = self.train_scores
            Xt = self.test_scores
            
            idf_vector = np.sum(X,axis=0)
            TFX = np.log(1+X)
            TFXt = np.log(1 + Xt)
            
            N = np.sum(X)
            idf_vector_2 = [(N-idf_vector[i])/idf_vector[i] for i in range(idf_vector.shape[0])]
            idf_vector_2 = np.asarray(idf_vector_2)
            idf_vector_2 = np.log(idf_vector_2)
            
            self.train_scores = [((TFX[i]*idf_vector_2)/np.sum(TFX[i]*idf_vector_2))*np.sum(X[i]) for i in range(TFX.shape[0])]
            self.train_scores = np.asarray(self.train_scores)
            
            self.test_scores = [((TFXt[i]*idf_vector_2)/np.sum(TFXt[i]*idf_vector_2))*np.sum(Xt[i]) for i in range(TFXt.shape[0])]
            self.test_scores = np.asarray(self.test_scores)
            
            if self.val_set:
                Xv = self.val_scores
                TFXv = np.log(1 + Xv)
                
                self.val_scores = [((TFXv[i]*idf_vector_2)/np.sum(TFXv[i]*idf_vector_2))*np.sum(Xv[i]) for i in range(TFXv.shape[0])]
                self.val_scores = np.asarray(self.val_scores)
        
        else:
            raise ValueError('Invalid pretransformer specified')
    
    def get_pairwise_from_votes(self, prev_scores):
        votes = np.array(prev_scores)
        counter = 0
        scores = np.zeros((votes.shape[0], np.int(votes.shape[1] * (votes.shape[1] - 1) / 2)))
        for i in range(votes.shape[1]):
            for j in range(i+1,votes.shape[1]):
                scores[:,counter] = np.array(np.logical_not(votes[:,i] < votes[:,j]), dtype=np.int)
                counter += 1
        return scores
    
    def get_train(self, prep = None, tsize = (224,224), bsize = 64, shuf=True):
        img_gen = ImageGenerator(self.train_image_paths, 
                              self.train_scores, 
                              prep_func = prep, 
                              target_size = tsize, 
                              batch_size = bsize, 
                              shuffle = shuf)
        
        return img_gen
              
    def get_val(self, prep = None, tsize = (224,224), bsize = 64):
        return ImageGenerator(self.val_image_paths, 
                              self.val_scores, 
                              prep_func = prep, 
                              target_size = tsize, 
                              batch_size = bsize)
    
    def get_test(self, prep = None, tsize = (224,224), bsize = 64):
        return ImageGenerator(self.test_image_paths, 
                              self.test_scores, 
                              prep_func = prep, 
                              target_size = tsize, 
                              batch_size = bsize)

    
class ImageGenerator(tf.keras.utils.Sequence):
    """ Generates inputs for the Keras model during training.
    Attributes: - batch_size (int): batch size during training
                - df (pandas.DataFrame): dataframe with columns: idImage, labels
                - img_size (int): shape of the inputs of the model
                - indexes (np.array): indexes used for splitting the dataset in batches
    Functions:
                - __init__
                - __len__
                - __get_item__
                - pad_small_image
                - data_generation
    """
    def __init__(self, files, labels, prep_func = None, target_size = (224,224), batch_size = 64, shuffle = False):

        self.files = files
        self.n_files = len(self.files)
        
        self.labels = labels
        self.prep_func = prep_func
        self.target_size = target_size
        self.nb_channels = 3
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.bboxes = []
        self.nb_crops = 1
        
        self.indexes = np.arange(self.n_files)
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.ceil(self.n_files / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data'
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,self.n_files)]

        # Find list of IDs
        files_batch = self.files[indexes]
        labels_batch = self.labels[indexes]
        
        if self.nb_crops > 1:
            bboxes_batch = [self.bboxes[k] for k in indexes]
            # Generate data
            X = self.crops_generation(files_batch, bboxes_batch)
        else:
            X = self.img_generation(files_batch)

        return X, labels_batch
    
    def set_bbox(self, bboxes):
        self.bboxes = bboxes
        self.nb_crops = self.bboxes[0].shape[0]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_files)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def crops_generation(self, files_b, bboxes_b):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.zeros((len(files_b), self.nb_crops, self.target_size[0], self.target_size[1], self.nb_channels),
                     dtype = np.float32)
        for i in range(len(files_b)):
            fn = files_b[i]
            img = imread(fn)
            bboxes = bboxes_b[i]
            for j in range(bboxes.shape[0]):
                bbox = bboxes[j]
                crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :].copy()
                if self.prep_func:
                    crop = self.prep_func(crop)
                X[i, j] = crop
        return X
    
            
    def img_generation(self, files_b):
        """
        Generates data containing batch_size samples
        """
        # Initialization
        X = np.zeros((len(files_b), self.target_size[0], self.target_size[1], self.nb_channels), dtype = np.float32)
        for i in range(len(files_b)):
            fn = files_b[i]
            img = tf.keras.preprocessing.image.load_img(fn, target_size=self.target_size)
            array = tf.keras.preprocessing.image.img_to_array(img)
            if self.prep_func:
                array = self.prep_func(array)
            X[i] = array
        return X