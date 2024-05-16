import numpy as np
import pandas as pd
import itertools
import skimage

import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.linalg import fractional_matrix_power, inv

import scipy

from MR import MR_saliency

class SelectAdaptativePatch(object):
    def __init__(self, filename, random_seed, patch_size=[112, 112]):
        self.image = skimage.io.imread(filename)
        self.random_seed = random_seed
        self.patch_size = patch_size
        
        self.i = 0
        
        mr = MR_saliency()
        S = mr.saliency(filename)
        self.S = S/255
        
        #check if image needs padding (if smaller than self.patch_size)
        pad_x = 0
        pad_y = 0
        if self.image.shape[0] < self.patch_size[0]:
            pad_y = (self.patch_size[0] - self.image.shape[0]) // 2 + 1 
        if self.image.shape[1] < self.patch_size[1]:
            pad_x = (self.patch_size[1] - self.image.shape[1]) // 2 + 1 
            
        self.image = np.pad(self.image, ((pad_y,pad_y), (pad_x,pad_x), (0,0)), mode='reflect')    
        self.S = np.pad(self.S, ((pad_y,pad_y), (pad_x,pad_x)), mode='reflect')

        # then, sobel
        self.E = skimage.filters.sobel(skimage.color.rgb2gray(self.image))

        # finally, the hue
        self.H = skimage.color.rgb2hsv(self.image)[:, :, 0]
        
        
        
    def initialize_five_centers_line(self):
        
        height, width = self.image.shape[0:2]
        
        h_step = height // 3
        w_step = width // 3
        
        h_safety = 0
        if h_step <= self.patch_size[0]:
            h_safety = self.patch_size[0] - h_step + 1
        w_safety = 0
        if w_step <= self.patch_size[1]:
            w_safety = self.patch_size[1] - w_step + 1
        
        x = [h_step + h_safety, w_step + w_safety,
             h_step*2 - h_safety, w_step + w_safety,
             h_step + h_safety, w_step*2 - w_safety,
             h_step*2 - h_safety, w_step*2 - w_safety,
             height // 2, width // 2]
        
        return np.array(x)
    
    def wassertsteinDistance(self, sigma_i, sigma_j):
        # Following F. Pitie and A. Kokaram
        for i in range(sigma_i.shape[0]):
            if sigma_i[i][i] == 0: 
                sigma_i[i][i] += 0.1
            if sigma_j[i][i] == 0: 
                sigma_j[i][i] += 0.1

        sigma_i_sqrt = fractional_matrix_power(sigma_i, .5)
        sigma_i_sqrt_inv = np.linalg.inv(sigma_i_sqrt)
        sigma_intermediate = np.dot(sigma_i_sqrt, sigma_j)
        sigma_temp = fractional_matrix_power(np.nan_to_num(np.dot(sigma_intermediate, sigma_i_sqrt)), .5)

        return np.dot(np.dot(sigma_i_sqrt_inv, sigma_temp), sigma_i_sqrt_inv)

    def D_p(self, E_i, E_j, H_i, H_j):
        # Pattern Diversity
        ## Edge 
        sigma_e_i = np.var(E_i)
        sigma_e_j = np.var(E_j)

        ## Chrominance (Hue)
        sigma_h_i = np.var(H_i)
        sigma_h_j = np.var(H_j)

        sigma_i = np.diag([sigma_e_i, sigma_h_i])
        sigma_j = np.diag([sigma_e_j, sigma_h_j])

        emd = self.wassertsteinDistance(sigma_i, sigma_j)

        return np.trace(emd).mean()
        
    def computeF(self, centers):
        
        #<0 : to minimize
        centers = centers.reshape(-1, 2).astype(int)

        #Init F
        F = 0

        ## To sum saliency, important : highly dependent on the combination method, here fitted for itertools.combinations
        first_loop = True
        number_loop = 0
        
        ## First, we check if all patches are inside the image.
        for center in centers:
            top = center[0] - self.patch_size[0]
            bottom = center[0] + self.patch_size[0] + 1
            left = center[1] - self.patch_size[1]
            right = center[1] + self.patch_size[1] + 1
            
            # check if bounding box is out of the image
            if (top <= 0) or (left <= 0) or (bottom >= self.image.shape[0]) or (right >= self.image.shape[1]):
                return 1
            
        # Saliency
        for center in centers:
            top = center[0] - self.patch_size[0]
            bottom = center[0] + self.patch_size[0] + 1
            left = center[1] - self.patch_size[1]
            right = center[1] + self.patch_size[1] + 1
            
            F += self.S[top:bottom, left:right].mean()
        
        # Compute patches
        for center in itertools.combinations(centers, 2):
            top_i = center[0][0] - self.patch_size[0]
            bottom_i = center[0][0] + self.patch_size[0] + 1
            left_i = center[0][1] - self.patch_size[0]
            right_i = center[0][1] + self.patch_size[0] + 1

            top_j = center[1][0] - self.patch_size[0]
            bottom_j = center[1][0] + self.patch_size[0] + 1
            left_j = center[1][1] - self.patch_size[0]
            right_j = center[1][1] + self.patch_size[0] + 1

            #print(number_loop)
            E_i = self.E[top_i:bottom_i, left_i:right_i]
            E_j = self.E[top_j:bottom_j, left_j:right_j]

            H_i = self.H[top_i:bottom_i, left_i:right_i]
            H_j = self.H[top_j:bottom_j, left_j:right_j]

            # Pattern diversity (emd)
            F = F + self.D_p(E_i, E_j, H_i, H_j)*0.3
        
        # Compute patches
        for center in itertools.combinations(centers, 2):
            # Euclidean distance between centers
            F = F + np.linalg.norm((center[0] - center[1]) / (self.image.shape[0],self.image.shape[1]))
        return - F
        
        
    def predict(self):
        #Setup
        x0 = self.initialize_five_centers_line()
        
        res = minimize(lambda x : self.computeF(x), \
                            x0, \
                            method='Nelder-Mead', \
                            options={'xatol': 10, 'maxiter': 200})
        
        centers = res.x.reshape(-1, 2).astype(int)
        self.bboxes = np.concatenate((centers[:, [0]] - self.patch_size[0], centers[:, [1]] - self.patch_size[1], centers[:, [0]] + self.patch_size[0], centers[:, [1]] + self.patch_size[1]), axis=1)
        
        return self.bboxes
    
    def draw(self):
        plt.figure(figsize=(20,10))
        plt.imshow(self.image)

        for i in self.bboxes:
            plt.gca().add_patch(plt.Rectangle((i[1], i[0]), i[3]-i[1], i[2]-i[0], fill=False,edgecolor='r', linewidth=3))

        plt.show()