# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 17:23:56 2025

@author: J_Taraz
"""
#tag:10_02_2026

## like freq_comparison.py but specifically for synthetic data, where P and U are gotten via a function call


import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
from nfft import nfft
import sys

from synthetic_gramschmidt import *

def get_colors():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "colors_cb.txt"), "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)

colors = get_colors()

## 2025.09.25
## the params = (0.1, -1.0, 20, 100, 2_000, 0.1)
## (together with synthetic_gramschmidt)
## are reasonable to get increasing frequencies according to TV and LE (projection is shit)
## e.g. m=10_000 (instead of 2_000) makes them frequency estimations even closer to monotonically increasing
## it hugely depends on the distribution from which the x_i and the k_j are drawn

## when choosing freqscale=-0.1 its nicely decreasing :)




#bid = int(sys.argv[1])

bids = [131, 132, 135] #111, 115, 116, 121, 125, 126]

show_singvals = False

#freqscale  = float(sys.argv[1])
#sigmaexp   = float(sys.argv[2])
#N          = int(sys.argv[3])
#n          = int(sys.argv[4])
##m          = int(sys.argv[5])#
#rf_x       = float(sys.argv[6])
#norm0scale = float(sys.argv[7])
#rf_k      = float(sys.argv[7])

#rf_k = 0.1

#scale = freqscale


N = 5
nummodes = N

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "sb_data") + os.sep

fullm = False
relative = False

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

## synthetic data generation stuff


def lambda_j(x, j):
    return np.sin(2*np.pi * (j+1) * x)

## if p's are chosen as sin(2 pi (j+1) x), then divide by sqrt(12) and sqrt(8)
## if p's are chosen as sin(pi (j+1) x), divide both by 1

def rho_j(x, kj, j):
    #y = np.exp(2*np.pi*1j * np.sum(kj*x)) - 1
    y = np.sin(2 * np.pi * np.sum(kj * x))
    return y
    if j == 0:
        return y/np.sqrt(12)
    else:
        return y/np.sqrt(8)

def get_synthetic_data():
    #n = 100
    #m = 5000
    # fix M = n (number of eval coords in input and output is equal)
    
    rs = np.linspace(0, 1, n)
    ## assemple Phi
    Phi = np.zeros((n, N))
    for j in range(N):
        Phi[:,j] = lambda_j(rs, j)
    
    #rf_x = 0.0
    #rf_k = 0.0
    
    ## assemple X (=P.T)
    X = np.zeros((n, m))
    for i in range(m):
        X[:,i] = np.random.rand() * (np.ones(n)  + rf_x*(np.random.rand(n)*2 - 1.0))
        #X[:,i] = np.sin(2.0/2.0 * np.pi * (i+1) * rs)
    
    ## assemble ks
    ks = np.zeros((n, N))
    knorm = np.zeros(N)
    for j in range(N):
        ktmp = np.ones(n) + rf_k*(np.random.rand(n)*2 - 1.0) #np.random.rand(n)*2 - 1.0 #np.sin(2 * np.pi * (j+1) * rs)
        ktmp /= np.sqrt(np.sum(ktmp**2))
        ks[:,j] = ktmp *np.exp(scale*j)
        knorm[j] = np.sum( ks[:,j]**2 )
        
    ## assemble V 
    V = np.zeros((m, N), dtype=complex)
    for j in range(N):
        for i in range(m):
            V[i,j] = rho_j(X[:,i], ks[:,j], j)
            
    ## normalize V
    VTV = np.matmul(V.conj().T, V)
    VTV_abs = np.sqrt(VTV.real**2 + VTV.imag**2)
    
    for j in range(N):
        V[:,j] /= np.sqrt(VTV_abs[j,j])
    
    VTV = np.matmul(V.conj().T, V)
    VTV_abs = np.sqrt(VTV.real**2 + VTV.imag**2)

    mask_diag = np.eye(N)
    mask_offdiag = np.ones((N,N)) - mask_diag
    print("max diag", np.max(np.diag(VTV_abs)), "min diag", np.min(np.diag(VTV_abs)), "max offdiag", np.max(mask_offdiag * VTV_abs))
    
    S = np.exp(-np.arange(0, N, 1))
    
    return X, Phi, S, V.conj().T, knorm
    

## fourier stuff

def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 50
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 50
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 50
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 50    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 50
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 50

    tag0 = 21
    for sign in [+1.0, -1.0]:
        for sigm in [-1.0,-0.5,-0.01]:
            for fs in [0.05, 0.1, 0.2, 0.4]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv2_n100_N20_m5000", f"fs{bs}ss{ss}", 20

                tag0 += 1
    
    tag0 = 51
    for sign in [+1.0]:
        for sigm in [-1.0,-0.5]:
            for fs in [0.05, 0.1, 0.2, 0.4, 1.0]:
                bs = sign*fs
                ss = sigm
                if tag0 == tag:
                    return "synthv3_n100_N5_m5000", f"fs{bs}ss{ss}", 5

                tag0 += 1

    tag0 = 61 
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv4_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    tag0 = 201 
    # truncated v4:
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv18_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # flipped v4:
    tag0 = 211 
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv19_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 71        
    freqincscales = [ -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0), 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            if tag0 >= 81:
                tmp = "v6"
            else:
                tmp = "v5"
            return "synth"+tmp+"_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 91
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv7_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # truncated v7:
    tag0 = 181
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv16_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # flipped v7:
    tag0 = 191
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv17_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    
    tag0 = 101
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv8_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 111
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv9_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # truncated v9:
    tag0 = 161
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv14_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # flipped v9:
    tag0 = 171
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv15_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
        
    tag0 = 121
    freqincscales = [-0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0] #, -0.2,  -0.2,  -0.2,  -0.2,  -0.4,  -1.0]
    sigmaexpss    = [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [0.2*np.exp(4*0.2), 0.4*np.exp(4*0.2), 1.0*np.exp(4*0.2), 2.0*np.exp(4*0.2), 0.2*np.exp(4*0.4), 0.2*np.exp(4*1.0)] #, 0.2*np.exp(5*0.2),  0.4*np.exp(5*0.2),  1.0*np.exp(5*0.2),  2.0*np.exp(5*0.2),  0.2*np.exp(5*0.4),  0.2*np.exp(5*1.0)]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv10_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(round(norm0scales[i], 4)), 5
        tag0 += 1
    
    tag0 = 131
    freqincscales = [ 0.2,  0.2,  0.2,  0.2,  0.4,  1.0]
    sigmaexpss    = [-0.0001, -0.0001, -0.0001, -0.0001, -0.0001, -0.0001] #, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1]
    norm0scales   = [ 0.2,  0.4,  1.0,  2.0,  0.2,  0.2]
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv11_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # truncated v11:
    tag0 = 141
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv12_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
    # flipped v11:
    tag0 = 151
    for i in range(len(freqincscales)):
        if tag0 == tag:
            return "synthv13_n100_N5_m5000", "fs"+str(freqincscales[i])+"ss"+str(sigmaexpss[i])+"ns"+str(norm0scales[i]), 5
        tag0 += 1
        
    return None    


def eventrigonometric(n, nc):
    transformX = np.zeros((nc, n))
    xax = np.linspace(0, 1, n)
    for modeid in range(nc):
        tmp = np.sin(2*np.pi*(1+modeid)*xax)
        transformX[modeid,:] = tmp / np.sqrt(np.sum(tmp**2))
    return transformX.T

def alltrigonometric(n, nc):
    transformX = np.zeros((nc, n))
    if n != 50:
        print("alltrigonometric only defined for n=50")
        return transformX.T
    xax = np.linspace(0, 1, 200)
    xax = xax[::4]
    for modeid in range(nc):
        tmp = np.sin(np.pi*(1+modeid)*xax)
        transformX[modeid,:] = tmp / np.sqrt(np.sum(tmp**2))
    return transformX.T

def generatebasis(nc, n):
    if n == 50:
        return alltrigonometric(n, nc)
    else:
        return eventrigonometric(n, nc)

    '''
    transformX = np.zeros((nc, n))
    xax = np.linspace(0, 1, n)
    for modeid in range(nc):
        tmp = np.sin(2*np.pi*(1+modeid)*xax)
        transformX[modeid,:] = tmp / np.sqrt(np.sum(tmp**2))
    return transformX.T
    '''

def reconstruct(X, uX, param):
    reconstruct = np.matmul(np.matmul(uX[:,:param], uX[:,:param].T), X)
    print("rec error", np.max(np.abs(reconstruct-X)))


class FunctionFrequencyComparator:
    def __init__(self, X):
        """
        Compare frequency content of two functions sampled at same points.
        
        Parameters:
        - X: array of shape (n, m) - sample points
        - Y1: array of shape (m,) - function g1 values
        - Y2: array of shape (m,) - function g2 values
        """
        self.X = X

        #self.Y1 = Y1
        #self.Y2 = Y2
        self.n, self.m = X.shape


        self.uX_SVD, self.sX, _ = np.linalg.svd(X)
        print(self.sX[:10])

        reconstruct(X, self.uX_SVD, 5)
        
        self.uX_Harm = generatebasis(nummodes, self.n)

        reconstruct(X, self.uX_Harm, 5)
   
    def total_variation(self, Y, X=None, param=5):
        k_neighbors = param
        """
        Compute total variation using k-nearest neighbors.
        Higher values indicate higher frequency content.
        
        This is a discretization of: TV(f) = ∫ |∇f(x)| dx
        """
        if X is None:
            X = self.X
            
        # Find k-nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(X.T)
        distances, indices = nbrs.kneighbors(X.T)
        
        tv = 0.0
        for i in range(self.m):
            # Skip the point itself (distance 0)
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Compute gradient approximation
            for j, dist in zip(neighbor_indices, neighbor_distances):
                if dist > 0:
                    # Finite difference approximation
                    gradient_approx = abs(Y[i] - Y[j]) / dist
                    tv += gradient_approx
                    
        return tv / self.m
    
    def laplacian_energy(self, Y, param=10):
        k_neighbors = param
        """
        Compute discrete Laplacian energy.
        Higher values indicate higher frequency content.
        
        Based on: E(f) = ∫ |Δf(x)|² dx
        """
        # Build graph Laplacian based on k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(self.X.T)
        distances, indices = nbrs.kneighbors(self.X.T)
        
        # Compute weighted Laplacian
        laplacian_values = np.zeros(self.m)
        
        for i in range(self.m):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Gaussian weights
            weights = np.exp(-neighbor_distances**2 / (2 * np.median(neighbor_distances)**2))
            weights /= weights.sum()
            
            # Discrete Laplacian: Δf(xi) ≈ Σ w_ij (f(xj) - f(xi))
            laplacian_values[i] = np.sum(weights * (Y[neighbor_indices] - Y[i]))
        
        # Energy is the squared norm
        return np.mean(laplacian_values**2)
    
    def laplacian_energy_rayleigh_sym(self, Y, param=10):
        k_neighbors = param
        """
        Compute discrete Laplacian energy.
        Higher values indicate higher frequency content.
        
        Based on: E(f) = ∫ |Δf(x)|² dx
        """
        # Build graph Laplacian based on k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(self.X.T)
        distances, indices = nbrs.kneighbors(self.X.T)
        
        L = np.zeros((self.m, self.m))
        for i in range(self.m):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Gaussian weights
            weights = np.exp(-neighbor_distances**2 / (2 * np.mean(neighbor_distances**2)))
            weights /= weights.sum()
            L[i,i] = weights.sum()
            for ij,j in enumerate(neighbor_indices):
                L[i,j] = -weights[ij]
        
        A = L + L.T

        return np.dot(Y, np.matmul(A, Y))/np.sum(Y*Y)

    def laplacian_energy_rayleigh(self, Y, param=10):
        k_neighbors = param
        """
        Compute discrete Laplacian energy.
        Higher values indicate higher frequency content.
        
        Based on: E(f) = ∫ |Δf(x)|² dx
        """
        # Build graph Laplacian based on k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors+1).fit(self.X.T)
        distances, indices = nbrs.kneighbors(self.X.T)
        
        L = np.zeros((self.m, self.m))
        for i in range(self.m):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Gaussian weights
            weights = np.exp(-neighbor_distances**2 / (2 * np.mean(neighbor_distances)**2))
            weights /= weights.sum()
            L[i,i] = weights.sum()
            for ij,j in enumerate(neighbor_indices):
                L[i,j] = -weights[ij]
        
        return np.dot(Y, np.matmul(L, Y))/np.sum(Y*Y)

        '''


        # Compute weighted Laplacian
        laplacian_values = np.zeros(self.m)
        
        for i in range(self.m):
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Gaussian weights
            weights = np.exp(-neighbor_distances**2 / (2 * np.mean(neighbor_distances)**2))
            weights /= weights.sum()
            
            # Discrete Laplacian: Δf(xi) ≈ Σ w_ij (f(xj) - f(xi))
            laplacian_values[i] = np.sum(weights * (Y[neighbor_indices] - Y[i]))
        
        # Energy is the squared norm
        return np.mean(laplacian_values**2)
    '''

    def modulus_of_continuity(self, Y, param=1000):
        num_samples = param
        """
        Estimate modulus of continuity: ω(h) = sup{|f(x) - f(y)| : |x-y| ≤ h}
        
        Functions with slower decay of ω(h) have higher frequency content.
        """
        # Randomly sample pairs of points
        idx1 = np.random.randint(0, self.m, num_samples)
        idx2 = np.random.randint(0, self.m, num_samples)
        
        # Compute distances and function differences
        distances = np.linalg.norm(self.X[:, idx1] - self.X[:, idx2], axis=0)
        differences = np.abs(Y[idx1] - Y[idx2])
        
        # Sort by distance
        sort_idx = np.argsort(distances)
        distances = distances[sort_idx]
        differences = differences[sort_idx]
        
        # Compute modulus of continuity at different scales
        h_values = np.percentile(distances[distances > 0], [10, 25, 50, 75, 90])
        omega_values = []
        
        for h in h_values:
            mask = distances <= h
            if np.any(mask):
                omega_values.append(np.max(differences[mask]))
            else:
                omega_values.append(0)
                
        # Return the decay rate (slope in log-log plot)
        # Slower decay (less negative slope) = higher frequency
        log_h = np.log(h_values + 1e-10)
        log_omega = np.log(np.array(omega_values) + 1e-10)
        
        # Fit line in log-log space
        slope = np.polyfit(log_h, log_omega, 1)[0]
        
        return slope #, h_values, omega_values
    
    def variance_based_measure(self, Y, scale_factors=[0.5, 1.0, 2.0, 4.0]):
        """
        Multi-scale variance measure.
        High frequency functions have high local variance.
        """
        variances = []
        
        for scale in scale_factors:
            # Define local neighborhoods at this scale
            radius = scale * np.median(cdist(self.X.T, self.X.T))
            
            local_variances = []
            for i in range(self.m):
                # Find points within radius
                distances = np.linalg.norm(self.X - self.X[:, i:i+1], axis=0)
                mask = distances <= radius
                
                if np.sum(mask) > 1:
                    local_var = np.var(Y[mask])
                    local_variances.append(local_var)
            
            if local_variances:
                variances.append(np.mean(local_variances))
        
        # Higher frequency content shows more variance at smaller scales
        if len(variances) > 1:
            # Ratio of small-scale to large-scale variance
            return variances[0] / (variances[-1] + 1e-10)
        else:
            return variances[0] if variances else 0
    
    def roughness_penalty(self, Y, p=2):
        """
        Compute roughness penalty using finite differences.
        Based on thin-plate spline energy for p=2.
        """
        # For each point, estimate derivatives using neighbors
        nbrs = NearestNeighbors(n_neighbors=min(2*self.n+1, self.m)).fit(self.X.T)
        
        roughness = 0.0
        count = 0
        
        for i in range(self.m):
            distances, indices = nbrs.kneighbors(self.X[:, i:i+1].T)
            neighbor_idx = indices[0, 1:]  # Exclude self
            
            if len(neighbor_idx) >= self.n + 1:
                # Fit local polynomial and compute derivatives
                X_local = self.X[:, neighbor_idx].T - self.X[:, i]
                Y_local = Y[neighbor_idx] - Y[i]
                
                # Compute pseudo-inverse for linear fit
                try:
                    coeffs = np.linalg.lstsq(X_local, Y_local, rcond=None)[0]
                    # Roughness is norm of gradient
                    roughness += np.linalg.norm(coeffs)**p
                    count += 1
                except:
                    pass
        
        return roughness / (count + 1)
    
    def spectral_gap_statistic(self, Y, param=20):
        num_random_phases = param
        """
        Compare function to random phase versions of itself.
        High frequency functions are more sensitive to phase randomization.
        """
        # Project onto random directions and compute 1D Fourier transforms
        scores = []
        
        for _ in range(num_random_phases):
            # Random projection direction
            direction = np.random.randn(self.n)
            direction /= np.linalg.norm(direction)
            
            # Project points onto this direction
            projections = direction @ self.X
            
            # Sort by projection
            sort_idx = np.argsort(projections)
            Y_sorted = Y[sort_idx]
            
            # Compute discrete differences (approximate derivative)
            differences = np.diff(Y_sorted)
            
            # Measure high frequency content
            score = np.mean(differences**2)
            scores.append(score)
            
        return np.mean(scores)

    def projection(self, Y, param=5):
        X = self.X
        uX = self.uX_SVD

        #print("sX", sX[:10])

        #spectrum = np.zeros((int(self.m/2), param))
        mean_freqs = np.zeros(param)
        for i in range(param):
            projectedX = np.matmul(uX[:,i], X)
            #print(np.shape(projectedX), np.shape(X), np.shape(uX[:,i]))
            indx = np.argsort(projectedX)
            yf = np.abs(nfft(projectedX[indx], Y[indx]))
            xf = np.fft.fftfreq(self.m,1./self.m)

            xf = xf[:int(self.m/2)]
            yf = yf[:int(self.m/2)]

            coef_cumsum = np.cumsum(np.abs(yf)**2)
            imed = np.where(coef_cumsum > coef_cumsum[-1]/2)[0][0]
            kmed = 0.5*(xf[max(0, imed-1)] + xf[imed])

            mean_freqs[i] = np.sum(xf*yf**2) / np.sum(yf**2)
            #mean_freqs[i] = kmed #xf[np.argmax(yf)] #np.sum(xf*yf**2) / np.sum(yf**2)
            #spectrum[:,i] = yf

        return np.mean(mean_freqs) #mean_freqs, spectrum

    def projection_harmonics(self, Y, param=5):
        X = self.X
        uX = self.uX_Harm

        #print("sX", sX[:10])

        #spectrum = np.zeros((int(self.m/2), param))
        mean_freqs = np.zeros(param)
        for i in range(param):
            projectedX = np.matmul(uX[:,i], X)
            #print(np.shape(projectedX), np.shape(X), np.shape(uX[:,i]))
            indx = np.argsort(projectedX)
            yf = np.abs(nfft(projectedX[indx], Y[indx]))
            xf = np.fft.fftfreq(self.m,1./self.m)

            xf = xf[:int(self.m/2)]
            yf = yf[:int(self.m/2)]

            coef_cumsum = np.cumsum(np.abs(yf)**2)
            imed = np.where(coef_cumsum > coef_cumsum[-1]/2)[0][0]
            kmed = 0.5*(xf[max(0, imed-1)] + xf[imed])

            mean_freqs[i] = kmed #xf[np.argmax(yf)] #np.sum(xf*yf**2) / np.sum(yf**2)
            #spectrum[:,i] = yf

        return np.mean(mean_freqs) #mean_freqs, spectrum

    def projection1(self, Y, param=5):
        X = self.X
        #uX, sX, _ = np.linalg.svd(X)
        uX = self.uX
        reconstruct = np.matmul(np.matmul(uX[:,:param], uX[:,:param].T), X)
        print("rec error", np.max(np.abs(reconstruct-X)))

        #print("sX", sX[:10])

        #mean_freqs = np.zeros(param)
        spectrum = np.zeros((int(self.m/2), param))
        freqs    = np.zeros((int(self.m/2), param))


        projs = np.zeros((self.m, param))
        value = np.zeros((self.m, param))
        for i in range(param):
            projectedX = np.matmul(uX[:,i], X)
            #print(np.shape(projectedX), np.shape(X), np.shape(uX[:,i]))
            indx = np.argsort(projectedX)
            yf = np.abs(nfft(projectedX[indx], Y[indx]))
            xf = np.fft.fftfreq(self.m,1./self.m)

            projs[:,i] = projectedX[indx]
            value[:,i] = Y[indx]

            xf = xf[:int(self.m/2)]
            yf = yf[:int(self.m/2)]
            
            #mean_freqs[i] = xf[np.argmax(yf)] #np.sum(xf*yf**2) / np.sum(yf**2)
            freqs[:,i]    = xf
            spectrum[:,i] = yf

        return projs, value, freqs, spectrum #np.mean(mean_freqs) #mean_freqs, spectrum

    def projection2(self, Y, param=5):
        X = self.X
        uX, sX, _ = np.linalg.svd(X)

        #print("sX", sX[:10])

        #spectrum = np.zeros((int(self.m/2), param))
        mean_freqs = np.zeros(param)
        for i in range(param):
            projectedX = np.matmul(uX[:,i], X)
            #print(np.shape(projectedX), np.shape(X), np.shape(uX[:,i]))
            indx = np.argsort(projectedX)
            yf = np.abs(nfft(projectedX[indx], Y[indx]))
            xf = np.fft.fftfreq(self.m,1./self.m)

            xf = xf[:int(self.m/2)]
            yf = yf[:int(self.m/2)]

            mean_freqs[i] = np.sum(xf*yf**2) / np.sum(yf**2)
            #spectrum[:,i] = yf

        return np.mean(mean_freqs) #mean_freqs, spectrum
        
    def lookatprojs(self, Ys, param=5):
        #plt.figure()
        fig, axs = plt.subplots(1, 4)
        k = 0
        x = 0
        xf = 0
        #color = ["blue", "green", "orange", "cyan", "purple"]

        allfreqs = np.zeros((nummodes, param))
        totalfreq = np.zeros(nummodes)

        for i in range(nummodes):
            print(i)
            projs, values, freqs, specs = self.projection1(Ys[:,i])
            weightedsum = 0
            normalsum   = 0
            for par in range(param):
                x = projs[:,par]
                axs[0].plot(x, 0*x-k, '--', color="gray")
                axs[0].plot(x, 0.5*values[:,par]/np.max(np.abs(values))-k)
                
                xf = freqs[:,par]
                yf = specs[:,par]
                axs[1].plot(xf, specs[:,par]/np.max(specs)-k)
                axs[1].plot(xf, 0*xf-k, '--', color="gray")

                a = np.sum(xf * specs[:,par]**2)
                b = np.sum(specs[:,par]**2)

                weightedsum += a
                normalsum   += b

                #axs[1].plot([a/b, a/b], [-k, -k+1], '--', color=color[par])
                coef_cumsum = np.cumsum(np.abs(yf)**2)
                imed = np.where(coef_cumsum > coef_cumsum[-1]/2)[0][0]
                kmed = 0.5*(xf[max(0, imed-1)] + xf[imed])

                allfreqs[i,par] = kmed #a/b

                k += 1
            axs[0].plot(x, 0*x-(k-0.5), '.-', color='k')
            #axs[1].plot([weightedsum/normalsum, weightedsum/normalsum], [-k+0.5, -k+0.5+param], '--', color="red")
            totalfreq[i] = weightedsum/normalsum
            axs[1].plot(xf, 0*xf-(k-0.5), '.-', color='k')
        
        for i in range(param):
            print(np.shape(allfreqs[:,i]))
            axs[2].plot(allfreqs[:,i], color=colors[i])


        print(np.shape(totalfreq))
        axs[2].plot(totalfreq, '.-', color=colors[7])

        for i in range(param):
            axs[3].plot(self.uX[:,i], color=colors[i], label=str(i))
        axs[3].legend()

    def get_frequencies(self, Ys, knorm, ax, ax_id):   


        functions0 = [self.projection,
                     self.projection_harmonics,
                     self.projection1,
                     self.projection2,
                     self.total_variation, 
                     self.laplacian_energy,
                     self.laplacian_energy_rayleigh,
                     self.laplacian_energy_rayleigh_sym, 
                     self.modulus_of_continuity, 
                     self.spectral_gap_statistic
                     #self.variance_based_measure, 
                     #self.roughness_penalty, 
                    ]

        params0 = [[1, 5], [5], [1], [1],
                  [3, 20,  50],
                  [3, 50],
                  [3, 50],
                  [3, 20, 50],
                  [10, 100, 200, 1000],
                  [10, 50, 100, 200]]

        fctnames0 = [r"Projection SVD", 
                     r"Projection Harmonics", 
                     r"P1", 
                     r"P2", 
                     r"Total Variation", 
                     r"LE", 
                     r"Laplacian Energy (unsym)", 
                     r"Laplacian Energy", r"ModCont", r"SpectrGap"]


        fctnames0 = [r"Projection", 
                     r"Proj. Trig.", 
                     r"P1", 
                     r"P2", 
                     r"TV", 
                     r"LE", 
                     r"Laplacian Energy (unsym)", 
                     r"LE", r"ModCont", r"SpectrGap"]


        takethem = [0, 4, 7] #0, 2]
        functions = []
        params    = []
        fctnames  = []
        for ind in takethem:
            functions.append(functions0[ind])
            params.append(params0[ind])
            fctnames.append(fctnames0[ind])

        num_fcts_total = 0
        for i in range(len(params)):
            num_fcts_total += len(params[i])


        num_sets = np.shape(Ys)[1] #len(Ys)
        frequencies = np.zeros((num_sets, num_fcts_total)) #len(functions)))
        
        #_, spectrums = self.projection(Ys[:,0], param=5)
        #for i in range(5):
        #    plt.plot(spectrums[:,i], label=str(i))


        for iy in range(num_sets): # ,Y in enumerate(Ys):
            print("mode", iy, "true_freq", knorm[iy])
            k = 0
            for indf,fct in enumerate(functions):
                for par in params[indf]:
                    Y_tmp = np.sqrt(Ys[:,iy].real**2 + Ys[:,iy].imag**2)
                    frequencies[iy,k] = fct(Y_tmp, param=par)
                    print("fct", fctnames[indf], par, frequencies[iy,k])
                    k += 1
        
        #color = ["red", "blue", "green", "orange"]
        marker = [".-", "--", "-.", ":"]

        outpfreqs = 0*frequencies
        labels   = []

        k = 0
        for i in range(len(functions)):
            for ip in range(len(params[i])):
                tag = str(fctnames[i])
                #if "Proj" in fctnames[i]:
                #    tag += "" # r" $J=$"
                #else:
                tag += "k"+str(params[i][ip]) #r" $k=$" + str(params[i][ip])
                labels.append(tag)

                if relative:
                    outpfreqs[:,k] = frequencies[:,k] / np.max(frequencies[:,k])
                else:
                    outpfreqs[:,k] = frequencies[:,k]

                ax.plot(frequencies[:,k] / np.max(frequencies[:,k]), marker[ip], color=colors[i], label=tag)
                k += 1

        #ax.legend(loc='lower right', fontsize=16)
        if ax_id == 1:
            ax.set_xlabel(r"mode index $i$", fontsize=10)
        if ax_id == 0:
            ax.set_ylabel(r"relative frequency $f_i/(\max_j f_j)$", fontsize=10)

        return outpfreqs, labels

    def compare_all_metrics(self):
        """
        Compare both functions using all metrics.
        Returns positive values when g1 has higher frequency content than g2.
        """
        results = {}
        
        # Total Variation
        tv1 = self.total_variation(self.Y1)
        tv2 = self.total_variation(self.Y2)
        results['total_variation'] = {
            'g1': tv1,
            'g2': tv2,
            'ratio': tv1 / (tv2 + 1e-10),
            'higher_freq': 'g1' if tv1 > tv2 else 'g2'
        }
        
        # Laplacian Energy
        lap1 = self.laplacian_energy(self.Y1)
        lap2 = self.laplacian_energy(self.Y2)
        results['laplacian_energy'] = {
            'g1': lap1,
            'g2': lap2,
            'ratio': lap1 / (lap2 + 1e-10),
            'higher_freq': 'g1' if lap1 > lap2 else 'g2'
        }
        
        # Modulus of Continuity
        slope1, _, _ = self.modulus_of_continuity(self.Y1)
        slope2, _, _ = self.modulus_of_continuity(self.Y2)
        results['modulus_continuity_slope'] = {
            'g1': slope1,
            'g2': slope2,
            'higher_freq': 'g1' if slope1 > slope2 else 'g2'
        }
        
        # Variance-based
        var1 = self.variance_based_measure(self.Y1)
        var2 = self.variance_based_measure(self.Y2)
        results['variance_ratio'] = {
            'g1': var1,
            'g2': var2,
            'ratio': var1 / (var2 + 1e-10),
            'higher_freq': 'g1' if var1 > var2 else 'g2'
        }
        
        # Roughness
        rough1 = self.roughness_penalty(self.Y1)
        rough2 = self.roughness_penalty(self.Y2)
        results['roughness'] = {
            'g1': rough1,
            'g2': rough2,
            'ratio': rough1 / (rough2 + 1e-10),
            'higher_freq': 'g1' if rough1 > rough2 else 'g2'
        }
        
        # Spectral gap
        spec1 = self.spectral_gap_statistic(self.Y1)
        spec2 = self.spectral_gap_statistic(self.Y2)
        results['spectral_gap'] = {
            'g1': spec1,
            'g2': spec2,
            'ratio': spec1 / (spec2 + 1e-10),
            'higher_freq': 'g1' if spec1 > spec2 else 'g2'
        }
        
        # Consensus
        votes_g1 = sum(1 for metric in results.values() if metric['higher_freq'] == 'g1')
        votes_g2 = len(results) - votes_g1
        results['consensus'] = 'g1' if votes_g1 > votes_g2 else 'g2'
        
        return results



def frequency_comparison():
    """Test the frequency comparison methods."""
    np.random.seed(7)

    #name, uend, nc = dic(bid)

    #print(name)
    #num_samples = int(name.split("_m")[-1])


    #X, uu, ss, vvhh, knorm = get_synthetic_data()
    
    # this is usually it, directly from synthetic stuff (2025.10.05)
    #knorm, X, uu, ss, vvhh, _ = generate_Xusvh_varfreq(N, n, m, freqscale, sigmaexp, rf_x, norm0scale=norm0scale, verbose=True)
    # i just wanna try:
        
    fig, axs = plt.subplots(1, len(bids), sharey=True, figsize=(5,2))
    for ib, bid in enumerate(bids):
        ax = axs[ib]
        name, uend, nc = dic(bid)
        print(bid, name, uend)
        print(uend.split("s"))
        fs = float(uend.split("s")[1])
        ns = float(uend.split("s")[4])
        ax.set_title(r"$F_0 =$ "+str(ns)+r", $\alpha =$ "+str(fs), fontsize=10)
        knorm = ns * np.exp(fs * np.arange(0, 5, 1))
        
        X = np.loadtxt(base+name+"_P.txt").T
        yy = np.loadtxt(base+name+"_"+uend+"_U.txt")
        #X = X[:,:1000]
        #yy = yy[:,:1000]
    
        uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
        vv = vvhh.conj().T
    
        n = np.shape(X)[0]
        m = np.shape(X)[1]
    
        
        print("n", n, "m", m, np.shape(vv))
    
        vv = vv[:,:nummodes]
    
        print("n", n, "m", m, np.shape(vv))
    
        comparator = FunctionFrequencyComparator(X)
        relfreqs, labels = comparator.get_frequencies(vv, knorm, ax, ib)
            
        ax.plot(knorm/np.max(knorm), 'o-', color=colors[3], label="True Freq", linewidth=3, alpha=0.7, zorder=2)
        
        ax.set_xticks([0, 2, 4], [1, 3, 5])
        
        if show_singvals:
            ax2 = ax.twinx()
            ax2.plot(ss[:nummodes], '.-', color="k")
        

def test_frequency_comparison():
    """Test the frequency comparison methods."""
    np.random.seed(42)
    
    # Generate test data
    n = 10  # dimensions
    m = 500  # samples
    
    # Non-uniform sampling
    X = np.random.randn(n, m)
    
    Y1 = np.zeros((m,))
    Y2 = np.zeros((m,))

    ks = np.arange(0, 30)
    coeffs1 = np.exp(-2*ks)
    coeffs2 = 1.0/(1+ks)
    
    vecs = np.zeros((len(ks), n))
    for k in ks:
        vec = 2.0*np.random.rand(n)-1.0
        vec = vec/np.sqrt(np.sum(vec**2))
        vecs[k,:] = vec

    for i in range(m):
        x = X[:,i]
        for k in ks:
            Y1[i] += coeffs1[k]*np.sum(x*vecs[k])
            Y2[i] += coeffs2[k]*np.sum(x*vecs[k])



    '''
    # g1: Low frequency function
    # Smooth function with a few low-frequency modes
    k1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0]])
    Y1 = np.sum([np.sin(k @ X) for k in k1], axis=0)
    print(np.shape(Y1))
    
    # g2: High frequency function
    # Function with many high-frequency modes
    #k2 = np.random.randn(20, n) * 3  # Higher frequency
    #Y2 = np.sum([0.3 * np.sin(k @ X) for k in k2], axis=0)
    k2 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 0, 0, 0, 0, 0, 2, 0],
                   [0, 0, 0.5, 0, 0, 0, 0, -3, 0, 0]])
    Y2 = np.sum([np.sin(k @ X) for k in k1], axis=0)
    '''

    # Add same noise level to both
    #noise = 0.1 * np.random.randn(m)
    #Y1 += noise
    #Y2 += noise
    
    # Compare
    comparator = FunctionFrequencyComparator(X, Y1, Y2)
    results = comparator.compare_all_metrics()
    
    print("Frequency Content Comparison")
    print("=" * 50)
    print("(Higher values indicate higher frequency content)\n")
    
    for metric, values in results.items():
        if metric != 'consensus':
            print(f"{metric}:")
            if 'g1' in values:
                print(f"  g1: {values['g1']:.6f}")
                print(f"  g2: {values['g2']:.6f}")
            if 'ratio' in values:
                print(f"  ratio (g1/g2): {values['ratio']:.3f}")
            print(f"  Higher frequency: {values['higher_freq']}")
            print()
    
    print(f"CONSENSUS: Function {results['consensus']} has higher frequency content")
    
    return comparator, results


if __name__ == "__main__":
    #comparator, results = 
    #get_synthetic_data()
    frequency_comparison()

#plt.ylim((-0.1,1.1))

#plt.savefig(r"/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier_v_2/bid"+str(bid)+"_nc"+str(nummodes)+"_rayleigh_ylim_fullm.pdf")



plt.subplots_adjust(bottom=0.20, right=0.97)

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "figures")
tmp = "Fig12"
plt.savefig(path+"/pdfs/"+tmp+".pdf")
plt.savefig(path+"/pngs/"+tmp+".png")
