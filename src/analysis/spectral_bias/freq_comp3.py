import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt 
from nfft import nfft
import sys

#bid = int(sys.argv[1])
#nummodes = int(sys.argv[2])

bids = [0, 3, 6]
nummodess = [20, 50, 50]

def get_frequencies_fromfile(bid):
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier_v_2/data/bid"+str(bid)+"_abs.txt", "r")
    lines = f.readlines()
    f.close()

    labels  = []
    freqs   = []
    strings = []
    f0 = []

    for line in lines:
        xx = line[:-1].split(" ")
        labels.append(xx[0])
        vec = np.zeros(len(xx)-1)
        for i,x in enumerate(xx[1:]):
            vec[i] = x 

        maxfreq = np.max(vec)

        tmpstr = "method "+xx[0]+" min0 "+str(vec[0])+" maxf "+str(maxfreq)
        f0.append(vec[0])
        print(tmpstr)
        strings.append(tmpstr)
        freqs.append(vec / maxfreq)
        #print(labels[-1])
        #print(freqs[-1])
        #print("-")
    
    return freqs, labels, tmpstr, f0


def get_colors():
    f = open("/home/johannes/Nextcloud/Documents/Uni/XI/MA/colors.txt", "r")
    lines = f.readlines()
    f.close()
    colors = []
    for line in lines:
        colors.append(line[:-1])

    return colors #print(colors)
colors = get_colors()


base = "data" #"/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

fig, axs = plt.subplots(1, 3, figsize=(14,6), sharey=True)


def dic(tag):
    if tag == 0:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1000", 20
    if tag == 1:
        return "advdiffnx201_dt0.0005_nc20_m1000", "1999", 20
    if tag == 2:
        return "kdvnx401_dt0.0001_nc5_m5000", "10", 5
    if tag == 3:
        return "kdvnx401_dt0.0001_nc5_m5000", "1999", 5
    if tag == 4:
        return "kdvnx401_dt0.0001_nc5_m5000", "5999", 5
    if tag == 5:
        return "kdvnx401_dt0.0001_nc5_m5000", "9999", 5    
    if tag == 6:
        return "burgers_dt0.0001_nc10_m3800", "100", 10
    if tag == 7:
        return "burgers_dt0.0001_nc10_m3800", "999", 10

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
    def __init__(self, X, nummodes):
        """
        Compare frequency content of two functions sampled at same points.
        
        Parameters:
        - X: array of shape (n, m) - sample points
        - Y1: array of shape (m,) - function g1 values
        - Y2: array of shape (m,) - function g2 values
        """
        self.X = X
        self.nummodes = nummodes

        #self.Y1 = Y1
        #self.Y2 = Y2
        self.n, self.m = X.shape


        self.uX_SVD, self.sX, _ = np.linalg.svd(X)
        print(self.sX[:10])

        reconstruct(X, self.uX_SVD, 5)
        
        self.uX_Harm = generatebasis(self.nummodes, self.n)

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

            mean_freqs[i] = kmed #xf[np.argmax(yf)] #np.sum(xf*yf**2) / np.sum(yf**2)
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
        color = ["blue", "green", "orange", "cyan", "purple"]

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
            axs[2].plot(allfreqs[:,i], color=color[i])


        print(np.shape(totalfreq))
        axs[2].plot(totalfreq, '.-', color="red")

        for i in range(param):
            axs[3].plot(self.uX[:,i], color=color[i], label=str(i))
        axs[3].legend()


    def get_frequencies(self, Ys, ax, ib, bid):   

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

        params0 = [[5], [5], [1], [1],
                  [3, 50],
                  [3, 50],
                  [3, 50],
                  [3, 50],
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
        #frequencies = np.zeros((num_sets, num_fcts_total)) #len(functions)))
        
        names = ["Projection", r"TV $k=3$", r"TV $k=50$", r"LE $k=3$", r"LE $k=50$"]

        frequencies, labels, tmpstr, f0 = get_frequencies_fromfile(bid)

        num_methods = len(frequencies)


        #_, spectrums = self.projection(Ys[:,0], param=5)
        #for i in range(5):
        #    plt.plot(spectrums[:,i], label=str(i))


        #for iy in range(num_sets): # ,Y in enumerate(Ys):
        #    print("mode", iy)
        #    k = 0
        #    for indf,fct in enumerate(functions):
        #        for par in params[indf]:
        #            frequencies[iy,k] = fct(Ys[:,iy], param=par)
        #            print("fct", fctnames[indf], par, frequencies[iy,k])
        #            k += 1
        
        # ["red", "blue", "green", "orange"]
        marker = ['-', '.-', 'x-', '-.', '--'] #["s-", "-.", ":", "-", ".-"]

        xax = np.linspace(0, 1, len(frequencies[0]))

        for i in range(num_methods):
            ax.plot(xax, frequencies[i], marker[i], color=colors[ib], markersize=9)
            ax.plot([], [], marker[i], color="gray", markersize=9, label=names[i])

        
        '''
        k = 0
        for i in range(len(functions)):
            for ip in range(len(params[i])):
                tag = str(fctnames[i])
                if "Proj" in fctnames[i]:
                    tag += "" # r" $J=$"
                else:
                    tag += r" $k=$" + str(params[i][ip])


                tmpx = np.arange(0, len(frequencies))
                tmpy = frequencies[:,k]

                ax.set_xlim((-0.1*len(frequencies), 1.1*len(frequencies)))

                ax.plot(tmpx, tmpy / np.max(frequencies[:,k]), marker[k], color=colors[ib], markersize=9)#, label=tag)

                ax.plot([], [], marker[k], color="gray", markersize=9, label=tag)

                k += 1
        '''

        ax.set_xlabel(r"Mode index $i$", fontsize=16)
        if ib == 0:
            ax.legend(loc='lower right', fontsize=16)
            ax.set_ylabel(r"Relative frequency $f_i/(\max_j f_j)$", fontsize=16)



    
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

    for ib, bid in enumerate(bids):        
        name, uend, nc = dic(bid)

        X = np.loadtxt(base+"/"+name+"_P.txt").T
        yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")
        X = X[:,:1000]
        yy = yy[:,:1000]

        uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
        vv = vvhh.T

        n = np.shape(X)[0]
        m = np.shape(X)[1]

        print("n", n, "m", m, np.shape(vv))

        vv = vv[:,:nummodess[ib]]

        print("n", n, "m", m, np.shape(vv))


        comparator = FunctionFrequencyComparator(X, nummodess[ib])
        comparator.get_frequencies(vv, axs[ib], ib, bid)



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
    frequency_comparison()

for ib in range(len(bids)):
    axs[ib].set_ylim((-0.1,1.1))


axs[0].text(0, -0.05, "A)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)
axs[1].text(0, -0.05, "B)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)
axs[2].text(0, -0.05, "C)", horizontalalignment='left', verticalalignment='bottom', fontsize=32)


axs[1].arrow(32/49, 0.3, 0, 0.09, width=0.02, color="fuchsia")
axs[1].arrow(36/49, 0.48, 0, 0.09, width=0.02, color="fuchsia")

axs[2].arrow(37/49, 0.65, 0, 0.09, width=0.02, color="fuchsia")

for i in range(3):
    n = nummodess[i]
    axs[i].set_xticks([0, int(n/2)/(n-1), 1.0], ["1", str(int(n/2)), str(n)])


plt.subplots_adjust(left=0.1, bottom=0.09, right=0.95, top=0.9, wspace=0.0, hspace=0.2) #[source]

#plt.savefig("/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/imgs/fourier_v_2/bid"+str(bid)+"_nc"+str(nummodes)+"_rayleigh_ylim.pdf")

plt.show()

