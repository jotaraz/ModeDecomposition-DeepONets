#import numpy as np 
#import matplotlib.pyplot as plt 
import sys

import numpy as np
from scipy.optimize import minimize
from itertools import product
import matplotlib.pyplot as plt

#uend = sys.argv[1]
bid = int(sys.argv[1])
modeid = int(sys.argv[2])
#kmode = int(sys.argv[2])
#num_direcs = int(sys.argv[4])

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

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


name, uend, nc = dic(bid)

print(name)

num_samples = int(name.split("_m")[-1])


X = np.loadtxt(base+"/"+name+"_P.txt").T
n = np.shape(X)[0]
xax = np.linspace(0, 1, n)

transformX = np.zeros((nc, n))
for modeid in range(nc):
    tmp = np.sin(np.pi*(1+modeid)*xax)
    transformX[modeid,:] = tmp / np.sqrt(np.sum(tmp**2))

uX, sX, _ = np.linalg.svd(X)

for i in range(5):
    plt.plot(uX[:,i], label=str(sX[i]))
plt.legend()
plt.show()

reconstructX = np.matmul(np.matmul(transformX.T, transformX), X)
print("||X-aX||", np.sum((reconstructX-X)**2), "||X||", np.sum((X)**2))

yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")

X = X[:,:100]
yy = yy[:,:100]

uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T

m = np.shape(X)[1]

print("n", n, "m", m)

Y = vv[:,modeid]


class NonUniformDFT:
    def __init__(self, X, Y, wave_vector_range=(-10, 10), wave_vector_resolution=1.0):
        """
        Initialize the Non-Uniform DFT solver.
        
        Parameters:
        - X: numpy array of shape (n, m) - input points
        - Y: numpy array of shape (m,) - output values
        - wave_vector_range: tuple - range for wave vector components
        - wave_vector_resolution: float - resolution for wave vector grid
        """
        self.X = X
        self.Y = Y
        self.n, self.m = X.shape
        self.wave_vector_range = wave_vector_range
        self.wave_vector_resolution = wave_vector_resolution
        print("initted")
        
    def fourier_basis(self, k, x):
        """
        Compute the Fourier basis function exp(i * k^T * x)
        
        Parameters:
        - k: wave vector (n,)
        - x: input point (n,)
        """
        return np.exp(1j * np.dot(k, x))
    
    def compute_fourier_coefficient(self, k):
        """
        Compute the Fourier coefficient for wave vector k using least squares.
        
        For non-uniform sampling, we solve:
        min_c ||Y - c * exp(i * k^T * X)||^2
        """
        # Compute the basis functions for all sample points
        basis_values = np.array([self.fourier_basis(k, self.X[:, j]) for j in range(self.m)])
        
        # Least squares solution: c = (B^H * B)^(-1) * B^H * Y
        # where B is the basis_values vector
        numerator = np.dot(np.conj(basis_values), self.Y)
        denominator = np.dot(np.conj(basis_values), basis_values)
        
        if np.abs(denominator) < 1e-10:
            return 0.0
        
        return numerator / denominator
    
    def reconstruct_with_wave_vectors(self, wave_vectors, coefficients):
        """
        Reconstruct Y using given wave vectors and coefficients.
        """
        Y_reconstructed = np.zeros(self.m, dtype=complex)
        
        for k, c in zip(wave_vectors, coefficients):
            for j in range(self.m):
                Y_reconstructed[j] += c * self.fourier_basis(k, self.X[:, j])
        
        return Y_reconstructed
    
    def generate_wave_vector_grid(self):
        """
        Generate a grid of wave vectors for testing.
        """
        k_min, k_max = self.wave_vector_range
        k_values = np.arange(k_min, k_max + self.wave_vector_resolution, 
                            self.wave_vector_resolution)
        
        # Generate all combinations for n-dimensional wave vectors
        wave_vectors = list(product(k_values, repeat=self.n))
        return np.array(wave_vectors)
    
    def find_significant_wave_vectors(self, threshold=0.1, max_vectors=None):
        """
        Find wave vectors that contribute significantly to reconstruction.
        
        Parameters:
        - threshold: reconstruction error threshold
        - max_vectors: maximum number of wave vectors to use
        
        Returns:
        - selected_wave_vectors: array of selected wave vectors
        - coefficients: corresponding Fourier coefficients
        - reconstruction_error: final reconstruction error
        """
        # Generate candidate wave vectors
        candidate_wave_vectors = self.generate_wave_vector_grid()
        
        # Compute coefficients for all candidates
        coefficients = []
        for k in candidate_wave_vectors:
            print(k)
            c = self.compute_fourier_coefficient(k)
            coefficients.append(c)
        
        coefficients = np.array(coefficients)
        
        # Sort by magnitude
        magnitudes = np.abs(coefficients)
        sorted_indices = np.argsort(magnitudes)[::-1]
        
        # Greedy selection of wave vectors
        selected_indices = []
        selected_wave_vectors = []
        selected_coefficients = []
        
        for idx in sorted_indices:
            if magnitudes[idx] < 1e-10:
                continue
                
            selected_indices.append(idx)
            selected_wave_vectors.append(candidate_wave_vectors[idx])
            selected_coefficients.append(coefficients[idx])
            
            # Reconstruct with current selection
            Y_reconstructed = self.reconstruct_with_wave_vectors(
                selected_wave_vectors, selected_coefficients
            )
            
            # Compute reconstruction error
            error = np.linalg.norm(self.Y - Y_reconstructed) / np.linalg.norm(self.Y)
            
            if error < threshold:
                break
                
            if max_vectors and len(selected_wave_vectors) >= max_vectors:
                break
        
        return (np.array(selected_wave_vectors), 
                np.array(selected_coefficients), 
                error)
    
    def adaptive_fourier_transform(self, threshold=0.1):
        """
        Perform adaptive Fourier transform with automatic wave vector selection.
        """
        wave_vectors, coefficients, error = self.find_significant_wave_vectors(threshold)
        
        return {
            'wave_vectors': wave_vectors,
            'coefficients': coefficients,
            'reconstruction_error': error,
            'num_components': len(wave_vectors)
        }


def example_usage():
    """
    Example usage with a 2D function sampled non-uniformly.
    """
    # Generate non-uniform sample points
    np.random.seed(42)
    #m = 50  # number of samples
    #n = 2   # dimension
    
    # Non-uniform sampling in [-π, π]^n
    #X = np.random.uniform(-np.pi, np.pi, (n, m))
    
    # Define a test function: f(x) = sin(2*x[0]) + cos(3*x[1])
    #Y = np.sin(2 * X[0, :]) + np.cos(3 * X[1, :])
    
    # Add some noise
    #Y += 0.1 * np.random.randn(m)
    
    # Initialize the NUDFT solver
    nudft = NonUniformDFT(X, Y, wave_vector_range=(-5, 5), wave_vector_resolution=1.0)
    
    # Perform adaptive Fourier transform
    result = nudft.adaptive_fourier_transform(threshold=0.1)
    
    print(f"Number of significant wave vectors: {result['num_components']}")
    print(f"Reconstruction error: {result['reconstruction_error']:.6f}")
    print("\nSignificant wave vectors and coefficients:")
    for k, c in zip(result['wave_vectors'], result['coefficients']):
        if np.abs(c) > 0.01:  # Only print significant ones
            print(f"k = {k}, |c| = {np.abs(c):.4f}, phase = {np.angle(c):.4f}")
    
    # Reconstruct and visualize
    Y_reconstructed = nudft.reconstruct_with_wave_vectors(
        result['wave_vectors'], result['coefficients']
    )
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[0, :], Y, c='blue', label='Original', alpha=0.6)
    plt.scatter(X[0, :], np.real(Y_reconstructed), c='red', label='Reconstructed', alpha=0.6)
    plt.xlabel('x[0]')
    plt.ylabel('y')
    plt.legend()
    plt.title('Original vs Reconstructed (projected on x[0])')
    
    plt.subplot(1, 2, 2)
    plt.scatter(X[1, :], Y, c='blue', label='Original', alpha=0.6)
    plt.scatter(X[1, :], np.real(Y_reconstructed), c='red', label='Reconstructed', alpha=0.6)
    plt.xlabel('x[1]')
    plt.ylabel('y')
    plt.legend()
    plt.title('Original vs Reconstructed (projected on x[1])')
    
    plt.tight_layout()
    plt.show()
    
    return nudft, result


#if __name__ == "__main__":
#    nudft, result = example_usage()