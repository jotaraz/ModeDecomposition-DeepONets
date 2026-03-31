import sys
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy.optimize import differential_evolution
import time


#uend = sys.argv[1]
bid = int(sys.argv[1])
modeid = int(sys.argv[2])
kmaxval = float(sys.argv[3])
numiter = int(sys.argv[4])
#kmode = int(sys.argv[2])
#num_direcs = int(sys.argv[4])

base = "/home/johannes/Nextcloud/Documents/Uni/XI/MA/new2/data"

def stringify(xx):
    s = str(xx[0])
    
    for x in xx[1:]:
        s += " "+str(x)
    
    return s


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


yy = np.loadtxt(base+"/"+name+"_"+uend+"_U.txt")

X = X[:,:1000]
yy = yy[:,:1000]

uu, ss, vvhh = np.linalg.svd(yy, full_matrices=False)
vv = vvhh.T

n = np.shape(X)[0]
m = np.shape(X)[1]

print("n", n, "m", m)

Y = vv[:,modeid]

class MemoryEfficientNUDFT:
    def __init__(self, X, Y):
        """
        Memory-efficient Non-Uniform DFT for high-dimensional data.
        
        Parameters:
        - X: numpy array of shape (n, m) - input points
        - Y: numpy array of shape (m,) - output values
        """
        self.X = X
        self.Y = Y
        self.n, self.m = X.shape
        
        # Precompute norms for efficiency
        self.X_norms = np.linalg.norm(X, axis=0)
        self.Y_norm = np.linalg.norm(Y)
        
    def fourier_basis_batch(self, K, X_batch=None):
        """
        Efficiently compute Fourier basis for multiple wave vectors and points.
        
        Parameters:
        - K: array of wave vectors, shape (num_k, n)
        - X_batch: subset of points, shape (n, batch_size). If None, use all X
        
        Returns:
        - basis matrix of shape (batch_size, num_k)
        """
        if X_batch is None:
            X_batch = self.X
            
        # Efficient matrix multiplication: K @ X_batch
        # K is (num_k, n), X_batch is (n, m) -> result is (num_k, m)
        phase = 1j * np.dot(K, X_batch)
        return np.exp(phase).T  # Transpose to get (m, num_k)
    
    def compute_coefficients_batch(self, K):
        """
        Compute Fourier coefficients for multiple wave vectors using least squares.
        
        Parameters:
        - K: array of wave vectors, shape (num_k, n)
        
        Returns:
        - coefficients: array of shape (num_k,)
        """
        # Build the basis matrix efficiently
        A = self.fourier_basis_batch(K)  # Shape: (m, num_k)
        
        # Solve least squares: min ||Y - A @ c||^2
        # Using normal equations: c = (A^H @ A)^(-1) @ A^H @ Y
        AH_A = np.conj(A.T) @ A  # Shape: (num_k, num_k)
        AH_Y = np.conj(A.T) @ self.Y  # Shape: (num_k,)
        
        # Solve the system
        try:
            coefficients = np.linalg.solve(AH_A + 1e-10 * np.eye(len(K)), AH_Y)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            coefficients, _, _, _ = np.linalg.lstsq(A, self.Y, rcond=None)
            
        return coefficients
    
    def random_search_wave_vectors(self, num_iterations=5000, num_candidates_per_iter=10,
                                 k_max=10.0, threshold=0.1, verbose=True):
        """
        Random search for significant wave vectors.
        
        Parameters:
        - num_iterations: number of search iterations
        - num_candidates_per_iter: wave vectors to test per iteration
        - k_max: maximum magnitude for wave vector components
        - threshold: reconstruction error threshold
        - verbose: print progress
        
        Returns:
        - Dictionary with results
        """
        selected_K = []
        selected_coeffs = []
        
        best_error = 1.0
        Y_reconstructed = np.zeros_like(self.Y, dtype=complex)
        
        for iter in range(num_iterations):
            # Generate random wave vectors
            if iter < num_iterations // 3:
                # Start with smaller wave vectors
                scale = k_max * (iter + 1) / (num_iterations // 3)
            else:
                scale = k_max
                
            # Sample wave vectors with different strategies
            if iter % 3 == 0:
                # Sparse wave vectors (many zeros)
                K_candidates = np.zeros((num_candidates_per_iter, self.n))
                for i in range(num_candidates_per_iter):
                    # Only set a few non-zero components
                    num_nonzero = np.random.randint(1, min(10, self.n))
                    indices = np.random.choice(self.n, num_nonzero, replace=False)
                    K_candidates[i, indices] = np.random.uniform(-scale, scale, num_nonzero)
            elif iter % 3 == 1:
                # Gaussian distributed
                K_candidates = np.random.randn(num_candidates_per_iter, self.n) * scale / 3
            else:
                # Uniform distributed
                K_candidates = np.random.uniform(-scale, scale, 
                                               (num_candidates_per_iter, self.n))
            
            # Compute coefficients for candidates
            coeffs = self.compute_coefficients_batch(K_candidates)
            
            # Find the best candidate
            errors = []
            for i, (k, c) in enumerate(zip(K_candidates, coeffs)):
                # Compute contribution of this wave vector
                contribution = c * self.fourier_basis_batch(k.reshape(1, -1)).flatten()
                
                # Test reconstruction with this additional component
                Y_test = Y_reconstructed + contribution
                error = np.linalg.norm(self.Y - Y_test) / self.Y_norm
                errors.append(error)
            
            # Select best candidate
            best_idx = np.argmin(errors)
            if errors[best_idx] < best_error:
                best_error = errors[best_idx]
                selected_K.append(K_candidates[best_idx])
                selected_coeffs.append(coeffs[best_idx])
                
                # Update reconstruction
                contribution = coeffs[best_idx] * self.fourier_basis_batch(
                    K_candidates[best_idx].reshape(1, -1)
                ).flatten()
                Y_reconstructed += contribution
                
                if verbose and (len(selected_K) % 10 == 0 or best_error < threshold):
                    print(f"Iteration {iter}: Selected {len(selected_K)} vectors, "
                          f"error = {best_error:.6f}")
                
                if best_error < threshold:
                    break
        
        return {
            'wave_vectors': np.array(selected_K),
            'coefficients': np.array(selected_coeffs),
            'reconstruction_error': best_error,
            'num_components': len(selected_K),
            'Y_reconstructed': Y_reconstructed
        }
    
    def matching_pursuit(self, num_iterations=1000, k_max=10.0, threshold=0.1, 
                        batch_size=250, verbose=True):
        """
        Orthogonal Matching Pursuit adapted for continuous wave vectors.
        
        Parameters:
        - num_iterations: maximum iterations
        - k_max: maximum magnitude for wave vector components  
        - threshold: reconstruction error threshold
        - batch_size: number of candidates to test per iteration
        - verbose: print progress
        """
        selected_K = []
        kmeans = []
        residual = self.Y.copy()
        
        for iter in range(num_iterations):
            # Generate candidate wave vectors
            if iter == 0:
                # Start with coordinate vectors
                K_candidates = np.zeros((self.n, self.n))
                np.fill_diagonal(K_candidates, 1.0)
                # Add some random ones
                #print(batch_size - self.n)
                K_random = np.random.randn(batch_size - self.n, self.n) * k_max / 3
                K_candidates = np.vstack([K_candidates, K_random])
            else:
                # Use gradient information from residual
                # Compute gradient of ||residual - c*exp(i*k'*X)||^2 w.r.t. k
                K_candidates = self._generate_smart_candidates(
                    residual, selected_K, k_max, batch_size
                )
            
            # Compute correlations with residual
            correlations = []
            for k in K_candidates:
                basis = self.fourier_basis_batch(k.reshape(1, -1)).flatten()
                correlation = np.abs(np.vdot(basis, residual))
                correlations.append(correlation)
            
            # Select best candidate
            best_idx = np.argmax(correlations)
            best_k = K_candidates[best_idx]
            
            # Add to selected set
            selected_K.append(best_k)
            
            # Recompute all coefficients (orthogonal matching pursuit)
            K_array = np.array(selected_K)
            coeffs = self.compute_coefficients_batch(K_array)
            
            # Update residual
            Y_reconstructed = self.reconstruct_batch(K_array, coeffs)
            residual = self.Y - Y_reconstructed
            
            error = np.linalg.norm(residual) / self.Y_norm
            
            if verbose and (len(selected_K) % 10 == 0 or error < threshold):
                kmaxnorm = 0
                kminnorm = np.linalg.norm(selected_K[0])
                weightedsum = 0
                normalsum   = 0
                for ik, k in enumerate(selected_K):
                    knorm = np.linalg.norm(k)
                    kmaxnorm = max(kmaxnorm, knorm)
                    kminnorm = min(kminnorm, knorm)
                    weightedsum += knorm * np.abs(coeffs[ik])**2
                    normalsum   += np.abs(coeffs[ik])**2
                max_coeff_id = np.argmax(np.abs(coeffs)**2)
                kargmax = np.linalg.norm(selected_K[max_coeff_id])

                print(f"Iteration {iter}: Selected {len(selected_K)} vectors, "
                      f"error = {error:.6f}, kminnorm = {kminnorm:.6f}, kmaxnorm = {kmaxnorm:.6f}, kargmax = {kargmax:.6f}, kmean = {weightedsum/normalsum:.6f}")
                kmeans.append(weightedsum/normalsum)
            
            if error < threshold:
                break
        
        return {
            'wave_vectors': np.array(selected_K),
            'coefficients': coeffs,
            'reconstruction_error': error,
            'num_components': len(selected_K),
            'Y_reconstructed': Y_reconstructed,
            'kmeans': kmeans,
        }
    
    def _generate_smart_candidates(self, residual, selected_K, k_max, batch_size):
        """Generate smart candidate wave vectors based on residual."""
        candidates = []
        
        # 1. Perturbations of existing wave vectors
        if selected_K:
            num_perturb = min(batch_size // 3, len(selected_K))
            for k in selected_K[-num_perturb:]:
                perturb = np.random.randn(self.n) * k_max * 0.1
                candidates.append(k + perturb)
        
        # 2. Gradient-based candidates
        # Approximate gradient by sampling
        num_gradient = batch_size // 3
        for _ in range(num_gradient):
            # Random direction
            direction = np.random.randn(self.n)
            direction /= np.linalg.norm(direction)
            
            # Line search along this direction
            alphas = np.linspace(-k_max, k_max, 10)
            correlations = []
            
            for alpha in alphas:
                k = alpha * direction
                basis = self.fourier_basis_batch(k.reshape(1, -1)).flatten()
                correlation = np.abs(np.vdot(basis, residual))
                correlations.append(correlation)
            
            best_alpha = alphas[np.argmax(correlations)]
            candidates.append(best_alpha * direction)
        
        # 3. Random candidates
        num_random = batch_size - len(candidates)
        random_candidates = np.random.randn(num_random, self.n) * k_max / 2
        candidates.extend(list(random_candidates))
        
        return np.array(candidates)
    
    def reconstruct_batch(self, K, coefficients):
        """Efficiently reconstruct Y using wave vectors and coefficients."""
        if len(K) == 0:
            return np.zeros_like(self.Y)
            
        # Compute all basis functions at once
        basis_matrix = self.fourier_basis_batch(K)  # Shape: (m, num_k)
        
        # Reconstruct: Y = basis_matrix @ coefficients
        return basis_matrix @ coefficients
    
    def sparse_fourier_transform(self, method='matching_pursuit', **kwargs):
        """
        Main interface for sparse Fourier transform.
        
        Parameters:
        - method: 'matching_pursuit' or 'random_search'
        - **kwargs: method-specific parameters
        """
        if method == 'matching_pursuit':
            print("match")
            return self.matching_pursuit(**kwargs)
        elif method == 'random_search':
            print("random")
            return self.random_search_wave_vectors(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


def test_high_dimensional():
    """Test with high-dimensional data."""
    np.random.seed(42)
    
    # Generate high-dimensional test data
    #n = 200  # dimensions
    #m = 1000  # samples
    
    print(f"Testing with {n}-dimensional data and {m} samples...")
    
    # Generate sparse ground truth
    # True function has only a few active Fourier modes
    #num_true_modes = 20
    #true_K = np.random.randn(num_true_modes, n) * 2.0
    #true_coeffs = np.random.randn(num_true_modes) + 1j * np.random.randn(num_true_modes)
    #true_coeffs *= np.exp(-np.arange(num_true_modes) * 0.1)  # Decay
    
    # Generate non-uniform sample points
    #X = np.random.randn(n, m) * 0.5
    
    # Compute Y values
    #Y = np.zeros(m, dtype=complex)
    #for k, c in zip(true_K, true_coeffs):
    #    Y += c * np.exp(1j * np.dot(k, X).T)
    
    # Add noise
    #Y += 0.01 * (np.random.randn(m) + 1j * np.random.randn(m))
    #Y = np.real(Y)  # Take real part for simplicity
    
    # Run sparse Fourier transform
    print("\nRunning sparse Fourier transform...")
    start_time = time.time()
    
    nudft = MemoryEfficientNUDFT(X, Y)
    result = nudft.sparse_fourier_transform(
        method='matching_pursuit',
        num_iterations=numiter,
        k_max=kmaxval,
        threshold=0.01,
        batch_size=2*n,
        verbose=True
    )

    kmeans = result['kmeans']
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Found {result['num_components']} wave vectors")
    print(f"Final reconstruction error: {result['reconstruction_error']:.6f}")
    
    f = open("fourierlog.txt", "a")
    f.write(str(bid)+" "+str(modeid)+" "+str(kmaxval)+" "+str(numiter)+" "+str(result['reconstruction_error'])+" "+stringify(kmeans)+"\n")
    f.close()

    # Memory usage estimate
    memory_mb = (X.nbytes + Y.nbytes + 
                result['wave_vectors'].nbytes + 
                result['coefficients'].nbytes) / (1024**2)
    print(f"Approximate memory usage: {memory_mb:.2f} MB")
    
    return nudft, result


if __name__ == "__main__":
    nudft, result = test_high_dimensional()