import numpy as np

def compute_pca(X, k):
    # Mean Face
    mean_face = np.mean(X, axis=1, keepdims=True)

    # Mean Zero Data
    A = X - mean_face

    # Surrogate Covariance
    C = np.dot(A.T, A)

    # Eigen Decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top k eigenvectors
    eigenvectors = eigenvectors[:, :k]

    # Eigenfaces
    eigenfaces = np.dot(A, eigenvectors)
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    return mean_face, eigenfaces
