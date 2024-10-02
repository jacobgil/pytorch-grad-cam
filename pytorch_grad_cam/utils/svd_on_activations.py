import numpy as np
from sklearn.decomposition import KernelPCA


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(
            activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)



def get_2d_projection_kernel(activation_batch, kernel='sigmoid', gamma=None):
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []
    for activations in activation_batch:
        reshaped_activations = activations.reshape(activations.shape[0], -1).transpose()
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        # Apply Kernel PCA
        kpca = KernelPCA(n_components=1, kernel=kernel, gamma=gamma)
        projection = kpca.fit_transform(reshaped_activations)
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)
