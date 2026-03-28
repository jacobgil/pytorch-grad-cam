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
        reshaped_activations = activations.reshape(
            activations.shape[0], -1).transpose()
        reshaped_activations = reshaped_activations - \
            reshaped_activations.mean(axis=0)
        # Apply Kernel PCA
        kpca = KernelPCA(n_components=1, kernel=kernel, gamma=gamma)
        projection = kpca.fit_transform(reshaped_activations)
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


def get_2d_projection_with_sign_correction(activation_batch: np.ndarray) -> np.ndarray:
    """
    Perform SVD on a batch of activation maps, project onto the first
    principal component, and apply sign correction.

    Sign correction addresses the inherent sign ambiguity of SVD:
    decomposing A = U Σ Vᵀ is equivalent to (-U) Σ (-Vᵀ), so the sign
    of the resulting projection is arbitrary. The correction ensures that
    class-discriminative information aligns with the positive direction by
    flipping the map when |min| > |max| (Eq. 13 in the paper).

    Reference:
        Chung, C.-T.; Ying, J.J.-C. Seg-Eigen-CAM. Appl. Sci. 2025,
        15(13), 7562. https://doi.org/10.3390/app15137562

    Args:
        activation_batch: Array of shape (B, C, H, W).

    Returns:
        np.ndarray of shape (B, H, W) with dtype float32.
    """
    activation_batch[np.isnan(activation_batch)] = 0
    projections = []

    for activations in activation_batch:
        reshaped = activations.reshape(activations.shape[0], -1).transpose()
        reshaped = reshaped - reshaped.mean(axis=0)

        _, _, VT = np.linalg.svd(reshaped, full_matrices=True)

        projection = reshaped @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])

        # Sign correction (Eq. 13): ensure salient regions are positive
        if abs(projection.min()) > abs(projection.max()):
            projection = -projection

        projections.append(projection)

    return np.float32(projections)
