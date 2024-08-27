from sklearn.decomposition import PCA
import numpy as np

def perform_pca(data, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on the dataset.

    Parameters:
    data (numpy.ndarray or pandas.DataFrame): The input data, where rows are samples and columns are features.
    n_components (int, float, None or str): Number of components to keep.
        If n_components is not set, all components are kept:
            n_components == min(n_samples, n_features)
        If n_components == 'mle', Minka's MLE is used to guess the dimension.
        If n_components < 1, it represents the percentage of variance explained.

    Returns:
    tuple: A tuple containing the following elements:
        - pca: Fitted PCA object.
        - transformed_data (numpy.ndarray): Data transformed to the new principal components space.
        - explained_variance_ratio (numpy.ndarray): Percentage of variance explained by each of the selected components.
    """

    # Initialize PCA with the number of components
    pca = PCA(n_components=n_components)

    # Fit the PCA model to the data and transform the data
    transformed_data = pca.fit_transform(data)

    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca, transformed_data, explained_variance_ratio


### Example Usage:

# Example data: 10 samples with 5 features
data = np.random.rand(10, 5)

# Perform PCA, reducing to 2 components
pca, transformed_data, explained_variance_ratio = perform_pca(data, n_components=2)

print("Transformed Data:\n", transformed_data)
print("Explained Variance Ratio:\n", explained_variance_ratio)
