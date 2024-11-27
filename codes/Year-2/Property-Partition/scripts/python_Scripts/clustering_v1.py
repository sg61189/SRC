import torch
import numpy as np
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed
import os
import pickle
from tqdm import tqdm
import argparse
import warnings
from sklearn.cluster import KMeans

# Suppress specific UserWarnings from K-Medoids about empty clusters
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn_extra.cluster._k_medoids")


# Check if CUDA is available for PyTorch and use it for distance calculations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_distance_gpu(ei, ej, distance_metric):
    """
    Computes the distance between all pairs of embeddings in ei and ej using the specified distance metric,
    utilizing GPU for faster calculations.
    """
    ei, ej = ei.to(device), ej.to(device)  # Move embeddings to GPU
    dist_matrix = None

    if distance_metric == 'cosine':
        dist_matrix = 1 - torch.nn.functional.cosine_similarity(ei.unsqueeze(1), ej.unsqueeze(0), dim=-1)
    elif distance_metric == 'euclidean':
        dist_matrix = torch.cdist(ei, ej)  # Efficient pairwise Euclidean distance
    elif distance_metric == 'manhattan':
        dist_matrix = torch.cdist(ei, ej, p=1)  # Manhattan distance
    # Add more distance metrics as needed

    return dist_matrix

def populate_distance_matrix_parallel(embeddings, distance_metric='cosine', n_jobs=-1):
    """
    Populates a distance matrix using the distance between pairs of embedding sets in parallel.
    Optimized for GPU processing, only computes the upper triangle of the matrix.
    """
    n = len(embeddings)
    distance_matrix = torch.zeros((n, n), device=device)

    def process_pair(i, j):
        if i < j:  # Only compute for the upper triangle
            dist_matrix = compute_distance_gpu(embeddings[i], embeddings[j], distance_metric)
            return i, j, dist_matrix.mean().item()  # Use mean distance as representative distance
        return None  # Skip lower triangle

    # Run computations in parallel to take advantage of CPU parallelism and GPU acceleration
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, j) for i in tqdm(range(n), desc='Computing distance matrix') for j in range(n)
    )

    # Populate the distance matrix
    for result in results:
        if result is not None:  # Only process valid results
            i, j, value = result
            distance_matrix[i, j] = value
            distance_matrix[j, i] = value  # Symmetric matrix

    return distance_matrix

def compute_single_k_cluster(distance_matrix, k, method='kmedoids'):
    """
    Computes cluster labels for a single value of k using the specified method (k-medoids or k-means).
    """
    distance_matrix_np = distance_matrix.cpu().numpy()  # Move matrix to CPU for sklearn clustering

    if method == 'kmedoids':
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', random_state=42)
        try:
            cluster_labels = kmedoids.fit_predict(distance_matrix_np)
        except ValueError as e:
            print(f"Error with K-Medoids for k={k}: {e}")
            return k, []  # Return empty labels if there's an error
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(distance_matrix_np)
    return k, cluster_labels

def compute_k_clusters_parallel(embeddings, k_range=range(1, 5), method='kmedoids', distance_metric='cosine', n_jobs=-1):
    """
    Computes clusters in parallel for different values of k using the specified method and distance metric.
    """
    print ("@compute_k_clusters_parallel")
    n_samples = len(embeddings)
    
    # Filter k_range to avoid exceeding the number of samples
    k_range = [k for k in k_range if k <= n_samples and k > 0]

    if not k_range:  # No valid k values found
        print("No valid k values for clustering. Please check the number of embeddings.")
        return {}

    # Ensure embeddings are PyTorch tensors on the correct device
    embeddings = [torch.tensor(embed, device=device) if not isinstance(embed, torch.Tensor) else embed.clone().detach().to(device) for embed in embeddings]

    # Populate distance matrix
    distance_matrix = populate_distance_matrix_parallel(embeddings, distance_metric, n_jobs=n_jobs)

    # Print distance matrix
    print("Distance Matrix:")
    distance_matrix_cpu = distance_matrix.cpu()  # Move to CPU for printing
    for row in distance_matrix_cpu:
        print(' '.join(f'{val:.3f}' for val in row))
    print()

    # Use parallel processing for clustering
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_k_cluster)(distance_matrix, k, method) for k in tqdm(k_range, desc='Clustering')
    )
    return dict(results)

def load_pickled_objects_from_directory(input_dir):
    """
    Reads all .pkl files in the specified directory, unpacks the objects/embeddings,
    and returns a list of the unpacked objects.
    """
    print("@load_pickeled_objects_from_directory...")
    objects_list = []

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(input_dir, file_name)
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
                objects_list.append(obj)

    return objects_list

def format_cluster_assignments(cluster_results):
    formatted_results = {}
    for k, labels in cluster_results.items():
        clusters_dict = {}
        for sample_index, cluster_label in enumerate(labels):
            if cluster_label not in clusters_dict:
                clusters_dict[cluster_label] = []
            clusters_dict[cluster_label].append(sample_index)
        
        formatted_results[k] = {key: value for key, value in clusters_dict.items()}
    return formatted_results

def main(input_dir, method='kmedoids', distance_metric='cosine'):
    #Step 1: Load the pickles hence tensors
    print (f"At main....")
    embeddings = load_pickled_objects_from_directory(input_dir) 

    #Step 2: 
    cluster_results = compute_k_clusters_parallel(embeddings, method=method, distance_metric=distance_metric)

    #Step 3: Print the results
    if cluster_results:
        print(f"Clusters using {method} with {distance_metric} distance")
        for k, labels in cluster_results.items():
            print(f"for k={k} : {labels}")

    if cluster_results:
        print("\nFormatted Output:")
        formatted_results = format_cluster_assignments(cluster_results)
        for k, clusters in formatted_results.items():
            cluster_str = ' '.join(f"{value}" for key, value in clusters.items())
            print(f"for k={k} : {{{cluster_str}}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster embeddings from pickled files.')
    parser.add_argument('--input_dir', type=str, help='Directory containing pickled embedding files.')
    parser.add_argument('--method', type=str, choices=['kmeans', 'kmedoids'], default='kmedoids', help='Clustering method to use.')
    parser.add_argument('--distance_metric', type=str, choices=['euclidean', 'manhattan', 'cosine'], default='cosine', help='Distance metric to use.')

    args = parser.parse_args()
    print ("At my workspace...")
    main(args.input_dir, method=args.method, distance_metric=args.distance_metric)
