#!/home/prateek/workspace_Soumik/soumik-experiment-env/bin/python3
#Status: Running properly but CUDA error for large input size
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn_extra.cluster import KMedoids
#from sklearn.cluster import KMeans
#import kmedoids
from sklearn.cluster import KMeans
import argparse

# Check if CUDA is available for PyTorch and use it for distance calculations
myDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096"
#myDevice = 'cpu'

def compute_distance_gpu(ei, ej, distance_metric):
    """
    Computes the distance between all pairs of embeddings in ei and ej using the specified distance metric,
    utilizing GPU for faster calculations.
    """
    print (distance_metric)
    
    torch.cuda.empty_cache()
    print (f"ei size: {ei.size()}--- ej size:{ej.size()}")
    print (f"ei type: {ei.dtype}---ej type: {ej.dtype}")
    print ("ei tensor: ")
    print (ei)
    print ("ej tensor")
    print (ej)
    #ei, ej = ei.to(myDevice,dtype=torch.float8_e5m2), ej.to(myDevice,dtype=torch.float8_e5m2)  # Move embeddings to GPU
    ei,ej=ei.to(myDevice),ej.to(myDevice)
    dist_matrix = None
    try:
        if distance_metric == 'cosine':
            #dist_matrix = 1 - torch.nn.functional.cosine_similarity(ei.unsqueeze(1), ej.unsqueeze(0), dim=-1)
            cos=torch.nn.CosineSimilarity(dim=-1)
            dist_matrix=1-cos(ei,ej)
        elif distance_metric == 'euclidean':
            dist_matrix = torch.cdist(ei, ej)  # Efficient pairwise Euclidean distance
        elif distance_metric == 'manhattan':
            print ("Manhattan")
            dist_matrix = torch.cdist(ei, ej, p=1)  # Manhattan distance
    # Add more distance metrics as needed

        return dist_matrix
    except Exception as e:
        print ("Exception caught")
        print (e)    
        print ("----------------")
        raise Exception("CUDA Exception thrown by SGR")

#def populate_distance_matrix_parallel(embeddings, distance_metric='cosine', n_jobs=-1):
def populate_distance_matrix_parallel(input_dir,distance_metric='cosine',n_jobs=-1):
    fileCount=0
    for file_name in os.listdir(input_dir):
        fileCount=fileCount+1
    #n = len(embeddings)
    n=fileCount
    print (f"Length of the embeddings: {n}")
    distance_matrix = torch.zeros((n, n), device=myDevice)
    print (distance_matrix)
    #embeddings = [torch.tensor(embed, device=myDevice, dtype=torch.float8_e5m2) if not isinstance(embed, torch.Tensor) else embed.clone().detach().to(myDevice) for embed in embeddings]
    """
    def process_pair(i, j):
        if i < j:  # Only compute for the upper triangle
            try:
                dist_matrix = compute_distance_gpu(embeddings[i], embeddings[j], distance_metric)
                return i, j, dist_matrix.mean().item()  # Use mean distance as representative distance
            except Exception as e:
                print ("Exception caught ar populate")
                print (e)
        
        return None  # Skip lower triangle
    """
    """
    # Run computations in parallel to take advantage of CPU parallelism and GPU acceleration
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_pair)(i, j) for i in tqdm(range(n), desc='Computing distance matrix') for j in range(n)
    )
    """
    
    results=[]
    """
    for i in range(n):
        for j in range(n):
            if i<j:
                dist_matrix = compute_distance_gpu(embeddings[i], embeddings[j], distance_metric)
                results.append((i,j,dist_matrix.mean().item()))
                #return i, j, dist_matrix.mean().item()  # Use mean distance as representative distance
    """
            
    #Need to load two pickeled file at a time and process them for distance calculation
    i=0
    for file_name_outer in os.listdir(input_dir):
        #print (f"File Name: {file_name}")
        j=0
        for file_name_inner in os.listdir(input_dir):
            if i<j:
                print (f"{file_name_outer} and {file_name_inner}")
                if file_name_outer.endswith('.pkl') and file_name_inner.endswith('.pkl'):
                    if file_name_outer != file_name_inner:
                        #Process the files for distance calculation
                        file_path_outer=os.path.join(input_dir,file_name_outer)
                        file_path_inner=os.path.join(input_dir,file_name_inner)
                        with open(file_path_outer,'rb') as pickle_file:
                            obj_pickle_outer=pickle.load(pickle_file)
                        with open(file_path_inner,'rb') as pickle_file:
                            obj_pickle_inner=pickle.load(pickle_file)
                        dist_matrix=compute_distance_gpu(obj_pickle_outer,obj_pickle_inner,distance_metric)    
                        results.append((i,j,dist_matrix.mean().item()))
            else:
                j=j+1
            #----End of inner for loop---
        i=i+1
    #----End of outer for loop---
       

    # Populate the distance matrix
    for result in results:
        if result is not None:  # Only process valid results
            i, j, value = result
            distance_matrix[i, j] = value
            distance_matrix[j, i] = value  # Symmetric matrix
    
    print ("Pupulate distance matrix done ....")
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

#def compute_k_clusters_parallel(embeddings, k_range=range(1, 5), method='kmedoids', distance_metric='cosine', n_jobs=-1):
def compute_k_clusters_parallel(input_dir,embeddings,k_range=range(1,5),method='kmediods',distance_metric='cosine',n_jobs=-1):
    """
    Computes clusters in parallel for different values of k using the specified method and distance metric.
    """
    print ("@compute_k_clusters_parallel")
    fileCount=0
    for file_name in os.listdir(input_dir):
        fileCount=fileCount+1
    #n_samples = len(embeddings)
    n_samples=fileCount
    
    # Filter k_range to avoid exceeding the number of samples
    k_range = [k for k in k_range if k <= n_samples and k > 0]

    if not k_range:  # No valid k values found
        print("No valid k values for clustering. Please check the number of embeddings.")
        return {}

    # Ensure embeddings are PyTorch tensors on the correct device
    embeddings = [torch.tensor(embed, device=myDevice,dtype=torch.float8_e5m2) if not isinstance(embed, torch.Tensor) else embed.clone().detach().to(myDevice) for embed in embeddings]
#dtype=torch.float8_e5m
    
    # Populate distance matrix
    #distance_matrix = populate_distance_matrix_parallel(embeddings, distance_metric, n_jobs=n_jobs)
    distance_matrix=populate_distance_matrix_parallel(input_dir,distance_metric,n_jobs=n_jobs)

    # Print distance matrix
    print("Final Distance Matrix:")
    distance_matrix_cpu = distance_matrix.cpu()  # Move to CPU for printing
    for row in distance_matrix_cpu:
        print(' '.join(f'{val:.3f}' for val in row))
    print()
    """
    # Use parallel processing for clustering
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_k_cluster)(distance_matrix, k, method) for k in tqdm(k_range, desc='Clustering')
    )
    return dict(results)
    """
    #method="kmeans"
    results=[]
    for k in k_range:
        #results=[]
        
        results.append(compute_single_k_cluster(distance_matrix,k,method))
        
        #print (f"Step {k} results:{results}")
    #print (dict(results))
    return dict(results)
    #print (results)
    


def load_pickle_files(input_dir):
    print("@load_pickeled_objects_from_directory...")
    objects_list = []
    for file_name in os.listdir(input_dir):
        print (f"File name : {file_name}")
        if file_name.endswith('.pkl'):
            file_path = os.path.join(input_dir, file_name)
            print (f"File path: {file_path}")
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

def main():
    input_dir="~/workspace_Soumik/multiproperty/output/Pickle_File/6s399/structPkl"
    method='kmedoids'
    #distance_metric='cosine'
    argDistance_Metric="manhattan"
    #Step 1: Load the piclkles
    #embeddings=load_pickle_files("/home/prateek/soumikSep13/6s101/structPkl")
    embeddings=load_pickle_files(input_dir)
    # print (embeddings)
    
    #for emb in embeddings:
      #  print ("---------------")
     #   print (emb.shape)
    #distanceMatrix=populate_distance_matrix_parallel(embeddings)
    distanceMatrix=populate_distance_matrix_parallel(input_dir)
    print ("Output of Populate method")
    print (distanceMatrix)

    #Step 2: 
    argMethod="kmeans"
    print (f"Length of Embedding: {len(embeddings)}") 
    print ("Computing clusters ...")
    print (argDistance_Metric)

    cluster_results = compute_k_clusters_parallel(input_dir,embeddings, method=argMethod, distance_metric=argDistance_Metric)
    #Step 3: Print the results
    if cluster_results:
        print(f"Clusters using {argMethod} with {argDistance_Metric} distance")
        for k, labels in cluster_results.items():
            print(f"for k={k} : {labels}")

    if cluster_results:
        print("\nFormatted Output:")
        formatted_results = format_cluster_assignments(cluster_results)
        for k, clusters in formatted_results.items():
            cluster_str = ' '.join(f"{value}" for key, value in clusters.items())
            print(f"for k={k} : {{{cluster_str}}}")

if __name__=="__main__":
    print ("At my workspace...")
    #main("/home/prateek/soumikSep13/6s101/structPkl", method=args.method, distance_metric=args.distance_metric)
    main()
