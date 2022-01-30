import numpy as np
import time
import pandas as pd
from PIL import Image
import os,sys
import matplotlib.pyplot as plt
from scipy import misc
import platform
import warnings

warnings.filterwarnings('ignore')
def graph_distance_by_iteration(distance_list):
    dist_array = np.array(distance_list)
    total_dists = np.sum(dist_array,axis=1)
    plt.plot(total_dists)

def reshape_plot(compressed_array,original_image):
    original_image_copy = np.copy(original_image)
    iheight,iwidth,ipx = original_image_copy.shape
    im_array = np.reshape(compressed_array,(iheight,iwidth,ipx))
    im_array = im_array.astype(np.int16)
    plt.imshow(im_array)
    return plt.show()

def random_inital_medoids(image_array,k):
    unique_vals,unique_index = np.unique(image_array,axis=0,return_index=True)
    medoid_indexes = np.random.choice(unique_index,size=k,replace=False)
    mean_array = image_array[medoid_indexes,:]
    return mean_array

def manually_set_centroids(image_array,k,centroids_array):
    assert image_array.shape[1] == centroids_array.shape[1]
    assert k == centroids_array.shape[0]
    return centroids_array

def image_to_numpy(image_path,show=True):
    image = Image.open(image_path)
    image.load()
    if show:
        print('------ Initial Image -------')
        plt.imshow(image)
        plt.show()
    else:
        pass
    image_array = np.array(image)
    initial_image_copy = np.copy(image_array)
    height, width, pixel_vals = image_array.shape
    image_array = np.reshape(image_array,(height*width,pixel_vals)).astype(np.int16)
    return image_array, initial_image_copy

def manhatten_distance(image_array,means_array):
    image_array_copy = np.copy(image_array)
    means_array_copy = np.copy(means_array)
    expanded_image_array = np.repeat(image_array_copy[:,:,np.newaxis],means_array_copy.shape[0],axis=2)
    expanded_means = means_array_copy[:,:,np.newaxis].T
    mean_labels = np.argmin(np.sum(np.absolute(expanded_image_array - expanded_means),axis=1),axis=1)
    inter_clust_dists = []
    for i in range(0,means_array_copy.shape[0]):
        inter_cluster_index = np.argwhere(mean_labels == i)[:,0]
        inter_cluster_values = image_array_copy[inter_cluster_index,:]
        current_inter_cluster_distance = np.sum(np.absolute(inter_cluster_values - means_array_copy[i,:]))
        inter_clust_dists.append(current_inter_cluster_distance)
    return mean_labels, inter_clust_dists


def ecludian_distance(image_array,means_array):
    image_array_float = np.copy(image_array).astype(np.float)
    means_array_float = np.copy(means_array).astype(np.float)
    expanded_image_array = np.repeat(image_array_float[:,:,np.newaxis],means_array_float.shape[0],axis=2)
    expanded_means = means_array_float[:,:,np.newaxis].T
    mean_labels = np.argmin(np.sum(np.square(expanded_image_array - expanded_means),axis=1),axis=1)
    inter_clust_dists = []
    for i in range(0,means_array.shape[0]):
        inter_cluster_index = np.argwhere(mean_labels == i)[:,0]
        inter_cluster_values = image_array_float[inter_cluster_index,:]
        current_inter_cluster_distance = np.sum(np.square(inter_cluster_values - means_array_float[i,:]))
        inter_clust_dists.append(current_inter_cluster_distance)

    return mean_labels,inter_clust_dists

def recalculate_manhatten_means(image_array,means_array,labels,down_sample=False,down_percent=.5,text_updated=False):
    new_means = np.copy(means_array)
    cluster_distances = []
    
    for i in np.unique(labels):
        inter_cluster_index = np.argwhere(labels == i)[:,0]
        inter_cluster_values = image_array[inter_cluster_index,:]
        current_inter_cluster_distance = np.sum(np.absolute(inter_cluster_values - new_means[i,:]))
        cluster_distances.append(current_inter_cluster_distance)
        unique_inter_cluster_values = np.unique(inter_cluster_values,axis=0)
        
        if down_sample == True:
            max_idx = unique_inter_cluster_values.shape[0]
            samples = int(max_idx * down_percent)
            down_idx = np.random.choice(range(0,max_idx),size=samples,replace=False)
            ds_values = unique_inter_cluster_values[down_idx,:]
        

            for j in ds_values:
                test_centroid = j
                dist_j = np.sum(np.absolute(inter_cluster_values - test_centroid))
                
                if cluster_distances[i] > dist_j:
                    if text_updated:
                        print('---------------------------------------')
                        print('inter cluster distance reduced for cluster %s' % i)
                        print('previous distance %s' % cluster_distances[i])
                        print('previous centroid %s' % new_means[i])
                        print('new distance equals %s' % dist_j)
                        print('new centroid %s' % test_centroid)
                        print('---------------------------------------')
                    cluster_distances[i] = dist_j
                    new_means[i,:] = test_centroid

        else:

            for j in unique_inter_cluster_values:
                test_centroid = j
                
                dist_j = np.sum(np.absolute(inter_cluster_values - test_centroid))
                
                if cluster_distances[i] > dist_j:
                    if text_updated:
                        print('---------------------------------------')
                        print('inter cluster distance reduced for cluster %s' % i)
                        print('previous distance %s' % cluster_distances[i])
                        print('previous centroid %s' % new_means[i])
                        print('new distance equals %s' % dist_j)
                        print('new centroid %s' % test_centroid)
                        print('---------------------------------------')
                    cluster_distances[i] = dist_j
                    new_means[i,:] = test_centroid

            
    return new_means, cluster_distances

def labels_to_means(initial_image_array,means_array,labels):
    assigned_means = np.copy(initial_image_array)
    increment = 0
    for i in labels:
        assigned_means[increment,:] = means_array[i,:]
        increment += 1
    return assigned_means

def recalculate_eluc_means(image_array,means_array,labels,down_sample=False,down_percent=.5,text_updated=True):
    image_array = np.copy(image_array).astype(float)
    new_means = np.copy(means_array).astype(float)
    cluster_distances = []
    for i in np.unique(labels):
        inter_cluster_index = np.argwhere(labels == i)
        inter_cluster_index = inter_cluster_index[:,0]
        inter_cluster_values = image_array[inter_cluster_index,:]
        current_inter_cluster_distance = np.sum(np.square(inter_cluster_values - new_means[i,:]))
        cluster_distances.append(current_inter_cluster_distance)
        unique_inter_cluster_values = np.unique(inter_cluster_values,axis=0)

        if down_sample == True:
            max_idx = unique_inter_cluster_values.shape[0]
            samples = int(max_idx * down_percent)
            down_idx = np.random.choice(range(0,max_idx),size=samples,replace=False)
            ds_values = unique_inter_cluster_values[down_idx,:]
        

            for j in ds_values:
                test_centroid = j
                dist_j = np.sum(np.square(inter_cluster_values - test_centroid))
                
                if cluster_distances[i] > dist_j:
                    if text_updated:
                        print('---------------------------------------')
                        print('inter cluster distance reduced for cluster %s' % i)
                        print('previous distance %s' % cluster_distances[i])
                        print('previous centroid %s' % new_means[i])
                        print('new distance equals %s' % dist_j)
                        print('new centroid %s' % test_centroid)
                        print('---------------------------------------')
                    cluster_distances[i] = dist_j
                    new_means[i,:] = test_centroid

        else:

            for j in unique_inter_cluster_values:
                test_centroid = j
                
                dist_j = np.sum(np.square(inter_cluster_values - test_centroid))
                
                if cluster_distances[i] > dist_j:
                    if text_updated:
                        print('---------------------------------------')
                        print('inter cluster distance reduced for cluster %s' % i)
                        print('previous distance %s' % cluster_distances[i])
                        print('previous centroid %s' % new_means[i])
                        print('new distance equals %s' % dist_j)
                        print('new centroid %s' % test_centroid)
                        print('---------------------------------------')
                    cluster_distances[i] = dist_j
                    new_means[i,:] = test_centroid

            
    return new_means, cluster_distances

def K_medoids(initial_image_path,k=3,iter_max=20,distance='ecludian',text_updated=True,random_means_init=True,mannual_centroid_array=None,delta_halt=False,delta_min=None,down_sample=False,down_percent=.5,show_images=True,sweep_k=False):
    iteration_counter = 0
    iteration_means_list = []
    iteration_distances_list = []
    iteration_labels_list = []
    compressed_arrays = []
    if sweep_k:
        initial_image_array, original_array = image_to_numpy(initial_image_path,show=False)
    else:
        initial_image_array, original_array = image_to_numpy(initial_image_path,show=True)
    if random_means_init == False:
        random_initial_means= manually_set_centroids(initial_image_array,k,mannual_centroid_array)
        iteration_means_list.append(random_initial_means)

    else:
        random_initial_means = random_inital_medoids(initial_image_array,k=k)
        iteration_means_list.append(random_initial_means)

    if distance == 'manhatten':
        initial_labels,initial_distance_matrix = manhatten_distance(initial_image_array,random_initial_means)
        

    elif distance == 'ecludian':
        initial_labels,initial_distance_matrix = ecludian_distance(initial_image_array,random_initial_means)
        
    else:
        print("we don't support that given distance metric")

    
    iteration_labels_list.append(initial_labels)
    if sweep_k == False:
        print('------- Image Based on Random Centers and Assignment  -------')
    compressed_first_image = labels_to_means(initial_image_array,random_initial_means,initial_labels)
    if sweep_k == False:
        reshape_plot(compressed_first_image,original_array)

    iteration_distances_list.append(initial_distance_matrix)
    for i in range(0,iter_max):
        iteration_counter +=1
        if text_updated:
            print('------- K-Medoids Iteration %s  -------' % i)

        if distance == 'manhatten':
            new_centroids, cluster_distances = recalculate_manhatten_means(initial_image_array,iteration_means_list[i],iteration_labels_list[i],down_sample=down_sample,down_percent=down_percent,text_updated=text_updated)

        else:
            new_centroids, cluster_distances = recalculate_eluc_means(initial_image_array,iteration_means_list[i],iteration_labels_list[i],down_sample=down_sample,down_percent=down_percent,text_updated=text_updated)

        dist_test_point = (sum(iteration_distances_list[i]) - sum(cluster_distances))/sum(iteration_distances_list[i])

        if delta_halt and delta_min > dist_test_point:
            if sweep_k == False:
                print('------The Medoids have reached convergence distance at min delta during iteration %s ------' % i)
            centroids_in_list = len(iteration_means_list) - 1
            labels_in_list = len(iteration_labels_list) -1
            
            compressed_image_array = labels_to_means(initial_image_array,iteration_means_list[centroids_in_list],iteration_labels_list[labels_in_list])
            compressed_arrays.append(compressed_image_array)
            if sweep_k == False:
                reshape_plot(compressed_image_array,original_array)
            break
        else:
            pass

        if np.array_equal(new_centroids,iteration_means_list[i]) :
            if sweep_k == False:
                print('------The Medoids have reached convergence algorithim halted during iteration %s ------' % i)
            centroids_in_list = len(iteration_means_list) - 1
            labels_in_list = len(iteration_labels_list) -1
            
            compressed_image_array = labels_to_means(initial_image_array,iteration_means_list[centroids_in_list],iteration_labels_list[labels_in_list])
            compressed_arrays.append(compressed_image_array)
            if sweep_k == False:
                reshape_plot(compressed_image_array,original_array)
            break
        
        else:
            iteration_means_list.append(new_centroids)
            iteration_distances_list.append(cluster_distances)
            
            if distance == 'manhatten':
                        updated_labels,dmx = manhatten_distance(initial_image_array,iteration_means_list[i+1])
        
            elif distance == 'ecludian':
                    updated_labels,dmx = ecludian_distance(initial_image_array,iteration_means_list[i+1])
    
            else:
                 print("we don't support that given distance metric")

            iteration_labels_list.append(updated_labels)

        centroids_in_list = len(iteration_means_list) - 1
        labels_in_list = len(iteration_labels_list) -1
        if show_images:
            compressed_image_array = labels_to_means(initial_image_array,iteration_means_list[centroids_in_list],iteration_labels_list[labels_in_list])
            compressed_arrays.append(compressed_image_array)
            reshape_plot(compressed_image_array,original_array)
    
    return np.array(iteration_distances_list),np.array(iteration_means_list[centroids_in_list]),np.array(iteration_labels_list[labels_in_list]),iteration_counter

def sweep_k_values(initial_image_path,iter_max=50,start_k=2,end_k=200,distance='ecludian',delta_halt=False,delta_min=None,down_sample=False,down_percent=.5,show_images=False):
    k_distance_list = []
    k_indexes = range(start_k,end_k)
    times_list = []
    number_of_iterations_required = []
    for i in range(start_k,end_k):
        start_time = time.time()
        distance_array, final_centers,final_labels,iterrs = K_medoids(initial_image_path=initial_image_path,k=i,iter_max=iter_max,distance=distance,text_updated=False,random_means_init=True,mannual_centroid_array=None,delta_halt=delta_halt,delta_min=delta_min,down_sample=down_sample,down_percent=down_percent,show_images=show_images,sweep_k=True)
        end_time = time.time()
        times_list.append(end_time-start_time)
        number_of_iterations_required.append(distance_array.shape[0]-1)
        k_distance = np.sum(distance_array[distance_array.shape[0]-1,:])
        k_distance_list.append(k_distance)
    return k_distance_list,times_list,k_indexes,number_of_iterations_required