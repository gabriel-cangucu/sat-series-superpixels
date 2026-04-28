import sys
import numpy as np
from collections import OrderedDict

sys.setrecursionlimit(50000)


def FuSC(s1, s2, img, min_size):
    """
    Method used to merge two different superpixel segmentations in O(n).

    Args:
        s1 and s2: a 2D array mapping the superpixel segmentations. Each pixel in the represented
            in the segmentation array is set with the number of the respective segment
        img: the segmented image 
        min_size: the minimum size threshold for the merged segmentation
    """
    
    assert s1.shape == s2.shape
    counter = -1
    ret = np.zeros(s1.shape, dtype=int)
    final_labels = {}
    
    # merges two different segmentations. setting sequential labels to each intersection between s1 and s2
    for i in range(0, s1.shape[0]):    # O(n) - assuming constant time for dict operations
        for j in range(0, s1.shape[1]):
            label1 = s1[i,j]
            label2 = s2[i,j]
            if label1 not in final_labels:
                final_labels[label1]={}
            if label2 not in final_labels[label1]:
                final_labels[label1][label2] = counter
                counter-=1
            ret[i,j]=final_labels[label1][label2]
    
    ret = -1 * ret
    counter = -1 * counter
    existing_areas={}
    
    # ensure connectivity for each segment. O(n)
    for i in range(0, ret.shape[0]):      
        for j in range(0, ret.shape[1]):   
            label = ret[i,j]
            if label>0:
                if label in existing_areas:
                    ret[ret==label] = counter  
                    # all operations are O(n) in the worst case when all labels must be replaced. This case is actually unfeasible.
                    label = counter
                    counter + 1
                existing_areas[label]=True    
                ret = track_continuos(ret, i, j, label) 
                # all operations are O(9n) in the worst case when all labels must be replaced. This case is actually unfeasible.
    
    ret = -1 * ret
    neighbors = {}
    # get the neighborhood for each segment
    for i in range(0, ret.shape[0]):         # O(n)
        for j in range(0, ret.shape[1]):
            label1 = ret[i,j]
            if label1 not in neighbors:
                neighbors[label1]={}
            
            for k in range(i-1,i+2):         # cte
                for h in range(j-1,j+2):
                    if (k==h and k==0) or (k<0 or h<0 or k>=ret.shape[0] or h>=ret.shape[1]):
                        continue
                    
                    label2 = ret[k,h]
                    if label1 != label2:
                        if label2 not in neighbors:
                            neighbors[label2]={}
                        neighbors[label1][label2]=True
                        neighbors[label2][label1]=True
            
    return merge_superpixels(ret, neighbors, img, min_size)


def merge_superpixels(sps, neighbors, img, min_size):
    """
    Main procedure that merges neighboring areas if the minimum pixel count is not respected.
    
    The complexity for the procedure is 3O(n)+ O(n * cte * minimum size^2) + 2O(n) = O(n * minimum size^2).
    As some assumed constants depend on the minimum size, we can say that the procedure is pseudo-polynomial.

    Args:
        sps: the 2D mapping superpixel segmentation
        neighbors: a list of each segment and its neighbors
        img: the segmented image
        min_size: the minimum size threshold for the merged segmentation
    """
    sps_sizes={}
    
    img = np.array(img, dtype = float)
    
    sps_uniques = np.unique(sps, return_counts = True) # O(n)
    
    sps_processed = {}
    flatten_superpixels = {}
    
    # pixel count
    for i in range(0,len(sps_uniques[0])):   # O(n)
        sps_sizes[sps_uniques[0][i]] = sps_uniques[1][i]
    
    # populate a dictionary with image pixels for each segment
    for i in range(0, sps.shape[0]):         # O(n)
        for j in range(0, sps.shape[1]):
            label = sps[i,j]
            if label not in flatten_superpixels:
                flatten_superpixels[label] = []
            flatten_superpixels[label].append(img[i,j])

    for key in flatten_superpixels:
        flatten_superpixels[key] = np.array(flatten_superpixels[key])
        
    sps_mapping = OrderedDict()
    
    # for each segment with less pixels of minimum pixel count
    # compare to all neighbors and merge with closest one
    for key in flatten_superpixels:       
        # O(n) as superpixel segmentation is an over segmentation of the image, the expected number of segments is n/cte implying in O(n/cte) = O(n) executions of the for loop
        if key in sps_mapping:
            continue
        while sps_sizes[key] < min_size:    

            min_dist = 99999999999999999  
            final_smaller_key = -1
            final_bigger_key = -1
            
            # closest neighbor search
            for n_key in neighbors[key]:  
            # worst case scenario is n/2 iterations - O(n)
            # assuming that superpixel segmentations produces segments approximately with the same pixel count. And assuming the maximum number of possible neighbors for each segment, the number of iterations are 2 x minimum size + 6. We can consider as O(cte*min_size) = O (cte)
            
                if n_key in sps_mapping:  
                    continue
                smaller_sps_label = n_key
                bigger_sps_label = key
                if flatten_superpixels[key].shape[0] < flatten_superpixels[n_key].shape[0]:
                    smaller_sps_label = key
                    bigger_sps_label = n_key
                    
                if final_smaller_key < 0:
                    final_smaller_key = smaller_sps_label
                    final_bigger_key = bigger_sps_label

                x = flatten_superpixels[smaller_sps_label]
                data = flatten_superpixels[bigger_sps_label]
                
                # computes the distance between the 2 segments
                dist = mahalanobis(x = np.mean(x,axis = 0), data = data) 
                # the complexity of mahalanobis is the greatest between O((d**4)*((log d)**2)) and O(n*(d**2)), for n the number of elements in the biggest segment and d the number of features.
                # in our particular case, we have few features and the probable number of elements in the segment is cte*minimum size. The final complexity for our case is the greatest between O((cte**4)*((log cte)**2)) and O((cte*minimum size)*(cte**2)) = O(minimum size)

                if min_dist > dist:
                    min_dist = dist
                    final_smaller_key = smaller_sps_label
                    final_bigger_key = bigger_sps_label
            
            if final_smaller_key == -1:
                break
            
            # create the merging mapping. In the end, the mapping is executed to produces the final segmentation. O(cte)
            sps_mapping[final_smaller_key] = final_bigger_key
            # compute the size of the merged segment. O(cte)
            sps_sizes[final_bigger_key] = sps_sizes[final_smaller_key] + sps_sizes[final_bigger_key]
            
            for n_key in neighbors[final_smaller_key]:            
                # as discussed before, the probable case is O(2*min_size) = O(cte)
                if final_smaller_key in neighbors[n_key]:    
                    del neighbors[n_key][final_smaller_key]
                neighbors[final_bigger_key][n_key]=True
            
            if final_smaller_key in neighbors[final_bigger_key]:
                del neighbors[final_bigger_key][final_smaller_key]
            if final_bigger_key in neighbors[final_bigger_key]:
                del neighbors[final_bigger_key][final_bigger_key]

            key = final_bigger_key

    sps = merge_mapped(sps, sps_mapping) #O(number of pixels)
    sps = relabel(sps)                   #O(number of pixels)
            
    return sps


def merge_mapped(sps, sps_mapping):
    """
    Auxiliary function to execute the relabel according to the intersection between the segmentations in O(number of pixels)

    Args:
        sps: the 2D mapping superpixel segmentation
        sps_mapping: a list maping merged superpixels
    """
    for i in range(0, sps.shape[0]):         
        for j in range(0, sps.shape[1]):
            label = sps[i,j]
            while label in sps_mapping:
                sps[i,j] = sps_mapping[label]
                label=sps[i,j]
    return sps


def relabel(sps):
    """
    Ensure that numbered lables are between 1 and n

    Args:
        sps: the 2D mapping superpixel segmentation
    """
    counter = -1
    for i in range(0, sps.shape[0]):         
        for j in range(0, sps.shape[1]):
            label = sps[i,j]
            if label<0:
                continue
            sps[sps==label] = counter
            counter -= 1
    return sps*-1


def track_continuos(input_array, i, j, label, rec=False):
    """
    Recursive procedure that selects a continuous area and relabel it
    """
    if input_array[i, j] != label:
        return input_array
    
    input_array[i, j] = input_array[i, j]*-1
    for k in range(i-1,i+2):
        for h in range(j-1,j+2):
            if (k==h and k==0) or (k<0 or h<0 or k>=input_array.shape[0] or h>=input_array.shape[1]):
                continue
            input_array = track_continuos(input_array, k, h, label, rec=True)
    return input_array


def mahalanobis(x=None, data=None, cov=None, eps=1e-6):
    # O((n**4)*((log n)**2)) or O(N*(n**2))
    """
    Compute the Mahalanobis Distance between each row of x and the data

    Args:
        x    : vector or matrix of data with, say, p columns.
        data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
        cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    # 1. Remover colunas com variância zero
    var = np.var(data, axis=0)
    mask = var > 1e-12
    data = data[:, mask]
    x = x[mask]

    # 2. Calcular média
    x_minus_mu = x - np.mean(data, axis=0)

    # 3. Covariância
    if cov is None:
        cov = np.cov(data.T)

    # --- CORREÇÃO CRÍTICA ---
    # Garantir que cov seja 2D
    cov = np.atleast_2d(cov)

    # Se for 1x1, vira matriz 1x1
    if cov.ndim == 1:
        cov = np.array([[cov[0]]])

    # 4. Regularização
    cov = cov + eps * np.eye(cov.shape[0])

    # 5. Pseudo-inversa estável
    inv_covmat = np.linalg.pinv(cov)

    # 6. Distância
    left = np.dot(x_minus_mu, inv_covmat)
    m = np.dot(left, x_minus_mu.T)

    if isinstance(m, np.float64):
        return m
    return m.diagonal()