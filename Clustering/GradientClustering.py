import numpy as np
import copy
import matplotlib.pyplot as plt

"""
Much of this code is Based on the following paper:
Visually Mining Through Cluster Hierarchies 
Brecheisen S, Kriegel HP, KrogerP, Pfeifle M. (2004)
Proceedings of the 2004 SIAM International Conference on Data Mining
"""

#calculate the Inflection Index as described in Definition 4, Brecheisen et al. (2004)
def InflectionIndex(reach_list, i, w):
    x_r = reach_list[i - 1] #previous
    y_r = reach_list[i] #current
    z_r = reach_list[i + 1] #next
    
    prev_vect = np.array([w, y_r - x_r])
    next_vect = np.array([w, z_r - y_r])
    prev_abs = np.sqrt(prev_vect[0]**2 + prev_vect[1]**2)
    next_abs = np.sqrt(next_vect[0]**2 + next_vect[1]**2)

    return (-w**2 + (x_r - y_r)*(z_r - y_r))/((prev_abs)*(next_abs))
    
#calculate the Gradient Determinant as described in Definition 6, Brecheisen et al. (2004)
def GradientDeterminant(reach_list, i, w):
    x_r = reach_list[i - 1] #previous
    y_r = reach_list[i] #current
    z_r = reach_list[i + 1] #next
    
    return w*(y_r - x_r) - w*(z_r - y_r)
    
#Based on PseudoCode from Figure 9, Brecheisen et al. (2004) 
def GradientCluster(reach_list, min_points, max_points, t, w):
    #reach_list from OPTICS
    #min_points needed to form a cluster
    #max_points needed to form a cluster
    #t minimum inflection point 
    #w defines the width between points in reachability plot 
    
    t = np.cos(np.radians(t))
    N = len(reach_list)
    
    endpoint = N - 1
    
    start_pts = [] #tracks where a cluster begins
    set_of_clusters = [] #clusters found so far
    curr_cluster = [] #tracks points in current cluster
    
    start_pts.append(0) #0 here as first point in reach list is infinite
    
    for i in range(1, N-1): #check all points
        if InflectionIndex(reach_list, i, w) > t: #check if inflection point
            if GradientDeterminant(reach_list, i, w) > 0: #check if starting point of a cluster
         
                #check if current cluster size is big enough to form a cluster
                if len(curr_cluster) >= min_points:
                    set_of_clusters.append(curr_cluster)
                curr_cluster = [] 
                
                #check if the last starting point has a smaller reachability than the current
                #if so, we remove the last starting point
                if start_pts:
                    if reach_list[start_pts[-1]] <= reach_list[i]:
                        start_pts.pop()

                #while the last point in start points has smaller reachability that current,
                #keep removing points and add them to the cluster
                if start_pts:
                    while reach_list[start_pts[-1]] < reach_list[i]:
                        
                        tmp_cluster = range(start_pts[-1], endpoint)
                        if len(tmp_cluster) >= min_points:
                            set_of_clusters.append(tmp_cluster)

                        start_pts.pop()
                    
                    #if it is not empty, we add another cluster to the end 
                    tmp_cluster = range(start_pts[-1], endpoint)
                    if len(tmp_cluster) >= min_points:
                        set_of_clusters.append(range(start_pts[-1], endpoint))
                
                
                #check if the current point is a starting point, if so store it                
                if reach_list[i+1] < reach_list[i]: 
                    start_pts.append(i)
                
            else: 
                #if current point is endpoint, all points from last start to here are a cluster
                if reach_list[i+1] > reach_list[i]:
                    endpoint = i+1
                    curr_cluster = range(start_pts[-1], endpoint)
    
    #add clusters to the end if any start points remain 
    while start_pts:
        curr_cluster = range(start_pts[-1], N)
        if (reach_list[start_pts[-1]] > reach_list[-1]) and (len(curr_cluster) >= min_points):
            set_of_clusters.append(curr_cluster)
        start_pts.pop()
      
    #remove clusters that are too large  
    Final_Clusters = []
    for cluster in set_of_clusters:
        if len(cluster) < max_points and len(cluster) > 0:
            Final_Clusters.append(cluster)
      
    return Final_Clusters

#merge clusters that share above a set ratio of points    
def Merge_Clusters(clusters, threshold):
    clusters = list(reversed(sorted(clusters, key=len)))
    previous_clusters = clusters
    merged_clusters = []
    
    #end when no change between amount of clusters
    while len(previous_clusters) != len(merged_clusters):
        clusters = list(reversed(sorted(clusters, key=len)))
        skip_list = []
        previous_clusters = copy.deepcopy(merged_clusters)
        merged_clusters = []
        
        for i, cluster1 in enumerate(clusters):
            if i in skip_list:
                continue
            
            found_similar = False
            for j, cluster2 in enumerate(clusters):
                if j <= i:
                    continue
                if j in skip_list:
                    continue
                
                #get number of common points
                intersect_points = len(set(cluster1).intersection(cluster2))
                
                if len(cluster1) > len(cluster2):
                    max_size = len(cluster1)
                else:
                    max_size = len(cluster2)

                #if ratio of common points is large enough
                if intersect_points >= threshold*max_size:
                    #merge two clusters
                    new_cluster = sorted(list(set(set(cluster1) | set(cluster2))))
                    new_cluster = range(min(new_cluster), max(new_cluster)+1) #nicer format

                    
                    merged_clusters.append(new_cluster)
                    
                    #this cluster needs to be skipped on this iteration
                    skip_list.append(j)
                    found_similar = True 
                    break
    
            #if the current cluster was not merged, add it to list
            if not found_similar:
                merged_clusters.append(cluster1)
            
        clusters = copy.deepcopy(merged_clusters)
    
    #sort before returning 
    clusters = list(reversed(sorted(clusters, key=len)))
    return clusters
 
#returns the labels of the clusters in same order as original data
def GetLabels(ordering, clusters):
    labels = np.ones(len(ordering))*-1 #any unclustered point will be -1 
    for i, cluster in enumerate(clusters):
        labels[ordering[cluster].astype(int)] = i
    return labels     
 
#plot reachability with clusters
def PlotReach(X, reach_list, labels, clusters, plot_clusters = True):
    Num_Clusters = len(clusters) + 1

    fig = plt.figure(figsize = (10,10))
    
    plt.rc('axes', labelsize = 18)
    plt.rc('xtick', labelsize = 14)
    plt.rc('ytick', labelsize = 14)
    
    cm = plt.get_cmap('jet')
    
    ax1 = plt.subplot(211)
    ax1.set_prop_cycle(color = [cm(1.*i/Num_Clusters) for i in range(Num_Clusters)])
    ax1.plot(range(len(reach_list)),reach_list)
    
    for i, cluster in enumerate(clusters):
        vertical_reach = max(reach_list[cluster][1], reach_list[cluster][-1])
        vertical_values = np.zeros(len(cluster)) + vertical_reach
        
        ax1.plot(cluster, vertical_values, zorder = i)
        
    ax1.set_xlim((0, len(reach_list)-1))
    ax1.set_ylim((0, np.max(reach_list[reach_list < np.inf])*1.1))
    
    ax1.set_xlabel('Point indices')
    ax1.set_ylabel('Reachability distance')
    
    if plot_clusters == True: 
        ax2 = plt.subplot(212)
        ax2.scatter(X[:,0], X[:,1], c = cm((labels+1)/Num_Clusters))
        
        #ensure the non labeled points are black
        nonlabel = np.where(labels == -1)[0][0]
        ax2.scatter(X[nonlabel][0], X[nonlabel][1], c = 'black')
        
        ax2.set_xlabel(r'$C_{4,2}$')
        ax2.set_ylabel(r'$C_{6,3}$')
        
    fig.tight_layout()
    plt.show()
