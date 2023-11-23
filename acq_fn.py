import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#from utils import desc_pp, desc_pp_notest,pre_rem_split
from utils_Kunal import * 
#from io_utils import append_write, out_time, fig_MDS_scatter_std, fig_MDS_scatter_label

def acq_fn(fn_name, i, prediction_idxs, remaining_idxs, prediction_set_size, rnd_size, X_heldout, K_high , gpr, preprocess, out_name, random_seed):
    std_heldout = None
    if  fn_name == "none":
        """
        A. random sampling 
        """
        print("Legacy")

    elif fn_name == "cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min

        """
        C. Clustering without chunk
        """
        K_high = prediction_set_size
        prediction_idxs_bef = prediction_idxs

        #-- without chunk
        prediction_idxs = remaining_idxs

        X_train = X_heldout[prediction_idxs, :]

        #-- Preprocessing                                
        X_train_pp = desc_pp_notest(preprocess, X_train)            

        num_clusters = K_high

        #-- clustering
        start = time.time()
        print("starting clustering \n")
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 24, random_state=random_seed)
        z_km = km.fit(X_train_pp)
        process_time = time.time() - start
        print( process_time)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        print("length of centers " + str(len(centers)) + "\n")
        
        start = time.time()
        print("starting calculat nearest points of centers \n")
        closest, _ = pairwise_distances_argmin_min(centers, X_train_pp)
        print("number of closest points " + str(len(closest)) + "\n")
        process_time = time.time() - start
        print( process_time)
        
        pick_idxs = np.array(prediction_idxs)[closest]
        print("length of pick idxs " + str(len(pick_idxs)) + "\n")

        #-- length of cluster
#        cluster_idxs = np.empty(num_clusters)
#        cluster_len = np.empty(num_clusters)

#        for j in range(num_clusters):
#            cluster_idxs[j] = np.array(np.where(labels == j)).flatten()
#            cluster_len[j] = len(cluster_idxs[j])
            
#        np.save(out_name + "_" + str(i+1) + "_cluster_len", cluster_len )
        
        prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
        remaining_idxs = np.setdiff1d(remaining_idxs, pick_idxs)

        X_train = X_heldout[prediction_idxs, :]
        #-- Preprocessing                                                        
        X_train_pp = desc_pp_notest(preprocess, X_train)

        #-- save the values 
        np.savez(out_name + "_" + str(i+1) + "_idxs.npz", remaining_idxs=remaining_idxs, prediction_idxs=prediction_idxs, pick_idxs = pick_idxs)        
        
    elif  fn_name == "rnd2":
        """
        A .random sampling without chunk (same as none)
        """
        
        #-- Preprocessing / Normalize                                                                                                                                                    
        #-- check mean and std in next dataset
        print("starting prediction \n", flush=True)
        start = time.time()
        y_pred, time_pred = gpr.predict(X_heldout[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        # Get prediction
        pred_heldout = y_pred.mean.detach().numpy()
        print("Unnormalize prediction")
        pred_heldout = gpr.unnormalize(pred_heldout)

        print(pred_heldout[:100], flush=True)
        heldout_pred_all = np.transpose( np.vstack( (X_heldout[:,0].flatten(), pred_heldout.flatten()) ) )
        std_heldout = heldout_pred_all
        print(heldout_pred_all.shape)

        X_add2train, X_heldout_new = train_test_split(X_heldout, train_size = prediction_set_size, random_state=random_seed)        
        
        
        print("IN THE DEPTHS OF ACQ")
        print(prediction_idxs.shape)
        print(remaining_idxs.shape)
        print(prediction_set_size)
            
    elif fn_name == "high2":
        """
        B. High std without chunk
        """
        K_high = prediction_set_size
        prediction_idxs_bef = prediction_idxs

        if len(remaining_idxs) == K_high :
            pick_idxs = remaining_idxs
            remaining_idxs = np.array([])
            prediction_idxs = np.r_[prediction_idxs_bef, pick_idxs]
            
        elif len(remaining_idxs) != K_high :
        
            prediction_idxs = remaining_idxs

            K_high = prediction_set_size # num_cluster
            
            print("starting prediction \n", flush=True)
            start = time.time()
            y_pred, time_pred = gpr.predict(X_heldout[:,1:])
            process_time = time.time() - start

            # Get std
            #print(y_pred.mean)
            std_heldout = y_pred.stddev.detach().numpy()

            print(std_heldout[:10], flush=True)
            print("HEHE", flush=True)

            # Get third with highest std
            # 1. Add NUMPY INDICES as column to array
            std_heldout_tmp = np.concatenate( ( std_heldout.reshape(len(std_heldout), 1), np.arange(len(std_heldout)).reshape( len(std_heldout), 1 ) ), axis=1 )
            print(std_heldout_tmp.shape, flush=True) # Should contain std of all X_heldout
            # Sort by std and choose partition
            std_heldout_tmp = np.flipud(std_heldout_tmp[std_heldout_tmp[:, 0].argsort()])[:prediction_set_size]
            print(std_heldout_tmp, flush=True)

            print(std_heldout_tmp.shape, flush=True)
            pick_idxs_tmp = std_heldout_tmp[:,1].astype("int")
            print(pick_idxs_tmp, flush=True)
            #-- check 
            print( "Max value of std within pick_idxs " + str(np.max(std_heldout[pick_idxs_tmp])) + "\n" , flush=True)
            print( "Min value of std within pick_idxs " + str(np.min(std_heldout[pick_idxs_tmp])) + "\n" , flush=True)

            #--
            X_train = X_heldout[pick_idxs_tmp, :]  # Held-out data with highest std
            print("Is this Held-out data with highest std? (Incl. idx)")
            print(X_train)
            print(X_train.shape)

        X_add2train   = X_train

        idx_heldout_new = np.setdiff1d(np.arange(len(X_train)), pick_idxs_tmp)
        X_heldout_new = X_train[idx_heldout_new]


    elif fn_name == "high_and_cluster":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        D. Combination of B and C
        without chunk,
        1. Choose mols with high std 
        2. Make cluster
        3. Choose mols which is near the center of clusters
        """
        K_pre = rnd_size # highest std # 1/K_pre is the fraction of highest std considered
        K_high = prediction_set_size # num_cluster
        prediction_idxs_bef = prediction_idxs
    
        #-- Preprocessing / Normalize                                                                                                                                                    
        #-- check mean and std in next dataset
        print("starting prediction \n", flush=True)
        start = time.time()
        y_pred, time_pred = gpr.predict(X_heldout[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        # Get std
        #print(y_pred.mean)
        std_heldout = y_pred.stddev.detach().numpy()

        print(std_heldout[:100], flush=True)
        
        #-- unsorted top K idxs  
        #            K = int(len(remaining_idxs)/2.0)
        print("HEHE", flush=True)
        print(K_pre, flush=True)
        print(len(X_heldout[:,0].flatten("C")), flush=True)
        K = int(len(X_heldout[:,0].flatten("C"))/K_pre)
        print(K, flush=True)

        # Get third with highest std
        # 1. Add numpy indices as column to array
        std_heldout_tmp = np.concatenate( ( std_heldout.reshape(len(std_heldout), 1), np.arange(len(std_heldout)).reshape( len(std_heldout), 1 ) ), axis=1 )
        print(std_heldout_tmp.shape, flush=True) # Should contain std of all X_heldout
        # Sort by std and choose partition
        std_heldout_tmp = np.flipud(std_heldout_tmp[std_heldout_tmp[:, 0].argsort()])[:K]
        print(std_heldout_tmp, flush=True)

        print(std_heldout_tmp.shape, flush=True)
        pick_idxs_tmp = std_heldout_tmp[:,1].astype("int")
        print(pick_idxs_tmp, flush=True)
        #-- check 
        print( "Max value of std within pick_idxs " + str(np.max(std_heldout[pick_idxs_tmp])) + "\n" , flush=True)
        print( "Min value of std within pick_idxs " + str(np.min(std_heldout[pick_idxs_tmp])) + "\n" , flush=True)
        
        #--
        X_train = X_heldout[pick_idxs_tmp, :]  # Held-out data with highest std
        print("Is this Held-out data with highest std? (Incl. idx)")
        print(X_train)
        print(X_train.shape) 
        #-- Preprocessing                                       
        #X_train_pp = desc_pp_notest(preprocess, X_train)   # Normalize
        
        #--
        num_clusters = K_high
        
        #-- Clustering
        start = time.time()
        print("starting clustering \n", flush=True)
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 128, random_state=random_seed)
        z_km = km.fit(X_train[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)
        
        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        print("length of centers " + str(len(centers)) + "\n", flush=True)
        
        start = time.time()
        print("starting calculate nearest points of centers \n", flush=True)
        closest, _ = pairwise_distances_argmin_min(centers, X_train[:,1:])
        print("number of closest points " + str(len(closest)) + "\n", flush=True)
        process_time = time.time() - start
        print( process_time, flush=True)
        print("Indices of selected X_heldout (upper third in std, that are close to the centroids")      
        print(closest, flush=True)
        print(closest.shape, flush=True)

        #-- Calculate centers
        pick_idxs = X_train[:,0][closest]
        print("length of pick idxs " + str(len(pick_idxs)) + "\n", flush=True)
        print(pick_idxs, flush=True)
        print(pick_idxs.shape, flush=True)

        X_add2train   = X_train[closest]

        idx_heldout_new = np.setdiff1d(np.arange(len(X_train)), pick_idxs)
        X_heldout_new = X_train[idx_heldout_new]

    elif fn_name == "lowPsat":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        Pick molecules in the specific pSat domain
        Then cluster them respective to their descriptor
        """
        testtrain_idx = prediction_idxs

        #-- Preprocessing / Normalize                                                                                                                                                    
        #-- check mean and std in next dataset
        print("starting prediction \n", flush=True)
        start = time.time()
        y_pred, time_pred = gpr.predict(X_heldout[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        # Get prediction
        pred_heldout = y_pred.mean.detach().numpy()
        print("Unnormalize prediction")
        pred_heldout = gpr.unnormalize(pred_heldout)

        print(pred_heldout[:100], flush=True)

        # Pick predictions in domain
        lowerlim = 1e-30
        upperlim = 3e-10

        # Get third with highest std
        # 1. Add numpy indices as column to array
        pred_heldout_tmp = np.concatenate( ( pred_heldout.reshape(len(pred_heldout), 1), np.arange(len(pred_heldout)).reshape( len(pred_heldout), 1 ) ), axis=1 )
        print(pred_heldout_tmp.shape, flush=True) # Should contain pred of all X_heldout
        # Sort by std and choose partition
        pred_heldout_tmp = pred_heldout_tmp[(pred_heldout_tmp[:,0] > np.log10(lowerlim)) & (pred_heldout_tmp[:,0] < np.log10(upperlim))]
        print(pred_heldout_tmp[:10], flush=True)

        print(pred_heldout_tmp.shape, flush=True)
        pick_idxs_tmp = pred_heldout_tmp[:,1].astype("int")
        print(pick_idxs_tmp, flush=True)
        #-- check 
        print( "Max value of pred within pick_idxs " + str(np.max(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)
        print( "Min value of pred within pick_idxs " + str(np.min(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)

        #--
        X_train = X_heldout[pick_idxs_tmp, :]  # Held-out data with highest std
        print("Is this Held-out data with highest std? (Incl. idx)")
        print(X_train)
        print(X_train.shape)
        #-- Preprocessing                                       
        #X_train_pp = desc_pp_notest(preprocess, X_train)   # Normalize

        #--
        num_clusters = prediction_set_size

        #-- Clustering
        start = time.time()
        print("starting clustering \n", flush=True)
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 128, random_state=random_seed)
        z_km = km.fit(X_train[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        print("length of centers " + str(len(centers)) + "\n", flush=True)

        start = time.time()
        print("starting calculate nearest points of centers \n", flush=True)
        closest, _ = pairwise_distances_argmin_min(centers, X_train[:,1:])
        print("number of closest points " + str(len(closest)) + "\n", flush=True)
        process_time = time.time() - start
        print( process_time, flush=True)
        print("Indices of selected X_heldout ")
        print(closest, flush=True)
        print(closest.shape, flush=True)

        #-- Calculate centers
        pick_idxs = X_train[:,0][closest]
        print("length of pick idxs " + str(len(pick_idxs)) + "\n", flush=True)
        print(pick_idxs, flush=True)
        print(pick_idxs.shape, flush=True)

        X_add2train   = X_train[closest]

        idx_heldout_new = np.setdiff1d(np.arange(len(X_train)), pick_idxs)
        X_heldout_new = X_train[idx_heldout_new]


    elif fn_name == "midPsat":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        Pick molecules in the specific pSat domain
        Then cluster them respective to their descriptor
        """
        testtrain_idx = prediction_idxs

        #-- Preprocessing / Normalize                                                                                                                                                    
        #-- check mean and std in next dataset
        print("starting prediction \n", flush=True)
        start = time.time()
        y_pred, time_pred = gpr.predict(X_heldout[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        # Get prediction
        pred_heldout = y_pred.mean.detach().numpy()
        print("Unnormalize prediction")
        pred_heldout = gpr.unnormalize(pred_heldout)

        print(pred_heldout[:100], flush=True)

        # Pick predictions in domain
        lowerlim = 1.78e-7
        upperlim = 1.12e-5

        # Get third with highest std
        # 1. Add numpy indices as column to array
        pred_heldout_tmp = np.concatenate( ( pred_heldout.reshape(len(pred_heldout), 1), np.arange(len(pred_heldout)).reshape( len(pred_heldout), 1 ) ), axis=1 )
        print(pred_heldout_tmp.shape, flush=True) # Should contain pred of all X_heldout
        # Sort by std and choose partition
        pred_heldout_tmp = pred_heldout_tmp[(pred_heldout_tmp[:,0] > np.log10(lowerlim)) & (pred_heldout_tmp[:,0] < np.log10(upperlim))]
        print(pred_heldout_tmp[:10], flush=True)

        print(pred_heldout_tmp.shape, flush=True)
        pick_idxs_tmp = pred_heldout_tmp[:,1].astype("int")
        print(pick_idxs_tmp, flush=True)
        #-- check 
        print( "Max value of pred within pick_idxs " + str(np.max(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)
        print( "Min value of pred within pick_idxs " + str(np.min(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)

        #--
        X_train = X_heldout[pick_idxs_tmp, :]  # Held-out data with highest std
        print("Is this Held-out data with highest std? (Incl. idx)")
        print(X_train)
        print(X_train.shape)
        #-- Preprocessing                                       
        #X_train_pp = desc_pp_notest(preprocess, X_train)   # Normalize

        #--
        num_clusters = prediction_set_size

        #-- Clustering
        start = time.time()
        print("starting clustering \n", flush=True)
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 128, random_state=random_seed)
        z_km = km.fit(X_train[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        print("length of centers " + str(len(centers)) + "\n", flush=True)

        start = time.time()
        print("starting calculate nearest points of centers \n", flush=True)
        closest, _ = pairwise_distances_argmin_min(centers, X_train[:,1:])
        print("number of closest points " + str(len(closest)) + "\n", flush=True)
        process_time = time.time() - start
        print( process_time, flush=True)
        print("Indices of selected X_heldout ")
        print(closest, flush=True)
        print(closest.shape, flush=True)

        #-- Calculate centers
        pick_idxs = X_train[:,0][closest]
        print("length of pick idxs " + str(len(pick_idxs)) + "\n", flush=True)
        print(pick_idxs, flush=True)
        print(pick_idxs.shape, flush=True)

        X_add2train   = X_train[closest]

        idx_heldout_new = np.setdiff1d(np.arange(len(X_train)), pick_idxs)
        X_heldout_new = X_train[idx_heldout_new]

    elif fn_name == "highPsat":
        from sklearn import cluster
        from sklearn.metrics import pairwise_distances_argmin_min
        """
        Pick molecules in the specific pSat domain
        Then cluster them respective to their descriptor
        """
        testtrain_idx = prediction_idxs

        #-- Preprocessing / Normalize                                                                                                                                                    
        #-- check mean and std in next dataset
        print("starting prediction \n", flush=True)
        start = time.time()
        y_pred, time_pred = gpr.predict(X_heldout[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        # Get prediction
        pred_heldout = y_pred.mean.detach().numpy()
        print("Unnormalize prediction")
        pred_heldout = gpr.unnormalize(pred_heldout)

        print(pred_heldout[:100], flush=True)

        # Pick predictions in domain
        lowerlim = 1.12e-5
        upperlim = 2e4

        # Get third with highest std
        # 1. Add numpy indices as column to array
        pred_heldout_tmp = np.concatenate( ( pred_heldout.reshape(len(pred_heldout), 1), np.arange(len(pred_heldout)).reshape( len(pred_heldout), 1 ) ), axis=1 )
        print(pred_heldout_tmp.shape, flush=True) # Should contain pred of all X_heldout
        # Sort by std and choose partition
        pred_heldout_tmp = pred_heldout_tmp[(pred_heldout_tmp[:,0] > np.log10(lowerlim)) & (pred_heldout_tmp[:,0] < np.log10(upperlim))]
        print(pred_heldout_tmp[:10], flush=True)

        print(pred_heldout_tmp.shape, flush=True)
        pick_idxs_tmp = pred_heldout_tmp[:,1].astype("int")
        print(pick_idxs_tmp, flush=True)
        #-- check 
        print( "Max value of pred within pick_idxs " + str(np.max(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)
        print( "Min value of pred within pick_idxs " + str(np.min(pred_heldout[pick_idxs_tmp])) + "\n" , flush=True)

        #--
        X_train = X_heldout[pick_idxs_tmp, :]  # Held-out data with highest std
        print("Is this Held-out data with highest std? (Incl. idx)")
        print(X_train)
        print(X_train.shape)
        #-- Preprocessing                                       
        #X_train_pp = desc_pp_notest(preprocess, X_train)   # Normalize

        #--
        num_clusters = prediction_set_size

        #-- Clustering
        start = time.time()
        print("starting clustering \n", flush=True)
        km = cluster.KMeans(n_clusters = num_clusters, n_jobs = 128, random_state=random_seed)
        z_km = km.fit(X_train[:,1:])
        process_time = time.time() - start
        print( process_time, flush=True)

        labels = np.array(z_km.labels_)
        centers = np.array(z_km.cluster_centers_)
        print("length of centers " + str(len(centers)) + "\n", flush=True)

        start = time.time()
        print("starting calculate nearest points of centers \n", flush=True)
        closest, _ = pairwise_distances_argmin_min(centers, X_train[:,1:])
        print("number of closest points " + str(len(closest)) + "\n", flush=True)
        process_time = time.time() - start
        print( process_time, flush=True)
        print("Indices of selected X_heldout ")
        print(closest, flush=True)
        print(closest.shape, flush=True)

        #-- Calculate centers
        pick_idxs = X_train[:,0][closest]
        print("length of pick idxs " + str(len(pick_idxs)) + "\n", flush=True)
        print(pick_idxs, flush=True)
        print(pick_idxs.shape, flush=True)

        X_add2train   = X_train[closest]

        idx_heldout_new = np.setdiff1d(np.arange(len(X_train)), pick_idxs)
        X_heldout_new = X_train[idx_heldout_new]

    else:
        print(fn_name + "\n")
        print("You should use defined acquisition function ! \n")
        print("program stopped ! \n")
        sys.exit()

    return X_add2train, X_heldout_new, std_heldout

                    
