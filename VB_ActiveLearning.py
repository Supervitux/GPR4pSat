import sys
sys.path.insert(0, "/scratch/project_2003525/beselvit/ActiveAtmos/A_MachineLearning/KUNAL/")
from Gpytorch_classes_g import *
from acq_fn import *
import torch
import gpytorch
import numpy as np 
import re
import joblib as jl
from sklearn.model_selection import train_test_split

def main():
        
        print("Active Learning python script starts..")
        name = sys.argv[1]
        wrkdir = sys.argv[2]

        print ("Load " + wrkdir + name + ".npz")
        try:

                # LOAD ALL THE INDICES AND THE MODEL
                print("Load indices from npz file")
                npzfile = np.load(wrkdir + name + ".npz")
                
        except:
                print("No input arguments were given", flush=True)                
                sys.exit(0) 

        # Prepare dscriptors and labels
        df_path        = "/scratch/project_2003525/beselvit/ActiveAtmos/A_MachineLearning/ACTIVE/ALL_LABELLED/DATA/DATAPOOL_FULL.dump"
        idx_train      = npzfile['train_idxs']
        idx_test       = npzfile['test_idxs']
        idx_heldout    = npzfile['heldout_idxs']
        it             = npzfile['it']
        train_set_size = len(idx_train)
        test_set_size  = len(idx_test)

        X_train, X_test, X_heldout = dataframe_prep(df_path, idx_train, idx_test, idx_heldout)
 
        print("Load model ...")
        gp = torch.load(wrkdir + name + '_MODEL.pt', map_location=torch.device('cpu')) 
        
        print(X_train.shape)
        print(X_test.shape)
        print(X_heldout.shape)
        ###### ACTIVE LEARNING #############

        # fn_name, i (some iteration),
        acq_strat = "rnd2"  #"high2"#"high_and_cluster" #"rnd2" #"high2" 
        iter_VB = 0     # Kann man entfernen, wird erst relevant wenn ich loop
        K_high  = 10    # Wieviele cluster, not used in ACQ-D 
        prediction_idxs = np.r_[idx_train, idx_test] # Indices of already used molecules
        remaining_idxs  = idx_heldout # Indices of UNlabeled molecules
        prediction_set_size = 500#int(len(idx_train)/it) # Batchsize + Buffer
        print("Gonna choose " + str(prediction_set_size) + " data points")
        rnd_size = 3.0  # Not quite sure what that is! Is read in from input file in the original code
                        # Apparently it defines how which part of data we pick according to std in ACQ-D
                        # which then is handed to clustering (2.0 -> SIZE(HELD-OUT) / 2.0; half of it)
        X_add2train, X_heldout_new, heldout_pred_all = acq_fn(acq_strat, iter_VB, 
                                            prediction_idxs, remaining_idxs,
                                            prediction_set_size, rnd_size,
                                            X_heldout, K_high , gp, "none", "Dummy.txt", 42)

        # Combine X_add2train and X_train
        X_train       = np.concatenate((X_train, X_add2train), axis=0)
        X_heldout     = X_heldout_new
        it            = it + 1

        # Save all the info!
        # Get name
        print("Save all the indices and the model that will be handed to ActiveLearning")
        print("This was " + name)
        # Indices for Training, Test & Heldout
        np.savez(wrkdir + name + "_Post.npz", train_idxs=X_train[:,0].flatten("C"), test_idxs=X_test[:,0].flatten("C"), heldout_idxs=X_heldout[:,0].flatten("C"), it=it) #, std_heldout=std_heldout)
        np.savetxt(wrkdir + name + "_AllHeldPred.txt", heldout_pred_all)

        # Save idx for Merlin!
        #print("Saving indices for Merlin to start from")
        #np.savez("/scratch/project_2003525/beselvit/ActiveAtmos/Indices/" + name + "_" + str(np.datetime64('today')) + ".npz", prediction_idxs=X_add2train[:,0].flatten("C"))
        np.savetxt(wrkdir + "ADD_" + name + ".txt", X_add2train[:,0].flatten("C"), fmt="%i")
       

        # A few sanity tests
        print(X_train.shape)
        #print(X_train)
        #print(prediction_idxs.shape)
###############################################################################


def create_heldout(idx_test, idx_train):
        
        # This functions takes the testset and trainings set as input and sets everything else to be the heldout set
        print("Load full datapool for heldout creation...", flush=True)
        df_all = jl.load(df_path)
        idx_used = np.r_(idx_test, idx_train)
        df_heldout = df_all[~idx_used]
        X_heldout = np.stack(df_labeled['TopFP'].values)
        idx_df_heldout = df_heldout.index

        return X_heldout, idx_df_heldout

def dataframe_prep(df_path, idx_train, idx_test, idx_heldout):
        # This functions load the test, training and heldout set from a previous model training 
        # it is NOT identical to the function the VB_GPytorch.py

        print("Load full datapool...", flush=True)
        df_all     = jl.load(df_path)
        df_all     = df_all.loc[df_all["NumOfN"] < 3]
        df_all_idx = df_all.index.values
        #df_labeled = df_all.dropna(subset = ["pSat_mbar"])

        df_train   = df_all.loc[idx_train]
        df_test    = df_all.loc[idx_test]
        df_heldout = df_all.loc[idx_heldout]

        print("Create arrays for descriptors and labels...", flush=True)
        X_train = np.stack(df_train['TopFP'].values)
        X_train = np.concatenate((idx_train.reshape([len(idx_train), 1]), X_train), axis=1)

        X_test = np.stack(df_test['TopFP'].values)
        X_test = np.concatenate((idx_test.reshape([len(idx_test), 1]), X_test), axis=1)

        X_heldout = np.stack(df_heldout['TopFP'].values)
        X_heldout = np.concatenate((idx_heldout.reshape([len(idx_heldout), 1]), X_heldout), axis=1)
        
        return X_train, X_test, X_heldout


if __name__ == "__main__":
        main()


# NOTES:
# - Have this code write prints directly to output files ! Makes debugging easier
