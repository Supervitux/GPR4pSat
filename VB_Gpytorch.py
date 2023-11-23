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
import copy

def main():


        run_name = sys.argv[1]
        wrkdir   = sys.argv[2]
        it       = int(sys.argv[3])

        # READ INPUT FILE
        print("Read input file...")
        descriptors_all, X_train_latest, y_train_latest,       \
        X_test, y_test,                                        \
        idx_file,                                              \
        batch_size, test_size,                                 \
        init_const, init_length, init_noise,                   \
        consta_V, length_scalea_V, constb_V, length_scaleb_V,  \
        num_restarts, rnd_seed, num_gpus                       \
        = readInputFile(wrkdir + "input_tmp.dat")
       
        #  Input files grabs X, y of last batch
        #  Now merge it with previous batch(es)
        print("This is an iteration ", it) 

        if it > 1:

                npzfile = np.load(wrkdir + run_name + "_" + str(it-1) + ".npz")
               
                print("Load previous trainingset")
                desc_all, X_train_previous, y_train_previous, X_test, y_test = dataframe_prep("/scratch/project_2003525/beselvit/ActiveAtmos/A_MachineLearning/ACTIVE/ALL_LABELLED/DATA/DATAPOOL_FULL.dump", npzfile['train_idxs'], npzfile['test_idxs'], npzfile['train_idxs'].shape[0], test_size, 1e100, 1e-100)
                # COMMENTED JUST FOR THE ACTIVE COMPARISON WITH RANDOM LEARNING
                X_train          = np.concatenate((X_train_previous, X_train_latest), axis = 0)
                y_train          = np.concatenate((y_train_previous, y_train_latest), axis = 0)

        else:
                X_train = X_train_latest
                y_train = y_train_latest 

        input_const = copy.deepcopy(init_const)
        input_length= copy.deepcopy(init_length)
        # C0
        # IS GIVEN INITIAL TRAININGS 
        # CHOOSE TEST SET
        # The test set should stay the same for all following iterations, this part will possibly be skipped later
        print("Trainingsetsize is ", y_train[:,0].flatten("C").shape)
        all_idx  = descriptors_all[:,0].flatten("C")
        heldout_idx = np.setdiff1d(all_idx, np.r_[X_train[:,0].flatten("C"), X_test[:,0].flatten("C")])
        #print(descriptors_all[:10])
        #print(descriptors_all.shape)
        #print(descriptors_chosen[:10])
        #print(y_train[1990:]) 
        #print(y_test[:10, -10:]) 
        print(heldout_idx.shape)
        #print(X_train.shape)
        #print(np.r_[X_train[:,0], X_test[:,0]])
 
        # Save them all to one .npz file (Chosen, Test, Heldout)

        # SET UP MODEL
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-4))
        gp = GPytorchGPModel(X_train[:,1:], y_train[:,1:].flatten("C"), likelihood)
        gp.set_params(consta_V, length_scalea_V, constb_V, length_scaleb_V)
        gp.set_params_nopriors(init_const, init_length, init_noise)
        pre_const  = copy.deepcopy(init_const)
        pre_length = copy.deepcopy(init_length) 
        pre_noise  = copy.deepcopy(init_noise) 
        trch_optimizer = torch.optim.Adam #SGD #LBFGS
        # FIT MODEL
        n_err = 0

        #### THIS IS WHERE THE ACTIVE LEARNING LOOP WILL START ONE DAY #######
        # Probably everything before this will be just saved and then loaded in at this point.
        # 
        run_no = 0
        while True:
                try:
                        print("FIRST TRAIN")
                        losses, noises, raw_noise, time_fit = gp.fit(X_train[:,1:], y_train[:,1:].flatten("C"), X_test[:,1:], y_test[:,1:].flatten("C"), torch_optimizer=trch_optimizer, lr=0.1, max_epochs=51) 
                        break
                except BaseException as err: #RuntimeError:
                        if (n_err == 0) or (n_err > 8):
                                print(err)
                                print("Error", flush=True)
                                init_const = gp.model.covar_module.outputscale_prior.sample((1,))
                                init_length = gp.model.covar_module.base_kernel.lengthscale_prior.sample((1,))
                                gp.model.covar_module.outputscale = init_const
                                gp.model.covar_module.base_kernel.lengthscale = init_length
                                gp.model.likelihood.raw_noise = torch.tensor(0.0)
                                n_err = n_err + 1
                        else:
                                init_const  = 10**(n_err-1)
                                init_length = 10**(0.5*n_err- -1)
                                gp.model.covar_module.outputscale = init_const
                                gp.model.covar_module.base_kernel.lengthscale = init_length
                                gp.model.likelihood.raw_noise = torch.tensor(0.0)
                                n_err = n_err + 1
 
        # Save the model if you want
        #jl.dump(gp, "GP_" + str(train_size) + ".dump")
        #return # just to stop the function

        # PREDICT
        y_pred, time_pred = gp.predict(X_test[:,1:])
        # Calculate MAE
        val_mae = gp.mae_loss(y_test[:,1:].flatten("C"), y_pred)
        
        # PRINT OUTPUT
        model_list = []
        timing_fit = []
        timing_pred = []
        consts = []
        lengths = []
        pre_consts  = []
        pre_lengths = []
        pre_noises  = []
        maes = []
        #train_maes = []
        loss_values = []
        noise_values = []


        params = gp.get_params()
        const_after = params["constant_value"]
        length_after = params["length_scale"]

        gp_save = copy.deepcopy(gp)
        model_list.append(gp_save)
        maes.append(val_mae)
        #train_maes.append(train_error)
        consts.append(const_after)
        lengths.append(length_after)
        pre_noises.append(pre_noise)
        pre_consts.append(pre_const)
        pre_lengths.append(pre_length)
        loss_values.append(losses[-1])
        noise_values.append(noises[-1])
        timing_fit.append(time_fit)
        timing_pred.append(time_pred)

        # RESTARTS
        print("RESTARTS\n", flush=True)
        for i in range(num_restarts):
                run_no = run_no + 1
                init_const = gp.model.covar_module.outputscale_prior.sample((1,))
                init_length = gp.model.covar_module.base_kernel.lengthscale_prior.sample((1,))
                pre_const  = copy.deepcopy(np.array(init_const.cpu())[0])
                pre_length = copy.deepcopy(np.array(init_length.cpu())[0])
                gp.model.covar_module.outputscale = init_const
                gp.model.covar_module.base_kernel.lengthscale = init_length

                gp.model.likelihood.raw_noise = torch.tensor(0.0)
                gp.model.likelihood.noise = init_noise
                gp.model.likelihood.raw_noise = torch.tensor(0.0)
                #gp.set_params(consta_V, length_scalea_V, constb_V, length_scaleb_V)    
                while True:
                        try:
                                re_loss, re_noise, raw_noise, time_fit = gp.fit(X_train[:,1:], y_train[:,1:].flatten("C"), X_test[:,1:], y_test[:,1:].flatten("C"), torch_optimizer=trch_optimizer, lr=0.1, max_epochs=51)               
                                break
                        except RuntimeError:
                                print("Error", flush=True)
                                init_const = gp.model.covar_module.outputscale_prior.sample((1,))
                                init_length = gp.model.covar_module.base_kernel.lengthscale_prior.sample((1,))
                                gp.model.covar_module.outputscale = init_const
                                gp.model.covar_module.base_kernel.lengthscale = init_length
                                gp.model.likelihood.raw_noise = torch.tensor(0.0)

                
                # PREDICT
                y_pred, time_pred = gp.predict(X_test[:,1:])
                
                # Calculate MAE
                val_mae = gp.mae_loss(y_test[:,1:].flatten("C"), y_pred)

                gp_save = copy.deepcopy(gp)
                model_list.append(gp_save)
                loss_values.append(re_loss[-1])
                noise_values.append(re_noise[-1])
                #train_maes.append(re_train_error)
                params = gp.get_params()
                print(params, flush=True)
                const_after = params["constant_value"]
                length_after = params["length_scale"]
                pre_consts.append(pre_const)
                pre_lengths.append(pre_length)
                length_after = params["length_scale"]
                maes.append(val_mae)
                consts.append(const_after)
                lengths.append(length_after)
                timing_fit.append(time_fit)
                timing_pred.append(time_pred)

        print("", flush=True)
        print("Constant before: " + str(pre_consts), flush=True)
        print("Constant after: " + str(consts), flush=True)
        print("Length scale before: " + str(lengths), flush=True)
        print("Length scale after: " + str(pre_lengths), flush=True)
        print("Noise values before: " + str(pre_noises), flush=True)
        print("Noise values after: " + str(noise_values), flush=True)
        print("Validation mae: " + str(maes), flush=True)
        #print("Train mae: " + str(train_maes), flush=True)
        print("Final loss: " + str(loss_values), flush=True)
        print("Fitting time: " + str(timing_fit), flush=True)
        print("Prediction time: " + str(timing_pred), flush=True)
        ###### Which run did best? ############
        run2save = np.argmin(maes)

        print("Best Validation mae " + str(maes[run2save]))
        print("Saving best model from run " + str(run2save))

        ###### Save Stuff that will be handed over to active learning #########

        # Get name
        print("Save all the indices and the model that will be handed to ActiveLearning")
        path_name = sys.argv[2]
        file_name = run_name + "_" + str(it) 
        print("This was " + path_name + " with " + file_name)
        # Indices for Training, Test & Heldout
        np.savez(path_name + file_name + ".npz", train_idxs=X_train[:,0].flatten("C"), test_idxs=X_test[:,0].flatten("C"), heldout_idxs=heldout_idx, it=it)

        torch.save(model_list[run2save], path_name + file_name + "_MODEL.pt")

        exit()
        ###### ACTIVE LEARNING #############
        ###### THIS IS OUTSOURCED TO A DIFFERENT SCRIPT
        # fn_name, i (some iteration), 
        iter_VB = 0     # Kann man entfernen, wird erst relevant wenn ich loop
        K_high  = 10    # Wieviele cluster
        prediction_idxs = idx_df_train   # Indices of already calculated molecules
        remaining_idxs  = idx_df_remaining # Indices of UNlabeled molecules
        prediction_set_size = 2000 # How many samples do we want to calculate next
        rnd_size = 2.0  # Not quite sure what that is! Is read in from input file in the original code
                        # Apparently it defines how which part of data we pick according to std in ACQ-D
                        # which then is handed to clustering (2.0 -> SIZE(HELD-OUT) / 2.0; half of it)
        prediction_idxs, remaining_idxs, X_train_pp = acq_fn("rnd2", iter_VB, 
                                                                        prediction_idxs, remaining_idxs,
                                                                        prediction_set_size, rnd_size,
                                                                        descriptors_all, labels,
                                                                        K_high , gp, "none", "Dummy.txt", 42)

        # A few sanity tests
        print(prediction_idxs.shape)
        print(remaining_idxs.shape)
        print(X_train_pp.shape)
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

def readInputFile(path2file):

        # Reads all important information from an input file
        file_inp = open(path2file, "r")        
        for l in file_inp:
                if re.search("df_path", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        df_path = temp[0]
                elif re.search("latest_idx_path", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        idx_path = temp[0]
                        idx_train = np.loadtxt(idx_path).astype("int")
                        print(l)
                        print(len(idx_train))
#                elif re.search("train_idx_path", l):
#                        temp = re.findall("^\w+\s+(.+)", l)
#                        idx_path = temp[0]
#                        idx_train = np.loadtxt(idx_path).astype("int")
                elif re.search("test_idx_path", l):        
                        temp = re.findall("^\w+\s+(.+)", l)
                        test_idx_path = temp[0]
                        idx_test = np.loadtxt(test_idx_path).astype("int")
                elif re.search("test_set_size", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        test_set_size = int(temp[0])
#                elif re.search("train_set_size", l):
#                        temp = re.findall("^\w+\s+(.+)", l)
#                        train_set_size = int(temp[0])
                elif re.search("batch_size", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        train_set_size = int(temp[0])
                elif re.search("upper_lim", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        upper_lim = float(temp[0])
                elif re.search("lower_lim", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        lower_lim = float(temp[0])
                elif re.search("init_const", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        init_const = float(temp[0])
                elif re.search("init_length", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        init_length = float(temp[0])
                elif re.search("init_noise", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        init_noise = float(temp[0])
                elif re.search("consta_V", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        consta_V = float(temp[0])
                elif re.search("length_scalea_V", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        length_scalea_V = float(temp[0])
                elif re.search("constb_V", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        constb_V = float(temp[0])
                elif re.search("length_scaleb_V", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        length_scaleb_V = float(temp[0])
                elif re.search("num_restarts", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        num_restarts = int(temp[0])
                elif re.search("rnd_seed", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        rnd_seed = int(temp[0])
                elif re.search("num_gpus", l):
                        temp = re.findall("^\w+\s+(.+)", l)
                        num_gpus = int(temp[0])

        # Could also load full df and apply restrictions on pSatfloatnges e.g.
        desc_all, X_train, y_train, X_test, y_test = dataframe_prep(df_path, idx_train, idx_test, idx_train.shape[0], test_set_size, upper_lim, lower_lim)

        return desc_all, X_train, y_train, X_test, y_test, idx_path, train_set_size, test_set_size, init_const, init_length, init_noise, consta_V, length_scalea_V, constb_V, length_scaleb_V, num_restarts, rnd_seed, num_gpus

def dataframe_prep(df_path, idx_train, idx_test, trains_set_size, test_set_size, upper_lim, lower_lim):
        # This function loads the full datapool, restricts it (Only N<3), than selects the df for 
        # trainings/test idx given, kicks out outliers and returns X and y for both.

        print("Load full datapool...", flush=True)
        df_all     = jl.load(df_path)
        df_all     = df_all.loc[df_all["NumOfN"] < 3]
        df_all_idx = df_all.index.values
        #df_labeled = df_all.dropna(subset = ["pSat_mbar"])

        
        
        df_chosen = df_all.loc[idx_train].dropna(subset = ["pSat_mbar"])
        df_test   = df_all.loc[idx_test].dropna(subset = ["pSat_mbar"])
        #df_unlabeled = df_all.drop(df_labeled.index)
        #df_unlabeled_idx = df_unlabeled.index.values

        # Remove highvolatile/exploded molecules
        print("Upper limit is " + str(upper_lim) + " and lower limit is " + str(lower_lim) + " in mbar", flush=True) 
        #df_labeled = df_labeled[df_labeled["pSat_mbar"] < upper_lim]
        #df_labeled = df_labeled[df_labeled["pSat_mbar"] > lower_lim ]
        #df_labeled_idx = df_labeled.index.values
        
        df_chosen = df_chosen[df_chosen["pSat_mbar"] < upper_lim]
        df_chosen = df_chosen[df_chosen["pSat_mbar"] > lower_lim]
        df_test   = df_test[df_test["pSat_mbar"] < upper_lim]
        df_test   = df_test[df_test["pSat_mbar"] > lower_lim ]

        # Choose train_set_size many molecules (we have more for error headspace)
        if len(idx_train) < trains_set_size:
                print("We have to little molecules to choose from after NaN and Outlier removal!")
                print("I exit")
                sys.exit()
        else:
                df_chosen = df_chosen.sample(trains_set_size, random_state = 42)
        
        # Choose test_set_size many molecules (we have more for error headspace)
        if len(idx_test) < test_set_size:
                print("We have to little molecules to choose from after NaN and Outlier removal!")
                print("I exit")
                sys.exit()
        else:
                df_test = df_test.sample(test_set_size, random_state = 42)

        df_chosen_idx = df_chosen.index.values
        df_test_idx = df_test.index.values

        print("Create arrays for descriptors and labels...", flush=True)
        X_all = np.stack(df_all['TopFP'].values)
        X_all = np.concatenate((df_all_idx.reshape([len(df_all_idx), 1]), X_all), axis=1)

        X = np.stack(df_chosen['TopFP'].values)
        X = np.concatenate((df_chosen_idx.reshape([len(df_chosen_idx), 1]), X), axis=1)
        y = np.log10(df_chosen['pSat_mbar'].values)
        y = np.concatenate((df_chosen_idx.reshape([len(df_chosen_idx), 1]), y.reshape([len(y), 1])), axis=1)

        X_test = np.stack(df_test['TopFP'].values)
        X_test = np.concatenate((df_test_idx.reshape([len(df_test_idx), 1]), X_test), axis=1)
        y_test = np.log10(df_test['pSat_mbar'].values)
        y_test = np.concatenate((df_test_idx.reshape([len(df_test_idx), 1]), y_test.reshape([len(y_test), 1])), axis=1)
        

        return X_all, X, y, X_test, y_test


if __name__ == "__main__":
        main()


# NOTES:
# - Have this code write prints directly to output files ! Makes debugging easier
