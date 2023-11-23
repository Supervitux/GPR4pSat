# GRP4pSat
This repository contains code for the training of GPR models on molecules represented with the topological fingerprint and a vapor pressure label.

The core of the GPR model is contained in the "Gpytorch_classes_g.py" file, these can be applied to any problem. All other codes are most likely not readily useable for any user, but need to be adjusted. If you intend to use these codes please contact the rep-admin for help on which bits of the code are relevant for you.

## Explanation

- input_template.dat: This contains the paths to the pandas dataframe stored in a joblib.dump file, as well as, files with the indices of the training and testset. Further this files contains hyperparameters for the model.

- Gpytorch_classes_g.py: This code contains the GPR model setups.

- VB_Gpytorch.py: Contains the GPR training iterations.

- VB_ActiveLearning.py: This code is used to predict on a held-out set, select and cluster a subset, to obtain a Active Learning batch. It calls "acq_fn.py" which contains the acquisition functions.
