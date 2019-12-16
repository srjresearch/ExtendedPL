# Code

This directory contains C code that can be used to obtain posterior samples via MCMC for both the Extended and Standard Plackett-Luce models. Each code is currently configured to analyse the F1 data.

---

## Extended Plackett-Luce model

Extended Plackett-Luce analyses can be performed using epl_code.c. Posterior samples are obtained via the Metropolis Coupled MCMC scheme outlined in the paper. This algorithm exploits parallel computation using OpenMP, if these facilites are not available to the user then the code can be changed to execute on a single core; further details are given in the configuation section below.

The code is currently configured to analyse the F1 data and can be compiled within a terminal environment using the command

`gcc epl_code.c -lgsl -lgslcblas -fopenmp -lm -O3 -o epl_mcmc_exe` 

and executed using the command

`./epl_mcmc_exe`

**WARNING:** Executing the code above will create the directory /epl_outputs in the current working directory if it does not already exist. Alternatively if the directory exists then any previous output files will be overwritten.

### Configuration

The WAND model contains many parameters and the MCMC code must be configured to run on different datasets. Lines 20 to 30 of mcmc.c should be used to specify the number of rankers/entities, the data file location, and the number of desired MCMC iterations. Lines 55 to 72 allow the user to specify different ranking types (complete/partial/top); additional information on these types of rankings are given within the Data directory. The prior distribution is specified though lines 75 to 107. Additional algorithm options, for example, whether it is desired to use the Weighted or standard Plackett-Luce model, or if a fixed seed is to be used, can be defined using lines 110 to 117. The code is configured to initialise at random draw from the prior distribution, however, this can be changed in lines 121 to 222 if desired. Detailed comments within mcmc.c should help guide the user.

---

## Standard Plackett-Luce model

Standard Plackett-Luce analyses can be performed using spl_code.c. Posterior samples are obtained using the Gibbs sampler of Caron & Doucet (2012) and therefore no tuning is required as all unknown quantities are sampled from their corresponding full conditional distributions.

The code is currently configured to analyse the F1 data and can be compiled within a terminal environment using the command

`gcc spl_code.c -lgsl -lgslcblas -lm -O3 -o spl_mcmc_exe` 

and executed using the command

`./spl_mcmc_exe`

**WARNING:** Executing the code above will create the directory /spl_outputs in the current working directory if it does not already exist. Alternatively if the directory exists then any previous output files will be overwritten.

### Configuration

The Standard Plackett-Luce model requires very little configuration. Lines 22 to 39 of spl_code.c should be used to specify the number of rankers/entities, the data file location, and the number of desired MCMC iterations. An indicator variable of whether or not to use a random seed (based on clock time) can be set on line 31. Similarly line 33 gives the user the option to perform inference for the Weighted Plackett-Luce model as in Johnson et al. (2019); more details can be found in srjresearch/BayesianWAND.
The prior distribution is specified though lines 54 to 66 and can be changed as desired. Note however that this Gibbs sampling algorithm relies on a (conjugate) Gamma prior for the skill parameters.
The code is configured to initialise at random draw from the prior distribution, however, this can be changed in lines 111 to 134 if desired. Detailed comments within spl_code.c should help guide the user.




