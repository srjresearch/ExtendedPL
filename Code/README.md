# Code

This directory contains C code that can be used to obtain posterior samples via MCMC for both the Extended and Standard Plackett-Luce models. Each code is currently configured to analyse the F1 data.

---

## Extended Plackett-Luce model

Extended Plackett-Luce analyses can be performed using epl_code.c. Posterior samples are obtained via the Metropolis Coupled MCMC scheme outlined in the paper. This algorithm exploits parallel computation using OpenMP; further details are given in the configuration section below.

The code is currently configured to analyse the F1 data and can be compiled within a terminal environment using the command

`gcc epl_code.c -lgsl -lgslcblas -fopenmp -lm -O3 -o epl_mcmc_exe` 

and executed using the command

`./epl_mcmc_exe`

**WARNING:** Executing the code above will create the directory /epl_outputs in the current working directory if it does not already exist. Alternatively if the directory exists then any previous output files will be overwritten.

### Configuration

The Metropolis Coupled MCMC sampling scheme is governed by several tuning parameters that must be configured to analyse different datasets. The bullet points below highlight the (beginning of) sections of code that may require user input. Detailed comments within epl_code.c should also help guide the user.

* Line 20 - used to specify the number of rankers/entities, the data file location, the number of desired MCMC iterations and also the number of chains and CPU threads to be used within the computation. An indicator variable of whether or not to use a random seed (based on clock time) is also provided.

* Line 62 - used to define the prior distribution for the unknown choice order and skill parameters. Recall that the entities are required to be labelled in order of preference (from 1 to K) when using a non-uniform prior distribution. Note that the prior distribution is common across all chains, that is, only the likelihood component is tempered. If specifying a multi-modal prior distribution (i.e. some a\_k values are equal) then the indication on line 88 should be set to 1. Further, in this case, you are also required to alter the function "sample\_x\_hat" on line 772 so that this function returns (at random) one of the modes of the prior predictive distribution.

* Line 92 - used to tune the Metropolis-Hastings proposals for both the choice order and skill parameters.

* Line 115 - used to specify the (inverse) temperatures of each chain within the sampling scheme. The temperatures should be increasing; with the final chain having temperature 1. 

The code is configured to initialise at random draw from the prior distribution although this can be changed in lines 144 onward if desired.

### Outputs

As mentioned above all outputs will be written to /epl_outputs in the current working directory. A summary of the files is given below.

* acc_probs.txt - provides details of the acceptance rates for all parameter proposals and also the between-chain proposals. The user should tune the proposal mechanisms to achieve near optimal acceptance rates as discussed within the paper.

* lambdaout.ssv - posterior samples of the skill parameters (after thin), row i column k gives sample i of skill parameter lambda_k.

* sigmaout.ssv - posterior samples of the choice order parameter (after thin). 

* likeliout.ssv - values of the (log) observed data likelihood evaluated for each posterior sample.

* postout.ssv - values of the (log) posterior (i.e. the chain with temperature = 1) evaluated for each posterior sample.

* targetout.ssv - values of the (log) target distribution (the joint distribution of all chains) evaluated for each posterior sample.

---

## Standard Plackett-Luce model

**Update (18/06/20):** Fixed an issue in calculating the observed data likelihood at each output iteration. Parameter inference was not affected by this issue and so posterior samples obtained before this time remain valid. If desired values of the log observed data likelihood can be re-computed (offline) using previously obtained posterior samples of the skill parameters. Apologies for any inconvenience caused.

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

### Outputs

As mentioned above all outputs will be written to /spl_outputs in the current working directory. A summary of the files is given below.

* lambdaout.ssv - posterior samples of the skill parameters (after thin), row i column k gives sample i of skill parameter lambda_k.

* likeliout.ssv - values of the (log) observed data likelihood evaluated for each posterior sample.

Note: when using the Weighted Plackett-Luce model both ranker_weights.ssv and pout.ssv and will also be provided. These files contain the posterior samples of the binary ranker weights along with the corresponding probabilities from which the ranker weights were drawn, respectively. Further details can be found within Johnson et al. (2019).


