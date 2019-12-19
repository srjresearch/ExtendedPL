//header files
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <omp.h> 
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



////////////////////////// EXTENDED Plackett-Luce MODEL \\\\\\\\\\\\\\\\\\\\\\\\\\\\

///////////// Global variables /////////////

//data
int n = 21; //number of rankers/observations
int K = 20; //number of entisties
char datafilename[] = "../Data/f1_data.ssv"; //data file location


//mcmc details -- note the total number of iterations performed is burn_in + (nsample*thin)
int burn_in = 10000; //number of burn-in iterations
int nsample = 10000; //number of desired posterior realisations
int thin = 100; //thin factor between iterations

int no_of_chains = 5; //number of chains within the MC^3 scheme
int num_thread = 5; //number of cores to use for computation. Use omp_get_num_threads() to get number of available cores
//Note: not efficient to consider num_thread > no_of_chains

int fix_seed = 1; //set = 1 for fixed rng seed. Otherwise seed based on clock time.
//Note: even with a fixed seed the output might differ between runs due to the schocastic nature of job allocations to threads. For debugging set both fix_seed = 1 and no_of_thread = 1


///////////// Functions /////////////

int** make_int_mat(int n, int m); //creates a 2D array of integers
double** make_double_mat(int n, int m); //creates a 2D array of doubles
void print_array_int(FILE* file_name, int* array, int size); //prints array of integers to file
void print_array_dbl(FILE* file_name, double* array, int size); //prints array of doubles to file
double loglikeli_oneranking(int** x, double* lambda_dagger, int ranki, int K, int* sigma); //evaluates (log) likelihood of a single observation (ranki)
int sample_bern(double U, double prob); //samples from bernoulli distribution with success probability p
void copy_array(int* a, int* b); //copies integer content of a[] into b[]
void copy_array_dbl(double* a, double* b); //copies double content of a[] into b[]
int sample_cum_probs(double U, double* probs, int size); //returns a sample when a cumulative probability array is passed
double spl_log_prob(int* ranking, double* params, int K); //standard PL prob of complete ranking. Note ranking must be of numbers 0 to (K-1)
void generate_perm_pl(int* sigma, double* rho, int K, gsl_rng* r); //samples a permutation (ranking) from PL distribution with parameters rho
int which_min_dbl(double* b, int size); //returns index of minimum value in array or doubles which has length size
void Order(int* sigma, int* sigma_inv); //generates inverse permutation of sigma, and stores in sigma_inv


int main()
{
	
	///////////// Prior /////////////
    
	double* rho = (double*)malloc(K * sizeof(double)); //sigma ~ PL(rho)
	for(int j = 0; j < K; j++)
	{
		rho[j] = (j+1.0);
	}
	
	double* a_vec_lambda = (double*)malloc(K * sizeof(double)); //lambda ~ Ga(a_lambda[sigma^{-1}[k]],b_lambda)
	for(int k = 0; k < K; k++)
    {
		a_vec_lambda[k] = 1.0;
	}
    double b_lambda = 1;
	
	
	///////////// Tuning parameters for MH steps /////////////
	
	double sd_lambda_rw = 1.0; //standard deviation for LNRW on lamdba (common across all lambdas and all chains)
	
	int num_of_prop = 5; //number of different proposal mechanisms for sigma. Note: proposal mechanisms are in a different order to as presented within the paper.
	double* proposal_probs = (double*)malloc(num_of_prop * sizeof(double));
	proposal_probs[0] = 0.30; //probability of using proposal 1 (random swap)
	proposal_probs[1] = 0.30; //probability of using proposal 2 (poisson swap)
	proposal_probs[2] = 0.30; //probability of using proposal 3 (random insertion)
	proposal_probs[3] = 0.05; //probability of using proposal 4 (reverse proposal)
	proposal_probs[4] = 0.05; //probability of using proposal 5 (prior proposal)

	double no_swaps = 1; //number of swaps performed in proposals 1 - 3

	double tau = 1.0; //poisson rate parameter for proposal 2
	
	
	///////////// Details for parallel tempering /////////////
	
	double* chain_powers = (double*)malloc(no_of_chains * sizeof(double)); //chain i has likelihood^(chain_powers[i])
	chain_powers[0] = 0.5;
	chain_powers[1] = 0.55;
	chain_powers[2] = 0.7;
	chain_powers[3] = 0.85; 
	chain_powers[4] = 1; //chain of interest
	//Note: need chain_powers[0] < chain_powers[1] < ... < chain_powers[no_of_chains-1]
	
	
	///////////// Initalisation \\\\\\\\\\\\\\\\
	
	//initalise RNGs
	omp_set_num_threads(num_thread); //number of cores to use
	gsl_rng *rng[num_thread];
	int seed = time(NULL);
	for(int i = 0; i < num_thread; i++)
	{
		rng[i]=gsl_rng_alloc(gsl_rng_mt19937);
		if(fix_seed) //fixed initalisation
		{
			gsl_rng_set(rng[i],i+10); 
		}else //random initalisation
		{
			gsl_rng_set(rng[i],i*4236 + seed); 
		}
	}
    
    //initalise choice order parameters sigma
    int** sigma = 	make_int_mat(no_of_chains, K);  //2D array to hold the choice order parameter sigma for each chain
    int** sigma_inv = 	make_int_mat(no_of_chains, K); //2D array to hold the inverse (permutation) of sigma for each chain
    for(int c = 0; c < no_of_chains; c++) //initalise from prior \sigma ~ PL(\rho)
    {
		generate_perm_pl(sigma[c], rho, K, rng[0]); //sample from PL distribution
		Order(sigma[c], sigma_inv[c]); //find inverse permutation and store in sigma_inv[c]
	}
	
	//initalise skill parameters lambda
	double** lambda = make_double_mat(no_of_chains, K); //2D array to hold the skill parameters lambda for each chain
    for(int c = 0; c < no_of_chains; c++) //initalise from prior \lambda|\sigma,a ~ Ga(a[sigma^{-1}[k]],1)
    {
		for(int k = 0; k < K; k++)
		{
			lambda[c][k] = gsl_ran_gamma(rng[0],a_vec_lambda[sigma_inv[c][k]],1.0/b_lambda); //sample lambdas from prior
		}
	}
	
    
    ///////////// Useful quantities for MCMC /////////////
    
    int** sigma_prop = 	make_int_mat(no_of_chains, K); //holds the proposed permutation
    int** sigma_prop_inv = 	make_int_mat(no_of_chains, K); //holds the inverse of the proposed permutation
    double* int_cum_probs_K = (double*)malloc(K * sizeof(double)); //
    int_cum_probs_K[0] = 1.0/K;
    for(int i = 1; i < K; i++)
    {
		int_cum_probs_K[i] = int_cum_probs_K[i-1] + 1.0/K;
	}
	double* proposal_cum_probs = (double*)malloc(num_of_prop * sizeof(double)); //
    proposal_cum_probs[0] = proposal_probs[0];
    for(int i = 1; i < num_of_prop; i++)
    {
		proposal_cum_probs[i] = proposal_probs[i] + proposal_cum_probs[i-1];
	}
	double* int_cum_probs_no_chain = (double*)malloc((no_of_chains-1) * sizeof(double)); // sample 1,...,no_chains-1. Then always propose swap with sample + 1
    int_cum_probs_no_chain[0] = 1.0/(no_of_chains-1.0);
    for(int i = 1; i < (no_of_chains-1); i++)
    {
		int_cum_probs_no_chain[i] = int_cum_probs_no_chain[i-1] + 1.0/(no_of_chains-1.0);
	}
	double sum_a_vec_lambda = 0.0;
    for(int k = 0; k < K; k++)
    {
		sum_a_vec_lambda += a_vec_lambda[k];
	}
	
	///////////// Varianbles for monitoring acceptance rates /////////////
	
	int* no_prop_made = (int*)calloc(num_of_prop , sizeof(int));
	int* no_prop_acc = (int*)calloc(num_of_prop , sizeof(int));
	int overall_acc = 0;
    int* no_prop_lambda_acc = (int*)calloc(K , sizeof(int));
	int** between_chain_prop = make_int_mat(no_of_chains, no_of_chains);
	int** between_chain_acc = make_int_mat(no_of_chains, no_of_chains);
	
	
	///////////// Read in data /////////////
	
	int** x = make_int_mat(n,K); //holds data. x[i][j] is ranking i position j.
	FILE *datafile;
	gsl_matrix *data;
	data = gsl_matrix_alloc(n,K);
	datafile=fopen(datafilename,"r");
	gsl_matrix_fscanf(datafile,data);
	fclose(datafile); //close file
	for(int i = 0; i < n; i++ ) 
	{
		for(int j = 0; j < K; j++ )
		{
			x[i][j]= gsl_matrix_get( data,i,j);
		}
	}
	
	
	///////////// Output files /////////////
	
	//check if epl_output directories exist, if not create one
	struct stat st = {0};
	if(stat("epl_outputs", &st) == -1)
	{
		mkdir("epl_outputs", 0777);
		printf("Created new output directory as /epl_outputs did not exist\n");
	}
	FILE* lambda_out = fopen("epl_outputs/lambdaout.ssv","w");
	FILE* likeli_out = fopen("epl_outputs/likeliout.ssv","w");
	FILE* sigma_out = fopen("epl_outputs/sigmaout.ssv","w");
	FILE* acc_probs_out = fopen("epl_outputs/acc_probs.txt","w");
	
	
	///////////// Print curret time and total number of iterations to be performed /////////////
	
	time_t rawtime;
	struct tm * timeinfo;
	printf("Starting MCMC for %d iterations\n",  burn_in + (nsample*thin));
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Current local time and date: %s", asctime(timeinfo));
	
	
	int nit = burn_in + (nsample*thin); //total number of iterations required
	
////START MCMC
for(int itercount = 0; itercount < (nit+1); itercount++)
{
	
	#pragma omp parallel for default(shared) //parallel loop
	for(int c = 0; c < no_of_chains; c++)
	{
	
	//// PROPOSE A NEW SIGMA \\\\
	
	double ran_U;
	ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
	int proposal = sample_cum_probs(ran_U, proposal_cum_probs, num_of_prop); //sample which proposal to use
	no_prop_made[proposal] ++;
	
	copy_array(sigma[c], sigma_prop[c]); //copy entries in sigma into sigma_prop (not pointer)
	
	int pos_1, pos_2; //positions to swap/insert
	//different proposal mechanisms
	if(proposal == 0)//random swap
	{
		for(int s = 0; s < no_swaps; s++)
		{
			ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
			pos_1 = sample_cum_probs(ran_U, int_cum_probs_K, K);	
			ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
			pos_2 = sample_cum_probs(ran_U, int_cum_probs_K, K);
			
			int tmp = sigma_prop[c][pos_1];
			sigma_prop[c][pos_1] = sigma_prop[c][pos_2];
			sigma_prop[c][pos_2] = tmp;
		}
		
	}else if (proposal == 1)//poisson swap
	{
		
		for(int s = 0; s < no_swaps; s++)
		{
			ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
			pos_1 = sample_cum_probs(ran_U, int_cum_probs_K, K);
			int move_val =  gsl_ran_poisson (rng[omp_get_thread_num()], tau);
			
			if(gsl_rng_uniform(rng[omp_get_thread_num()]) < 0.5)
			{
				pos_2 = ((pos_1 + move_val) % K);
			}else
			{
				pos_2 = (pos_1 - move_val);
				while(pos_2 < 0)
				{
					pos_2 += K;
				}
				pos_2 = pos_2 % K;
			}
						
			int tmp = sigma_prop[c][pos_1];
			sigma_prop[c][pos_1] = sigma_prop[c][pos_2];
			sigma_prop[c][pos_2] = tmp;
		}
	}else if(proposal == 2)//random insertion
	{
		for(int s = 0; s < no_swaps; s++)
		{
			ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
			pos_1 = sample_cum_probs(ran_U, int_cum_probs_K, K);	
			ran_U = gsl_rng_uniform(rng[omp_get_thread_num()]);
			pos_2 = sample_cum_probs(ran_U, int_cum_probs_K, K);
			int tmp = sigma_prop[c][pos_1];
			
			if(pos_1 < pos_2)
			{
				for(int i = pos_1; i < pos_2; i++)
				{
					sigma_prop[c][i] = sigma_prop[c][i+1];
				}
			}else if(pos_1 > pos_2)
			{
				for(int i = pos_1; i > pos_2; i--)
				{
					sigma_prop[c][i] = sigma_prop[c][i-1];
				}
			}
			
			sigma_prop[c][pos_2] = tmp;
		}		
	}else if(proposal == 3) //reverse proposal
	{
		for(int i = 0; i < K; i++)
		{
			sigma_prop[c][i] = sigma[c][K-1-i];
		}
	}else if(proposal == 4) //prior proposal
	{
		generate_perm_pl(sigma_prop[c], rho, K, rng[omp_get_thread_num()]);
	}
	
	Order(sigma_prop[c], sigma_prop_inv[c]); //compute inverse of proposed permutation and store in sigma_prop_inv[c]

	
	double ll_cur = 0; //evaluate (log) likelihood under current and proposed permutation
	double ll_prop = 0;
	for(int i = 0; i < n; i++)
	{
		ll_cur += loglikeli_oneranking(x,lambda[c], i, K, sigma[c]);
		ll_prop += loglikeli_oneranking(x,lambda[c], i, K, sigma_prop[c]);
	}
	
	double acc_prob = chain_powers[c] * (ll_prop - ll_cur); //temper likelihood
	acc_prob += spl_log_prob(sigma_prop[c], rho, K); //prior ratio of sigma
	acc_prob -= spl_log_prob(sigma[c], rho, K);
	for(int k = 0; k < K; k++) //prior ratio for lambda|\sigma
	{
		acc_prob += (a_vec_lambda[sigma_prop_inv[c][k]] - a_vec_lambda[sigma_inv[c][k]])*log(lambda[c][k]);
	}
	

	if(log(gsl_rng_uniform(rng[omp_get_thread_num()])) < acc_prob) //accept new sigma
	{
		copy_array(sigma_prop[c], sigma[c]); //change sigma to be sigma_prop
		copy_array(sigma_prop_inv[c], sigma_inv[c]); //change sigma_inv to be sigma_prop_inv
		
		if(c == (no_of_chains-1))
		{
			no_prop_acc[proposal] ++;
		}
	}

	
	///// PROPOSE NEW LAMBDAS \\\\
	
	for(int k = 0; k < K; k++) //set up loop to sample lambda 
	{
		double lambda_prop =  exp(log(lambda[c][k]) + gsl_ran_gaussian(rng[omp_get_thread_num()], sd_lambda_rw)); //LNRW proposal
		
		double ll_cur = 0; //evalute (log) likelihood under current and proposed lamba
		double ll_prop = 0;
		for(int i = 0; i < n; i++)
		{
			ll_cur += loglikeli_oneranking(x,lambda[c], i, K, sigma[c]);
		}
		double tmp_lambda = lambda[c][k];
		lambda[c][k] = lambda_prop;
		for(int i = 0; i < n; i++)
		{
			ll_prop += loglikeli_oneranking(x,lambda[c], i, K, sigma[c]);
		}
		
		double acc_prob = chain_powers[c] * (ll_prop - ll_cur); //temper likelihood
		acc_prob += a_vec_lambda[sigma_inv[c][k]] * (log(lambda_prop) - log(tmp_lambda)); //correct with prior and proposal ratio
		acc_prob += tmp_lambda - lambda_prop; 

		
		if(log(gsl_rng_uniform(rng[omp_get_thread_num()])) < acc_prob) //accept
		{
			if(c == (no_of_chains-1))
			{
				no_prop_lambda_acc[k] ++;
			}
		}else
		{
			lambda[c][k] = tmp_lambda; //place back origional lambda
		}
	}


	///////////// Rescale lambda values /////////////	
	
	double lambda_sum = 0;
	for(int k = 0; k < K; k++) //compute sum of lambda values within each chain
	{
		lambda_sum += lambda[c][k]; 
	}
	double cap_lambda = gsl_ran_gamma(rng[omp_get_thread_num()], sum_a_vec_lambda, 1.0/b_lambda); //draw prior realisation for sum of lambdas
	double ratio = cap_lambda/lambda_sum; //resacling ratio
	for(int k = 0; k < K; k++)
	{
		lambda[c][k] = lambda[c][k]*ratio; //rescale
	}
    
    
	
	}//end parallel loop
	
	
	
	//// MAKE PROPOSAL BETWEEN CHAINS \\\\
	
    double ran_U = gsl_rng_uniform(rng[0]);
	int chain_1 = sample_cum_probs(ran_U, int_cum_probs_no_chain, (no_of_chains-1)); //sample chain at random
	int chain_2 = chain_1 + 1; //propose to swap with next chain
	
	between_chain_prop[chain_1][chain_2] ++;
	
	double ll_chain_1 = 0; //compute (log) likelihood of each chain under current parameterisations
	double ll_chain_2 = 0;
	for(int i = 0; i < n; i++) 
	{
		ll_chain_2 += loglikeli_oneranking(x,lambda[chain_2], i, K, sigma[chain_2]);
		ll_chain_1 += loglikeli_oneranking(x,lambda[chain_1], i, K, sigma[chain_1]);
	}
	
	double acc_prob = (chain_powers[chain_1] * (ll_chain_2 - ll_chain_1)) + (chain_powers[chain_2] * (ll_chain_1 - ll_chain_2)); //acceptance ratio
	
    
	if(log(gsl_rng_uniform(rng[0])) < acc_prob) //accept and swap the state space
	{
		between_chain_acc[chain_1][chain_2] ++;
		
		double tmp_dbl;
		double tmp_int;
		for(int j = 0; j < K; j++)
		{
			tmp_dbl = lambda[chain_1][j];
			lambda[chain_1][j] = lambda[chain_2][j];
			lambda[chain_2][j] = tmp_dbl;
			
			tmp_int = sigma[chain_1][j];
			sigma[chain_1][j] = sigma[chain_2][j];
			sigma[chain_2][j] = tmp_int;
			
			tmp_int = sigma_inv[chain_1][j];
			sigma_inv[chain_1][j] = sigma_inv[chain_2][j];
			sigma_inv[chain_2][j] = tmp_int;
			
		}
				
	}


    
	///////////// Print statements /////////////
	
	if(itercount > burn_in && itercount % thin == 0)  
	{
		double likeli = 0;
		for(int i = 0; i < n; i++)
		{
			likeli += loglikeli_oneranking(x, lambda[no_of_chains-1], i, K, sigma[no_of_chains-1]);
		}
		
		fprintf(likeli_out, "%f\n", likeli);
		print_array_dbl(lambda_out, lambda[no_of_chains-1], K);
		print_array_int(sigma_out, sigma[no_of_chains-1], K);
			
	}
	
	
	///////////// Update user on iterations and time /////////////
	
	if(itercount > 0 && itercount % (thin*1000) == 0) //print iteration and time update to screen
	{
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		printf("Current iteration: %d \t Current local time and date: %s", itercount, asctime(timeinfo));
	}	
	
	
}


//print details on acceptance rates to file
fprintf(acc_probs_out, "Number of proposals made: \n");	
for(int i = 0; i < num_of_prop; i++)
{
	fprintf(acc_probs_out,"%d ", no_prop_made[i]);
}
fprintf(acc_probs_out,"\nNumber of proposals accepted: \n");
for(int i = 0; i < num_of_prop; i++)
{
	fprintf(acc_probs_out,"%d ", no_prop_acc[i]);
}
fprintf(acc_probs_out,"\nWithin move acc rate: \n");
for(int i = 0; i < num_of_prop; i++)
{
	fprintf(acc_probs_out,"%f ", (double) no_prop_acc[i]/no_prop_made[i]);
}
fprintf(acc_probs_out,"\nSigma overall acc rate: \n");
for(int i = 0; i < num_of_prop; i++)
{
	fprintf(acc_probs_out,"%f ", (double) overall_acc/nit);
}
fprintf(acc_probs_out,"\nLambda overall acc rate: \n");
for(int i = 0; i < K; i++)
{
	fprintf(acc_probs_out,"%f ", (double) no_prop_lambda_acc[i]/nit);
}
fprintf(acc_probs_out,"\nBeween chain proposals: \n");
for(int i = 0; i < (no_of_chains-1); i++)
{
	fprintf(acc_probs_out,"%f ", (double) between_chain_prop[i][i+1]);	
}
fprintf(acc_probs_out,"\nBeween chain acceptances: \n");
for(int i = 0; i < (no_of_chains-1); i++)
{
	fprintf(acc_probs_out,"%f ", (double) between_chain_acc[i][i+1]);	
}
fprintf(acc_probs_out,"\nBeween chain probs: \n");
for(int i = 0; i < (no_of_chains-1); i++)
{
	fprintf(acc_probs_out,"%f ", (double) between_chain_acc[i][i+1]/between_chain_prop[i][i+1]);
}




//close output files
fclose(lambda_out);
fclose(likeli_out);
fclose(sigma_out);
fclose(acc_probs_out);
	

return 0;

}


//functions
void print_array_int(FILE* file_name, int* array, int size)
{	
	for(int i = 0; i < (size-1); i++)
	{
		fprintf(file_name, "%d ", array[i]);
	}
	fprintf(file_name, "%d\n", array[size-1]);
} 

void print_array_dbl(FILE* file_name, double* array, int size)
{
	for(int i = 0; i < (size-1); i++)
	{
		fprintf(file_name, "%f ", array[i]);
	}
	fprintf(file_name, "%f\n", array[size-1]);
} 



int** make_int_mat(int n, int m)
{
	int** arr = (int**)calloc(n,sizeof(int*));
	for(int i = 0; i < n; i++)
	{
		arr[i] = (int*)calloc(m,sizeof(int));
	}
	return arr;
}


double** make_double_mat(int n, int m)
{
	double** arr = (double**)calloc(n,sizeof(double*));
	for(int i = 0; i < n; i++)
	{
		arr[i] = (double*)calloc(m,sizeof(double));
	}
	return arr;
}

double loglikeli_oneranking(int** x, double* lambda_dagger, int ranki, int K, int* sigma)
{

	double sum, likeli;
	
	sum = 0;
	for(int j = 0; j < K; j++) //sum all lambdas
	{
		sum += lambda_dagger[x[ranki][sigma[j]]-1];
	}

	likeli = 0;
	for(int j = 0; j < K; j++) //### CHANGE
	{
		likeli += log(lambda_dagger[x[ranki][sigma[j]]-1]) - log(sum);
		
		sum -= lambda_dagger[x[ranki][sigma[j]]-1]; //take off the jth lambda value from the sum...saves looping each time
	}	
	
	return(likeli);
}

double spl_log_prob(int* ranking, double* params, int K)
{
	double sum, prob;
	
	sum = 0;
	for(int j = 0; j < K; j++) //sum all lambdas
	{
		sum += params[ranking[j]];
	}

	prob = 0;
	for(int j = 0; j < K; j++)
	{
		prob += log(params[ranking[j]]) - log(sum);
		
		sum -= params[ranking[j]]; //take off the jth lambda value from the sum...saves looping each time
	}	

	return(prob);
}

int sample_bern(double U, double prob)
{
	if(U < prob)
	{
		return(1);
	}else
	{
		return(0);
	}
}


void copy_array(int* a, int* b)
{
	for(int i = 0; i < K; i++)
	{
		b[i] = a[i];
	}
}

void copy_array_dbl(double* a, double* b)
{
	for(int i = 0; i < K; i++)
	{
		b[i] = a[i];
	}
}




int sample_cum_probs(double U, double* probs, int size) //returns a sample when a cumulative probability array is passed
{
	int sample = -1;
	for(int i = 0; i < size; i++)
	{
		if(U <= probs[i])
		{
			sample = i;
			break;
		}
	}
	return(sample);
}



void generate_perm_pl(int* sigma, double* rho, int K, gsl_rng* r) //samples a permutation (ranking) from PL distribution with parameters rho
{
	double* latent_vars = (double*)malloc(K * sizeof(double));
	double sum_of_latents = 0;
	for(int k = 0; k < K; k++) //sample latent exponential random variables
	{
		latent_vars[k] = gsl_ran_gamma(r, 1.0, 1.0/rho[k]);
		sum_of_latents += latent_vars[k];
	}
	for(int k = 0; k < K; k++) //maximum latent variable now has value 1. Order of smallest to largest is still preserved
	{
		latent_vars[k] /= sum_of_latents;
	}
	for(int k = 0; k < K; k++) //ranking is obtained by ordering latent vars from smallest to largest
	{
		int min_ind = which_min_dbl(latent_vars, K); //find index of smallest latent var
		sigma[k] = min_ind;
		latent_vars[min_ind] = 1.5; //1.5 > 1 (largest possible latent var) so this position will not be selected again
	}
	
}



int which_min_dbl(double* b, int size) //function, returns index of minimum value in array, size is dimension of array.
{
	double min;
	int min_ind;

	min = b[0];
	min_ind = 0;
	for(int i = 1; i < size; i++)
	{
		if(b[i] < min)
		{
			min = b[i];
			min_ind = i;
		}
	}

	return(min_ind);

}

void Order(int* sigma, int* sigma_inv)
{
	int i;
	for(i = 0; i < K; i++)
	{
		sigma_inv[sigma[i]] = i;
	}
}

//eof
