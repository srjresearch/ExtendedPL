//header files
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>



////////////////////////// STANDARD PL MODEL \\\\\\\\\\\\\\\\\\\\\\\\\\\\


///////////// Global variables /////////////

//data
int n = 21; //number of rankers/observations
int K = 20; //number of entisties
char datafilename[] = "../Data/f1_data.ssv"; //data file location

//mcmc details -- note the total number of iterations performed is burn_in + (nsample*thin)
int burn_in = 10000; //number of burn-in iterations
int nsample = 10000; //number of desired posterior realisations
int thin = 10; //thin factor between iterations

int fix_seed = 0; //set = 1 for fixed rng seed. Otherwise seed is based on clock time.

int weighted_pl = 0; //set = 0 for Standard PL model; Set = 1 if you want Weighted PL model (Johnson et al. 19)



///////////// Functions /////////////

int** make_int_mat(int n, int m); //creates a 2D array of integers
double** make_double_mat(int n, int m); //creates a 2D array of doubles
double loglikeli_one_rank_pl(int** x, double* lambda_dagger, double** z, int* ni_vec, int i, int* K_vec, int* r_vec); //evalutes complete data likelihood of ranking i
double observed_data_loglikeli_one_rank_pl(int** x, double* lambda_dagger, int* ni_vec, int i, int* K_vec, int* r_vec); //evaluates observed data likelihood of ranking i
void print_array_int(FILE* file_name, int* array, int size); //prints array of integers to file
void print_array_dbl(FILE* file_name, double* array, int size); //prints array of doubles to file
int sample_bern(double U, double prob); //samples from bernoulli distribution with success probability p




int main()
{	
	///////////// Prior /////////////
    
    double* a_vec_lambda = (double*)malloc(K * sizeof(double)); //lambda_k ~ Ga(a_vec_lambda[k],b_lambda)
    for(int k = 0; k < K; k++)
    {
		a_vec_lambda[k] = 1.0;
	}
    double b_lambda = 1;
	
	//Prior for ranker weights in the weighted PL model. Ignore for SPL.
	double* p_vec = (double*)malloc(n * sizeof(double)); //w_i ~indep Bern(p_i)
	for(int i = 0; i < n; i++) 
	{
		p_vec[i] = 0.5;
	}
	
	
	///////////// Read in data /////////////
	
	int** x = make_int_mat(n,K); //holds data. x[i][j] is ranking i position j.
	FILE *datafile;
	gsl_matrix *data;
	data = gsl_matrix_alloc(n,K);
	datafile=fopen(datafilename,"r");
	gsl_matrix_fscanf(datafile,data);
	fclose(datafile); //close file
	for(int i = 0; i < n; i++) 
	{
		for(int j = 0; j < K; j++)
		{
			x[i][j]= gsl_matrix_get( data,i,j);
		}
	}
	
	///////////// DATA SPECIFICATION /////////////
	
	int* K_vec = (int*)malloc(n*sizeof(int)); //holds how many entities ranker i was given to consider.
	for(int i = 0; i < n; i++) 
	{
		K_vec[i] = K; //every ranker was given the full set of entities
	}
	
	int* ni_vec = (int*)malloc(n*sizeof(int)); //holds how many entites are reported back by ranker i
	for(int i = 0; i < n; i++) //every ranker provides a complete ranking
	{
		ni_vec[i] = K;
	}
	
    
    ///////////// Initalisation /////////////
    
    //rng
    int seed = time(NULL);
	gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
	if(!fix_seed) //random initalisation based on time
	{
		gsl_rng_set(r,1 + seed); 
	}
    
    double ran_U; //random uniform variable
    int mix_over_r = weighted_pl; //rename
    int* r_vec = (int*)malloc(n * sizeof(int)); //holds the value of w_i for each ranker 
	if(mix_over_r) //if weighted PL model
	{
		for(int i = 0; i < n; i++) //initialise r_i (w_i) from the prior probabilities p_vec
		{
			ran_U = gsl_rng_uniform(r);
			r_vec[i] = sample_bern(ran_U, p_vec[i]);
		}
	}
	else
	{
		for(int i = 0; i < n; i++)  //fix equal to one (SPL)
		{
			r_vec[i] = 1;
		}
	}	
    
    double* lambda = (double*)calloc(K, sizeof(double));  
	for(int j = 0; j < K; j++)
	{
		lambda[j] = gsl_ran_gamma(r,a_vec_lambda[j],1.0/b_lambda); //sample lambdas from prior
	}

	
	//draw latent variables z_ij from  their FCD
	double** z = make_double_mat(n,K); //hold latent variables 
	for(int i = 0; i < n; i++)
	{
		if(r_vec[i] == 1)
		{
			double sum = 0;
			for (int q = 0; q < K_vec[i]; q++)
			{
				sum += lambda[x[i][q]-1];
			}
			
			for (int j = 0; j < ni_vec[i]; j++) 
			{
				z[i][j] = gsl_ran_exponential(r, 1.0/sum);
				
				sum -= lambda[x[i][j]-1];
			}
			
		}else
		{
			for(int j = 0; j < ni_vec[i]; j++) 
			{
				z[i][j] = gsl_ran_exponential(r, 1.0/(K_vec[i]-j));
			}	
		}	
	}
	
	///////////// Quantities used in MCMC /////////////
		
	double* betavec = (double*)calloc(K, sizeof(double));
    double log_likeli_r1, log_likeli_r0;
    double* ptilde = (double*)malloc(n * sizeof(double));
    
    double sum_a_vec_lambda = 0.0;
    for(int k = 0; k < K; k++)
    {
		sum_a_vec_lambda += a_vec_lambda[k];
	}
	
	///////////// Output files /////////////
	
	//check if output directories exist, if not create them
	struct stat st = {0};
	if(stat("spl_outputs", &st) == -1)
	{
		mkdir("spl_outputs", 0777);
		printf("Created new output directory as /spl_outputs did not exist\n");
	}
	FILE* lambda_out = fopen("spl_outputs/lambdaout.ssv","w");
	FILE* likeli_out = fopen("spl_outputs/likeliout.ssv","w");
	FILE* p_out = fopen("spl_outputs/pout.ssv","w"); //Uncomment if using Weighted PL model
	FILE* rankers_p_out = fopen("spl_outputs/ranker_weights.ssv", "w"); //Uncomment if using Weighted PL model
	if(weighted_pl ==  0)
	{
		remove("spl_outputs/pout.ssv");
		remove("spl_outputs/ranker_weights.ssv");
	}
	
	
				
	///////////// Print curret time and total number of iterations to be performed /////////////
	
	time_t rawtime;
	struct tm * timeinfo;
	printf("Starting MCMC for %d iterations\n",  burn_in + (nsample*thin));
	time(&rawtime);
	timeinfo = localtime(&rawtime);
	printf("Current local time and date: %s", asctime(timeinfo));
	
	
///////////// Start MCMC /////////////	
for(int itercount = 0; itercount < (burn_in + (nsample*thin)+1); itercount++)
{
	///////////// Sample lambda /////////////	
	
	//compute beta_mat. beta[s][t] is the number of times lambda[s][t] corresponds to an entity in an informative ranking 
	memset(betavec, 0, K*sizeof(double));
	for(int i = 0; i < n; i++)
	{
		if(r_vec[i] == 1)
		{
			for(int j = 0; j < ni_vec[i]; j++) 
			{	
				betavec[x[i][j]-1] ++;
			}	
		}
	}
	
	for(int k = 0; k < K; k++)
	{
		double sum_lambda = 0;
		for(int i = 0; i < n; i++)
		{
			if(r_vec[i] == 1)
			{	
				for(int j = 0; j < ni_vec[i]; j++) 
				{
					for(int m = j; m < K_vec[i]; m++) 
					{
						if((x[i][m]-1) == k)
						{
							sum_lambda += z[i][j];
						}
					}
				}		
			}
		}

		lambda[k] = gsl_ran_gamma(r, (double) a_vec_lambda[k] + betavec[k], (double) 1.0/(b_lambda + sum_lambda)); //sample from FCD
	}
	
	
	///////////// Rescale lambda values /////////////	
 
    double rescale_sum = 0;
    for(int k = 0; k < K; k++)
    {
		rescale_sum += lambda[k]; //sum all unique lambda values
	}
	double cap_lambda = gsl_ran_gamma(r, sum_a_vec_lambda, 1.0/b_lambda); //draw prior realisation of sum of unqiue lambdas
    double ratio = cap_lambda/rescale_sum; //scaling ratio
    for(int k = 0; k < K; k++)
    {
		lambda[k] *= ratio;//rescale
	}
	
	
	///////////// Sample latent variables Z /////////////
	
	for(int i = 0; i < n; i++)
	{
		if(r_vec[i] == 1)
		{
			double sum = 0;
			for(int q = 0; q < K_vec[i]; q++)
			{
				sum += lambda[x[i][q]-1];
			}
			
			for(int j = 0; j < ni_vec[i]; j++) 
			{
				z[i][j] = gsl_ran_exponential(r, 1.0/sum);
				
				sum -= lambda[x[i][j]-1];
			}
			
		}else
		{
			for (int j = 0; j < ni_vec[i]; j++)
			{
				z[i][j] = gsl_ran_exponential(r, 1.0/(K_vec[i]-j));
			}	
		}	
	}
	
	
	
	///////////// Sample ranker weights (w_i in paper) if using Weighted PL model /////////////	
	
	if(mix_over_r)
	{
		for(int i = 0; i < n; i++)
		{
			r_vec[i] = 1;
			log_likeli_r1 = loglikeli_one_rank_pl(x,lambda,z,ni_vec,i,K_vec,r_vec);
			
			r_vec[i] = 0;
			log_likeli_r0 = loglikeli_one_rank_pl(x,lambda,z,ni_vec,i,K_vec,r_vec);
			
				
			ptilde[i] = 1.0/(1.0 + (((1.0-p_vec[i])/p_vec[i])*exp(log_likeli_r0 - log_likeli_r1)));				
			
			ran_U = gsl_rng_uniform(r);
			r_vec[i] = sample_bern(ran_U, ptilde[i]);

		}
	}
		
	
	
	///////////// Print statements /////////////
	
	if(itercount > burn_in && itercount % thin == 0)  
	{
		
		double full_loglikeli_out = 0;		
		for(int i = 0; i < n; i++) //compute observed data (log) likelihood
		{	
			full_loglikeli_out += observed_data_loglikeli_one_rank_pl(x, lambda, ni_vec, i,  K_vec, r_vec);
		}
		fprintf(likeli_out, "%f \n", full_loglikeli_out);
		print_array_dbl(lambda_out, lambda, K); //print lambda values
		
		if(weighted_pl == 1) //print ranker weights if using Weighted PL model
		{
			print_array_dbl(p_out, ptilde, n); //Uncomment if using Weighted PL model
			print_array_int(rankers_p_out, r_vec, n); //Uncomment if using Weighted PL model
		}
		
			 
	}
	
	///////////// Update user on iterations and time /////////////
	
	if(itercount > 0 && itercount % (thin*1000) == 0) //print iteration and time update to screen
	{
		time(&rawtime);
		timeinfo = localtime(&rawtime);
		printf("Current iteration: %d \t Current local time and date: %s", itercount, asctime(timeinfo));
	}	
	
	
}
	
return 0;
}



int** make_int_mat(int n, int m)
{
	int i;
	// this does the same as int arr[n][m]
	int** arr = (int**)calloc(n,sizeof(int*));
	for(i = 0; i < n; i++)
	{
		arr[i] = (int*)calloc(m,sizeof(int));
	}
	return arr;
}


double** make_double_mat(int n, int m)
{
	int i;
	// this does the same as int arr[n][m]
	double** arr = (double**)calloc(n,sizeof(double*));
	for(i = 0; i < n; i++)
	{
		arr[i] = (double*)calloc(m,sizeof(double));
	}
	return arr;
}


double loglikeli_one_rank_pl(int** x, double* lambda_dagger, double** z, int* ni_vec, int i, int* K_vec, int* r_vec)
{

	int j,ranking;

	double sum;
	double likeli;
	
	if(r_vec[i] == 1)
	{
		likeli = 0;
		sum = 0;
		for(j = 0; j < K_vec[i]; j++) //sum all lambdas
		{
			sum += lambda_dagger[x[i][j]-1];
		}
			
		for(j = 0; j < ni_vec[i]; j++) 
		{
			likeli += log(lambda_dagger[x[i][j]-1]) - z[i][j]*sum;
			sum -= lambda_dagger[x[i][j]-1]; //take off the jth lambda value from the sum...saves looping each time
		}
	}else
	{
		likeli = 0;
		for(j = 0; j < ni_vec[i]; j++) 
		{
			likeli +=  - z[i][j]*(K_vec[i]-j);
		}
	}
	return(likeli);

}

double observed_data_loglikeli_one_rank_pl(int** x, double* lambda_dagger, int* ni_vec, int i, int* K_vec, int* r_vec)
{
	int j;

	double sum;
	double likeli;
	
	if(r_vec[i] == 1)
	{
		likeli = 0;
		sum = 0;
		for(j = 0; j < K_vec[i]; j++) //sum all lambdas
		{
			sum += lambda_dagger[x[i][j]-1];
		}
			
		for(j = 0; j < ni_vec[i]; j++)
		{
			likeli += log(lambda_dagger[x[i][j]-1]) - log(sum);
			sum -= lambda_dagger[x[i][j]-1]; //take off the jth lambda value from the sum...saves looping each time
		}
	}else
	{
		likeli = 0;
		for(j = 0; j < ni_vec[i]; j++) 
		{
			likeli +=  - log(K_vec[i]-j);
		}
	}
	return(likeli);
	
}


void print_array_int(FILE* file_name, int* array, int size)
{
	int i;
	
	for(i = 0; i < (size-1); i++)
	{
		fprintf(file_name, "%d ", array[i]);
	}
	fprintf(file_name, "%d\n", array[size-1]);
} 

void print_array_dbl(FILE* file_name, double* array, int size)
{
	int i;
	
	for(i = 0; i < (size-1); i++)
	{
		fprintf(file_name, "%f ", array[i]);
	}
	fprintf(file_name, "%f\n", array[size-1]);
} 

int sample_bern(double U, double prob)
{
	if( U < prob)
	{
		return(1);
	}else
	{
		return(0);
	}
}

//eof
