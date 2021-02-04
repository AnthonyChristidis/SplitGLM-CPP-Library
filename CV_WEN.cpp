/*
 * ===========================================================
 * File Type: CPP
 * File Name: CV_WEN.hpp
 * Package Name: SplitGLM
 * 
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#include <RcppArmadillo.h>

#include "config.h"

#include "WEN.hpp"
#include "CV_WEN.hpp"

#include <vector>

// Constructor - with data
CV_WEN::CV_WEN(arma::mat & x, arma::vec & y,
               arma::uword & type, arma::uword & include_intercept,
               double & alpha, arma::uword & n_lambda_sparsity,
               double & tolerance, arma::uword & max_iter,
               arma::uword & n_folds,
               arma::uword & n_threads): 
  x(x), y(y), 
  type(type), include_intercept(include_intercept), 
  alpha(alpha), n_lambda_sparsity(n_lambda_sparsity),
  tolerance(tolerance), max_iter(max_iter), 
  n_folds(n_folds),
  n_threads(n_threads){  
  
  // Initialization of the weighted elastic net models (one for each fold)
  Initialize();
}

// Function to initial the object characteristics
void CV_WEN::Initialize(){ 
  
  // Setting the parameters for the data dimension
  n = x.n_rows;
  p = x.n_cols; 
  
  // Initializing the size of the parameter variables for CV object
  intercepts = arma::zeros(n_lambda_sparsity);
  betas = arma::zeros(p, n_lambda_sparsity);
  cv_errors = arma::zeros(n_lambda_sparsity);
  
  // Computing the grid for lambda_sparsity
  if(n>p)
    eps = 1e-4;
  else
    eps = 1e-2;
  Compute_Lambda_Sparsity_Grid();
  
  // Setting function pointers for the deviance
  if(type==1){ // Linear Model
    Compute_Deviance = &CV_WEN::Linear_Deviance;
  } 
  else if(type==2){ // Logistic Regression
    Compute_Deviance = &CV_WEN::Logistic_Deviance;
  } 
  else if(type==3){ // Gamma GLM
    Compute_Deviance = &CV_WEN::Gamma_Deviance;
  }  
  else if(type==4){ // Poisson GLM
    Compute_Deviance = &CV_WEN::Poisson_Deviance;
  }
}

// Function to create the folds
arma::uvec CV_WEN::Set_Diff(const arma::uvec & big, const arma::uvec & small){
  
  // Find set difference between a big and a small set of variables.
  // Note: small is a subset of big (both are sorted).
  arma::uword m = small.n_elem;
  arma::uword n = big.n_elem;
  arma::uvec test = arma::uvec(n, arma::fill::zeros);
  arma::uvec zeros =arma:: uvec(n - m, arma::fill::zeros);
  
  for (arma::uword j = 0 ; j < m ; j++){
    test[small[j]] = small[j];
  }
  
  test = big - test;
  if(small[0] != 0){
    test[0] = 1;
  }
  zeros = find(test != 0);
  return(zeros);
}

// Method to set lambda to new value and return current lambda
void CV_WEN::Compute_Lambda_Sparsity_Grid(){
  
  // Standardization of design matrix
  arma::rowvec mu_x = arma::mean(x);
  arma::rowvec sd_x = arma::stddev(x, 1);
  arma::mat x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  double lambda_sparsity_max;
  
  // Maximum lambda_sparsity that kills all variables
  lambda_sparsity_max = (1/alpha)*arma::max(abs(y.t()*x_std))/n;
  
  lambda_sparsity_grid =  arma::exp(arma::linspace(std::log(eps*lambda_sparsity_max), std::log(lambda_sparsity_max), n_lambda_sparsity));
} 

// Private function to compute the CV-MSPE over the folds
void CV_WEN::Compute_CV_Deviance(int sparsity_ind,
                                                  arma::mat x_test, arma::vec y_test,
                                                  double intercept, arma::vec betas){
  
  // Computing the CV-Error over the folds
  for(arma::uword fold_ind=0; fold_ind<n_folds; fold_ind++){ 
    cv_errors[sparsity_ind] += (*Compute_Deviance)(x_test, y_test, intercept, betas) / n_folds;
  }
}
 
// Functions to set new data
void CV_WEN::Set_X(arma::mat & x){
  this->x = x;
}
void CV_WEN::Set_Y(arma::vec & y){
  this->y = y;
}

// Method to set alpha to new value and return current alpha
void CV_WEN::Set_Alpha(double alpha){
  this->alpha = alpha;
}
double CV_WEN::Get_Alpha(){
  return(alpha);
}
// Method to get the grid of lambda sparsity
arma::vec CV_WEN::Get_Lambda_Sparsity_Grid(){
  return(lambda_sparsity_grid);
}

// Functions to set maximum number of iterations and tolerance
void CV_WEN::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void CV_WEN::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Cross-validation
arma::vec CV_WEN::Get_CV_Error(){
  return(cv_errors);
}

// Optimal penalty parameter - Sparsity
double CV_WEN::Get_lambda_sparsity_opt(){
  return(Get_Lambda_Sparsity_Grid()[(Get_CV_Error()).index_min()]);
}

// Methods to return coefficients
arma::mat CV_WEN::Get_Coef(){
  return(betas);
}
arma::vec CV_WEN::Get_Intercept(){
  return(intercepts);
}

// Optimal sparsity parameters
arma::uword CV_WEN::Get_Optimal_Index(){
  return(cv_errors.index_min());
}

// Coordinate descent algorithms for coefficients
void CV_WEN::Compute_CV_Betas(){
  
  // Creating indices for the folds of the data
  arma::uvec sample_ind = arma::linspace<arma::uvec>(0, n-1, n);
  arma::uvec fold_ind = arma::linspace<arma::uvec>(0, n, n_folds+1);
  
  // Looping over the folds
  # pragma omp parallel for num_threads(n_threads)
  for(arma::uword fold=0; fold<n_folds; fold++){ 
    
    // Get test and training samples
    arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold], 
                                                 fold_ind[fold + 1] - 1, 
                                                 fold_ind[fold + 1] - fold_ind[fold]);
    arma::uvec train = Set_Diff(sample_ind, test);
    
    // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
    WEN WEN_model_fold = WEN(x.rows(train), y.rows(train),   
                             type, include_intercept,
                             alpha, lambda_sparsity_grid[lambda_sparsity_grid.n_elem-1],
                             tolerance, max_iter);    
    
    // Looping over the different sparsity penalty parameters
    for(int sparsity_ind=lambda_sparsity_grid.n_elem-1; sparsity_ind>=0; sparsity_ind--){

      // Setting the lambda_sparsity value
      WEN_model_fold.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      WEN_model_fold.Compute_Coef();
      // Computing the deviance for the fold (new lambda_sparsity)
      Compute_CV_Deviance(sparsity_ind,
                          x.rows(test), y.rows(test), 
                          WEN_model_fold.Get_Intercept_Scaled(), WEN_model_fold.Get_Coef_Scaled());
      
    } // End of loop over the sparsity parameter values
    
  } // End of loop over the folds
  
  // Computing the parameters for the full data
  WEN WEN_model_full = WEN(x, y,   
                           type, include_intercept,
                           alpha, lambda_sparsity_grid[lambda_sparsity_grid.n_elem-1],
                           tolerance, max_iter);    
  
  // Looping over the different sparsity penalty parameters
  for(int sparsity_ind=lambda_sparsity_grid.n_elem-1; sparsity_ind>=0; sparsity_ind--){
    
    // Setting the lambda_sparsity value
    WEN_model_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
    // Computing the betas for the fold (new lambda_sparsity)
    WEN_model_full.Compute_Coef();
    // Storing the full data models
    intercepts[sparsity_ind] =  WEN_model_full.Get_Intercept_Scaled();
    betas.col(sparsity_ind) = WEN_model_full.Get_Coef_Scaled();

  } // End of loop over the sparsity parameter values
  
}
// Coordinate descent using active set of variables strategy
void CV_WEN::Compute_CV_Betas_Active(){
  
  // Creating indices for the folds of the data
  arma::uvec sample_ind = arma::linspace<arma::uvec>(0, n-1, n);
  arma::uvec fold_ind = arma::linspace<arma::uvec>(0, n, n_folds+1);
  
  // Looping over the folds
  # pragma omp parallel for num_threads(n_threads)
  for(arma::uword fold=0; fold<n_folds; fold++){ 
    
    // Get test and training samples
    arma::uvec test = arma::linspace<arma::uvec>(fold_ind[fold], 
                                                 fold_ind[fold + 1] - 1, 
                                                 fold_ind[fold + 1] - fold_ind[fold]);
    arma::uvec train = Set_Diff(sample_ind, test);
    
    // Initialization of the WEN objects (with the maximum value of lambda_sparsity_grid)
    WEN WEN_model_fold = WEN(x.rows(train), y.rows(train),   
                             type, include_intercept,
                             alpha, lambda_sparsity_grid[lambda_sparsity_grid.n_elem-1],
                             tolerance, max_iter);    
    
    // Looping over the different sparsity penalty parameters
    for(int sparsity_ind=lambda_sparsity_grid.n_elem-1; sparsity_ind>=0; sparsity_ind--){
      
      // Setting the lambda_sparsity value
      WEN_model_fold.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
      // Computing the betas for the fold (new lambda_sparsity)
      WEN_model_fold.Compute_Coef_Active();
      // Computing the deviance for the fold (new lambda_sparsity)
      Compute_CV_Deviance(sparsity_ind,
                          x.rows(test), y.rows(test), 
                          WEN_model_fold.Get_Intercept_Scaled(), WEN_model_fold.Get_Coef_Scaled());
      
    } // End of loop over the sparsity parameter values
    
  } // End of loop over the folds
  
  // Computing the parameters for the full data
  WEN WEN_model_full = WEN(x, y,   
                           type, include_intercept,
                           alpha, lambda_sparsity_grid[lambda_sparsity_grid.n_elem-1],
                           tolerance, max_iter);    
  
  // Looping over the different sparsity penalty parameters
  for(int sparsity_ind=lambda_sparsity_grid.n_elem-1; sparsity_ind>=0; sparsity_ind--){
    
    // Setting the lambda_sparsity value
    WEN_model_full.Set_Lambda_Sparsity(lambda_sparsity_grid[sparsity_ind]);
    // Computing the betas for the fold (new lambda_sparsity)
    WEN_model_full.Compute_Coef_Active();
    // Storing the full data models
    intercepts[sparsity_ind] =  WEN_model_full.Get_Intercept_Scaled();
    betas.col(sparsity_ind) = WEN_model_full.Get_Coef_Scaled();
    
  } // End of loop over the sparsity parameter values
}

// Class destructor
CV_WEN::~CV_WEN(){
}

/*
 * ________________________________________________
 * Static Functions - Deviance
 * ________________________________________________
 */

// Linear Deviance (MSPE)
double CV_WEN::Linear_Deviance(arma::mat x, arma::vec y, 
                               double intercept, arma::vec betas){
  
  return(arma::mean(arma::square(y - (x*betas+intercept))));
}
// Logistic Deviance
double CV_WEN::Logistic_Deviance(arma::mat x, arma::vec y, 
                                 double intercept, arma::vec betas){

  return(-2*arma::mean(y % (intercept + x*betas) - arma::log(1.0 + arma::exp(intercept + x*betas))));
}
// Gamma Deviance (MSPE)
double CV_WEN::Gamma_Deviance(arma::mat x, arma::vec y, 
                              double intercept, arma::vec betas){
  
  return(arma::mean(arma::square(y + 1/(x*betas+intercept))));
}
// Poisson Deviance
double CV_WEN::Poisson_Deviance(arma::mat x, arma::vec y, 
                                double intercept, arma::vec betas){
  
  return(-2*arma::mean(y % (intercept + x*betas) - arma::exp(intercept + x*betas)));
}





