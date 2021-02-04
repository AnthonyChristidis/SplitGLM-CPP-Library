/*
 * ===========================================================
 * File Type: CPP
 * File Name: Split_WEN.cpp
 * Package Name: SplitGLM
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#include "Split_WEN.hpp"

#include <RcppArmadillo.h>

#include "config.h"

#include <math.h>

// Constants - COMPUTATION 
const static double DIVERGENCE_CONST = 1e-4;
const static double ACTIVE_SET_PRE_ITER = 2;

// Constructor for Split WEN
Split_WEN::Split_WEN(arma::mat x, arma::vec y,
                     arma::uword & type,
                     arma::uword & G, arma::uword & include_intercept,
                     double alpha_s, double alpha_d, 
                     double lambda_sparsity, double lambda_diversity,
                     double tolerance, arma::uword max_iter):
  x(x), y(y),
  type(type),
  G(G), include_intercept(include_intercept),
  alpha_s(alpha_s), alpha_d(alpha_d), lambda_sparsity(lambda_sparsity), lambda_diversity(lambda_diversity),
  tolerance(tolerance), max_iter(max_iter){

  // Initializing the object
  Initialize();
}

void Split_WEN::Initialize(){

  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  x_std_2 = x_std % x_std;
  mu_y = arma::mean(y);

  // Setting the parameters
  n = x.n_rows;
  p = x.n_cols;
  expected_val = arma::zeros(n,G);
  weights = arma::zeros(n,G);
  residuals = arma::zeros(n,G);
  betas = arma::zeros(p,G);
  new_betas = arma::zeros(p,G);
  intercept = arma::zeros(G);
  new_intercept = arma::zeros(G);

  // Setting initial values and function pointers for expected values and weights
  if(type==1){ // Linear Model

    weights = arma::ones(n,G);
    if(include_intercept==1){
      intercept.fill(arma::mean(y));
      Compute_Expected_Weights = &Split_WEN::Linear_Update_Intercept;
    }
    else
      Compute_Expected_Weights = &Split_WEN::Linear_Update;
  }
  else if(type==2){ // Logistic Regression

    if(include_intercept==1){
      intercept.fill(std::log(arma::mean(y)/(1-arma::mean(y))));
      Compute_Expected_Weights = &Split_WEN::Logistic_Update_Intercept;
    }
    else
      Compute_Expected_Weights = &Split_WEN::Logistic_Update;
  }
  else if(type==3){ // Gamma GLM

    if(include_intercept==1){
      intercept.fill(-1/arma::mean(y));
      Compute_Expected_Weights = &Split_WEN::Gamma_Update_Intercept;
    }
    else
      Compute_Expected_Weights = &Split_WEN::Gamma_Update;

  }
  else if(type==4){ // Poisson GLM

    if(include_intercept==1){
      intercept.fill(std::log(arma::mean(y)));
      Compute_Expected_Weights = &Split_WEN::Poisson_Update_Intercept;
    }
    else
      Compute_Expected_Weights = &Split_WEN::Poisson_Update;
  }
  
  // Convenience vector for soft-thresholding
  xj_y = x_std.t()*y;

  // Initializing the expected values, weights, and residuals
  for(arma::uword group_ind=0; group_ind<G; group_ind++){
    Adjust_Expected_Weights(group_ind);
  }
}

// Function to compute the sparsity penalty
double Split_WEN::Sparsity_Penalty(){ 

  return(lambda_sparsity*((1-alpha_s)*0.5*pow(arma::norm(betas, "fro"), 2) + alpha_s*arma::accu(arma::abs(betas))));
}
// Function to compute the diversity penalty
double Split_WEN::Diversity_Penalty(){

  double diversity_penalty = 0;
  arma::mat gram_betas = arma::zeros(betas.n_rows, betas.n_rows);
  gram_betas = arma::abs(betas.t())*arma::abs(betas);
  gram_betas.diag().zeros();
  diversity_penalty = 0.5*arma::accu(gram_betas);
  diversity_penalty *= lambda_diversity;
  return(diversity_penalty);
}

// Functions to set new data
void Split_WEN::Set_X(arma::mat & x){

  this->x = x;
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  x_std_2 = x_std % x_std;
}
void Split_WEN::Set_Y(arma::vec & y){
  this->y = y;
}

// Functions to set maximum number of iterations and tolerance
void Split_WEN::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void Split_WEN::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

// Function to adjust the expected value and weight vector
void Split_WEN::Adjust_Expected_Weights(arma::uword & group){

  (*Compute_Expected_Weights)(group,
                              this->x_std,
                              this->new_intercept, this->new_betas,
                              this->expected_val, this->weights);
}
// Function to adjust the residuals
void Split_WEN::Adjust_Residuals(arma::uword & group){
  residuals.col(group) = y - expected_val.col(group);
}
// Iterative Soft function
double Split_WEN::Soft(double z, double gamma){

  return(((z<0) ? -1 : 1)*fmax(std::abs(z)-gamma,0));
}

// Function to compute the weights penalty (from diversity penalty) - ABS
arma::vec Split_WEN::Beta_Weights_Abs(arma::uword & group){
  
  arma::mat sum_abs = arma::zeros(new_betas.n_rows,1);
  arma::vec indices = arma::ones(new_betas.n_cols,1);
  indices[group] = 0;
  sum_abs = arma::abs(new_betas)*indices;
  return(sum_abs);
}
// Function to compute the weights penalty (from diversity penalty) - SQ
arma::vec Split_WEN::Beta_Weights_Sq(arma::uword & group){
  
  arma::mat sum_sq = arma::zeros(new_betas.n_rows,1);
  arma::vec indices = arma::ones(new_betas.n_cols,1);
  indices[group] = 0;
  sum_sq = arma::square(new_betas)*indices;
  return(sum_sq);
}

void Split_WEN::Cycle_Full_Set(arma::uword & group){
  
  // Computation of weight penalty from other groups
  arma::vec soft_penalty = lambda_sparsity*alpha_s + alpha_d*lambda_diversity*Beta_Weights_Abs(group);
  arma::vec denom_penalty = (1-alpha_s)*lambda_sparsity + (1-alpha_d)*lambda_diversity*Beta_Weights_Sq(group);
  
  // Initial iteration over all the variables
  new_intercept(group) = ((include_intercept) ? (intercept(group) + (n*(mu_y-arma::mean(expected_val.col(group))))/arma::accu(weights.col(group))) : 0);
  // Update expected values, weights and residuals if there is a change (intercept)
  if(std::fabs(new_intercept(group)-intercept(group))>=EQUAL_TOLERANCE){
    Adjust_Expected_Weights(group);
  }
  // Initial cycle over all variables (and intercept)
  for(arma::uword j=0; j<p; j++){

    w_xj2 = arma::dot(x_std_2.col(j), weights.col(group));
    new_betas(j,group) = Soft((xj_y[j]-arma::dot(x_std.col(j),expected_val.col(group)))/n + betas(j,group)*w_xj2/n, soft_penalty[j]) /
      (w_xj2/n + denom_penalty[j]);
    // Update expected values, weights and residuals if there is a change (coefficients)
    if(std::fabs(new_betas(j,group)-betas(j,group))>=EQUAL_TOLERANCE){
      Adjust_Expected_Weights(group);
    }
  }
}

void Split_WEN::Cycle_Active_Set(arma::uword & group){
  
  // Active variables
  arma::uvec active_set_group = arma::find(betas.col(group)!=0);
  
  // Computation of weight penalty from other groups
  arma::vec soft_penalty = lambda_sparsity*alpha_s + lambda_diversity*Beta_Weights_Abs(group);
  arma::vec denom_penalty = (1-alpha_s)*lambda_sparsity + (1-alpha_d)*lambda_diversity*Beta_Weights_Sq(group);
  
  // Initial iteration over all the variables
  new_intercept(group) = ((include_intercept) ? (intercept(group) + (n*(mu_y-arma::mean(expected_val.col(group))))/arma::accu(weights.col(group))) : 0);
  // Update expected values, weights and residuals if there is a change (intercept)
  if(std::fabs(new_intercept(group)-intercept(group))>=EQUAL_TOLERANCE){
    Adjust_Expected_Weights(group);
  }
  // Initial cycle over all variables (and intercept)
  for(arma::uword j=0; j<active_set_group.n_elem; j++){
    
    w_xj2 = arma::dot(x_std_2.col(active_set_group[j]), weights.col(group));
    new_betas(active_set_group[j],group) = Soft((xj_y[active_set_group[j]]-arma::dot(x_std.col(active_set_group[j]),expected_val.col(group)))/n + betas(active_set_group[j],group)*w_xj2/n, soft_penalty[active_set_group[j]]) /
      (w_xj2/n + denom_penalty[j]);
    // Update expected values, weights and residuals if there is a change (coefficients)
    if(std::fabs(new_betas(active_set_group[j],group)-betas(active_set_group[j],group))>=EQUAL_TOLERANCE){
      Adjust_Expected_Weights(group);
    }
  }
}

void Split_WEN::Compute_Coef(){

  for(arma::uword iter=0; iter<max_iter; iter++){

    // Cycle over all variables for all groups
    for(arma::uword group_ind=0; group_ind<G; group_ind++){
      Cycle_Full_Set(group_ind);
    }

    // End of coordinate descent if variables are already converged
    if(arma::square(arma::mean(new_betas,1)-arma::mean(betas,1)).max()<tolerance){
      intercept = new_intercept;
      betas = new_betas;
      Scale_Coefficients();
      Scale_Intercept();
      return;
    }

    // Adjusting the intercept and betas
    intercept = new_intercept;
    betas = new_betas;
  }

  // Scaling of coefficients and intercept
  Scale_Coefficients();
  Scale_Intercept();
}

// Comparison of active sets of variables after cycling
arma::uword Split_WEN::Compare_Active_Set(){  
  
  // Finding the candidate active set
  arma::mat candidate_active_set = arma::zeros(p,G);
  arma::colvec candidate_active_set_helper = arma::zeros(p);
  
  for(arma::uword group_ind=0; group_ind<G; group_ind++){   
    
    candidate_active_set_helper.zeros(); 
    candidate_active_set_helper(arma::find(new_betas.col(group_ind)!=0)).fill(1);
    candidate_active_set.col(group_ind) = candidate_active_set_helper;
  }
  
  // Updating the parameters
  intercept = new_intercept;
  betas = new_betas;
  
  // Difference between active sets
  arma::uword active_difference = arma::accu(arma::abs(active_set - candidate_active_set));
  
  if(active_difference==0)
    return 1;
  else 
    return 0;
}

void Split_WEN::Compute_Coef_Active(){
  
  for(arma::uword iter=0; iter<ACTIVE_SET_PRE_ITER; iter++){
    
    // Cycle over all variables for all groups
    for(arma::uword group_ind=0; group_ind<G; group_ind++){
      Cycle_Full_Set(group_ind);
    }
    
    // End of coordinate descent if variables are already converged
    if(arma::square(arma::mean(new_betas,1)-arma::mean(betas,1)).max()<tolerance){
      intercept = new_intercept;
      betas = new_betas;
      Scale_Coefficients();
      Scale_Intercept();
      return;
    }
    
    // Adjusting the intercept and betas
    intercept = new_intercept;
    betas = new_betas;
  }
  
  // Matrix to store active variables
  active_set = arma::zeros(p, G);
  arma::colvec active_set_helper = arma::zeros(p);
  
  // Active set iterations
  do{
    
    // Active set for the variables
    for(arma::uword group_ind=0; group_ind<G; group_ind++){   
      
      active_set_helper.zeros(); 
      active_set_helper(arma::find(betas.col(group_ind)!=0)).fill(1);
      active_set.col(group_ind) = active_set_helper;
    }
    
    for(arma::uword iter=0; iter<max_iter; iter++){
      
      // Cycle over all variables for all groups
      for(arma::uword group_ind=0; group_ind<G; group_ind++){
        Cycle_Active_Set(group_ind);
      }
      
      // End of coordinate descent if variables are already converged
      if(arma::square(arma::mean(new_betas,1)-arma::mean(betas,1)).max()<tolerance){
        intercept = new_intercept;
        betas = new_betas;
        break;
      }
      
      // Adjusting the intercept and betas
      intercept = new_intercept;
      betas = new_betas;
    }
    
    // Cycle over all variables for all groups
    for(arma::uword group_ind=0; group_ind<G; group_ind++){
      Cycle_Full_Set(group_ind);
    }
      
  } while(Compare_Active_Set()!=1);
  
  // Scaling of coefficients and intercept
  Scale_Coefficients();
  Scale_Intercept();
}


// Functions to set and get alpha_s
void Split_WEN::Set_Alpha_S(double alpha_s){
  this->alpha_s = alpha_s;
}
double Split_WEN::Get_Alpha_S(){
  return(this->alpha_s);
}

// Functions to set and get lambda_sparsity
void Split_WEN::Set_Lambda_Sparsity(double lambda_sparsity){
  this->lambda_sparsity = lambda_sparsity;
}
double Split_WEN::Get_Lambda_Sparsity(){
  return(this->lambda_sparsity);
}
// Functions to set and get lambda_diversity
void Split_WEN::Set_Lambda_Diversity(double lambda_diversity){
  this->lambda_diversity = lambda_diversity;
}
double Split_WEN::Get_Lambda_Diversity(){
  return(this->lambda_diversity);
}

// Functions to return expected values and weights
arma::mat Split_WEN::Get_Expected(){
  return(expected_val);
}
arma::mat Split_WEN::Get_Weights(){
  return(weights);
}

// Functions to return coefficients and the intercept
arma::mat Split_WEN::Get_Coef(){
  return(betas);
}
arma::vec Split_WEN::Get_Intercept(){
  return(intercept);
}

arma::mat Split_WEN::Get_Coef_Scaled(){
  return(betas_scaled);
}
arma::vec Split_WEN::Get_Intercept_Scaled(){
  return(intercept_scaled);
}



// Function to return objective function value
double Split_WEN::Get_Objective_Value(){
  
  // Initializing the expected values, weights, and residuals
  for(arma::uword group_ind=0; group_ind<G; group_ind++){
    Adjust_Residuals(group_ind);
  }
  
  return(arma::accu(arma::square(residuals))/(2*n) +
         Sparsity_Penalty() +
         Diversity_Penalty());
}

void Split_WEN::Scale_Coefficients(){
  
  betas_scaled = betas;
  betas_scaled.each_col() /= (sd_x.t());
}

void Split_WEN::Scale_Intercept(){
  
  intercept_scaled = intercept;
  for(arma::uword group_ind=0; group_ind<G; group_ind++){
    intercept_scaled(group_ind) = ((include_intercept) ? 1 : 0)*(intercept(group_ind) - arma::accu(betas_scaled.col(group_ind) % mu_x.t()));
  }
}

Split_WEN::~Split_WEN(){
  // Class destructor
}

/*
* ________________________________________________
* Static Functions - Weights and Expected Values
* ________________________________________________
*/

void Split_WEN::Linear_Update(arma::uword & group,
                              arma::mat & x,
                              arma::vec & intercept, arma::mat & betas,
                              arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = x*betas.col(group);
}
void Split_WEN::Linear_Update_Intercept(arma::uword & group,
                                        arma::mat & x,
                                        arma::vec & intercept, arma::mat & betas,
                                        arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = intercept(group)+x*betas.col(group);
}

void Split_WEN::Logistic_Update(arma::uword & group,
                                arma::mat & x,
                                arma::vec & intercept, arma::mat & betas,
                                arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = arma::exp(x*betas.col(group)) % (1/(1+arma::exp(x*betas.col(group))));
  weights.col(group) = expected_val.col(group) % (1-expected_val.col(group));
  // Correction factor to avoid divergence
  arma::colvec weights_temp = weights.col(group);
  weights_temp.elem(find(expected_val.col(group)<DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
  expected_val.elem(find(expected_val.col(group)<DIVERGENCE_CONST)).zeros();
  weights_temp.elem(find(expected_val.col(group)>1-DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
  expected_val.elem(find(expected_val.col(group)>1-DIVERGENCE_CONST)).ones();
  weights.col(group) = weights_temp;
}
void Split_WEN::Logistic_Update_Intercept(arma::uword & group,
                                          arma::mat & x,
                                          arma::vec & intercept, arma::mat & betas,
                                          arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = arma::exp(intercept(group)+x*betas.col(group)) % (1/(1+arma::exp(intercept(group)+x*betas.col(group))));
  weights.col(group) = expected_val.col(group) % (1-expected_val.col(group));
  // Correction factor to avoid divergence
  arma::colvec weights_temp = weights.col(group);
  weights_temp.elem(find(expected_val.col(group)<DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
  expected_val.elem(find(expected_val.col(group)<DIVERGENCE_CONST)).zeros();
  weights_temp.elem(find(expected_val.col(group)>1-DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
  expected_val.elem(find(expected_val.col(group)>1-DIVERGENCE_CONST)).ones();
  weights.col(group) = weights_temp;
}

void Split_WEN::Gamma_Update(arma::uword & group,
                             arma::mat & x,
                             arma::vec & intercept, arma::mat & betas,
                             arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = -1/(x*betas.col(group));
  weights.col(group) = arma::square(expected_val.col(group));
}
void Split_WEN::Gamma_Update_Intercept(arma::uword & group,
                                       arma::mat & x,
                                       arma::vec & intercept, arma::mat & betas,
                                       arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = -1/(intercept(group)+x*betas.col(group));
  weights.col(group) = arma::square(expected_val.col(group));
}

void Split_WEN::Poisson_Update(arma::uword & group,
                               arma::mat & x,
                               arma::vec & intercept, arma::mat & betas,
                               arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = arma::exp(x*betas.col(group));
  weights.col(group) = expected_val.col(group);
}
void Split_WEN::Poisson_Update_Intercept(arma::uword & group,
                                         arma::mat & x,
                                         arma::vec & intercept, arma::mat & betas,
                                         arma::mat & expected_val, arma::mat & weights){

  expected_val.col(group) = arma::exp(intercept(group) + x*betas.col(group));
  weights.col(group) = expected_val.col(group);
}





