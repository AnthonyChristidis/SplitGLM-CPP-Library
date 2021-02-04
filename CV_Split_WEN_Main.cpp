/*
 * ===========================================================
 * File Type: CPP
 * File Name: CV_Split_WEN_Main.cpp
 * Package Name: SplitGLM
 * 
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "config.h"

#include "CV_WEN.hpp"
#include "CV_Split_WEN.hpp" 

// [[Rcpp::export]]
Rcpp::List CV_SWEN_Main(arma::mat & x, arma::vec & y,  
                        arma::uword & type, 
                        arma::uword & G, arma::uword & include_intercept, 
                        double & alpha_s, double & alpha_d, 
                        arma::uword & n_lambda_sparsity, arma::uword & n_lambda_diversity,
                        double & tolerance, arma::uword & max_iter,
                        arma::uword & n_folds,
                        arma::uword & active_set,
                        arma::uword & full_diversity,
                        arma::uword & n_threads){
  
  // Case for a single model
  if(G==1){ 
    
    CV_WEN model = CV_WEN(x, y,
                          type, 
                          include_intercept, 
                          alpha_s,
                          n_lambda_sparsity,
                          tolerance, max_iter,
                          n_folds,
                          n_threads);
    // Computation of coefficients
    if(active_set)
      model.Compute_CV_Betas_Active();
    else
      model.Compute_CV_Betas(); 
    
    // List for output
    Rcpp::List output;
    output["Lambda_Sparsity"] = model.Get_Lambda_Sparsity_Grid();
    output["Lambda_Sparsity_Min"] = model.Get_lambda_sparsity_opt();
    output["CV_Errors"] = model.Get_CV_Error();
    output["Optimal_Index"] = (model.Get_CV_Error()).index_min() + 1;
    output["Intercept"] = model.Get_Intercept();
    output["Betas"] = model.Get_Coef();
    return(output); 
  }
  else{ // Case for more than one model
    
    CV_Split_WEN model = CV_Split_WEN(x, y,
                                      type, 
                                      G, include_intercept, 
                                      alpha_s, alpha_d, 
                                      n_lambda_sparsity, n_lambda_diversity,
                                      tolerance, max_iter,
                                      n_folds,
                                      n_threads);
    
    // Computation of coefficients
    if(full_diversity)
      model.Compute_CV_Betas_Full_Diversity(); 
    else
      model.Compute_CV_Betas();
    
    // List for output
    Rcpp::List output;
    output["Lambda_Diversity_Min"] = model.Get_lambda_diversity_opt();
    output["Lambda_Sparsity"] = model.Get_Lambda_Sparsity_Grid();
    output["Lambda_Sparsity_Min"] = model.Get_lambda_sparsity_opt();
    output["CV_Errors"] = model.Get_CV_Error_Sparsity();
    output["Optimal_Index"] = (model.Get_CV_Error_Sparsity()).index_min() + 1;
    output["Intercept"] = model.Get_Intercept();
    output["Betas"] = model.Get_Coef();
    return(output); 
  }
}
