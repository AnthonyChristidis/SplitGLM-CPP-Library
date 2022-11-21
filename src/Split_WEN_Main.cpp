/*
 * ===========================================================
 * File Type: CPP
 * File Name: Split_WEN_Main.cpp
 * Package Name: SplitGLM
 * 
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include "config.h"

#include "WEN.hpp" 
#include "Split_WEN.hpp"

// [[Rcpp::export]]
Rcpp::List Split_WEN_Main(arma::mat & x, arma::vec & y,  
                          arma::uword & type, 
                          arma::uword & G, arma::uword & include_intercept, 
                          double & alpha_s, double & alpha_d, 
                          double & lambda_sparsity, double & lambda_diversity,
                          double & tolerance, arma::uword & max_iter,
                          arma::uword & active_set_convergence){
  
  if(G==1){ // Weighted Elastic Net Case (Single Group)
    
    WEN model = WEN(x, y, 
                    type, include_intercept, 
                    alpha_s, 
                    lambda_sparsity, 
                    tolerance, max_iter);
    
    if(active_set_convergence)
      model.Compute_Coef_Active();
    else
      model.Compute_Coef();
    Rcpp::List output;
    output["Intercept"] = model.Get_Intercept_Scaled();
    output["Betas"] = model.Get_Coef_Scaled();
    return(output);
  }
  else{ // Split Weighted Elastic Net Case (More Than One Group)
    
    Split_WEN model = Split_WEN(x, y, 
                                type, 
                                G, include_intercept, 
                                alpha_s, alpha_d,
                                lambda_sparsity, lambda_diversity,
                                tolerance, max_iter);
    
    if(active_set_convergence)
      model.Compute_Coef_Active();
    else
      model.Compute_Coef(); 
    Rcpp::List output;
    output["Intercept"] = model.Get_Intercept_Scaled();
    output["Betas"] = model.Get_Coef_Scaled();
    return(output);
  }
}

