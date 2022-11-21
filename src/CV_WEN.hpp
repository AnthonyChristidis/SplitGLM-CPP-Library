/*
 * ===========================================================
 * File Type: HPP
 * File Name: CV_WEN.hpp
 * Package Name: SplitGLM
 * 
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#ifndef CV_WEN_hpp
#define CV_WEN_hpp

#include <RcppArmadillo.h>

#include "config.h"

#include "WEN.hpp"
#include <vector>

class CV_WEN{
  
private: 
  
  // Variables supplied by user
  arma::mat x; 
  arma::vec y;
  arma::uword type;
  arma::uword include_intercept;
  double alpha;
  arma::uword n_lambda_sparsity;
  double tolerance;
  arma::uword max_iter;
  arma::uword n_folds;
  // Variables created inside class
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (does not include intercept term)
  arma::vec lambda_sparsity_grid;
  double eps;
  arma::vec intercepts;
  arma::mat betas;
  arma::mat cv_errors_mat;
  arma::vec cv_errors;
  arma::uword index;
  arma::uword n_threads;

  // Private function to initialize the weighted elastic models (each fold)
  void Initialize();
  
  // Method to get the grid of lambda sparsity
  void Compute_Lambda_Sparsity_Grid();
  
  // Private function to create the folds
  arma::uvec Set_Diff(const arma::uvec & big, const arma::uvec & small);
  
  // Private function to compute the CV-MSPE over the folds
  double (*Compute_Deviance)(arma::mat x, arma::vec y, 
                             double intercept, arma::vec betas);

public:
  
  // Constructor - with data
  CV_WEN(arma::mat & x, arma::vec & y,
         arma::uword & type, arma::uword & include_intercept, 
         double & alpha, arma::uword & n_lambda_sparsity,
         double & tolerance, arma::uword & max_iter,
         arma::uword & n_folds,
         arma::uword & n_threads);
  
  // Functions to set new data
  void Set_X(arma::mat & x);
  void Set_Y(arma::vec & y);
  
  // Method to set alpha to new value and return current alpha
  void Set_Alpha(double alpha);
  double Get_Alpha();
  // Method to get the grid of lambda sparsity
  arma::vec Get_Lambda_Sparsity_Grid();
  
  // Functions to set maximum number of iterations and tolerance
  void Set_Max_Iter(arma::uword & max_iter);
  void Set_Tolerance(double & tolerance);
  
  // Cross-validation
  arma::vec Get_CV_Error();
  
  // Optimal penalty parameter - Sparsity
  double Get_lambda_sparsity_opt();
  
  // Methods to return coefficients
  arma::mat Get_Coef();
  arma::vec Get_Intercept();
  
  // Optimal sparsity parameters
  arma::uword Get_Optimal_Index();
  
  // Coordinate descent algorithms for coefficients
  void Compute_CV_Betas();
  void Compute_CV_Betas_Active();
  
  // Static functions for the deviance
  static double Linear_Deviance(arma::mat x, arma::vec y, 
                                double intercept, arma::vec betas);
  static double Logistic_Deviance(arma::mat x, arma::vec y, 
                                  double intercept, arma::vec betas);
  static double Gamma_Deviance(arma::mat x, arma::vec y, 
                               double intercept, arma::vec betas);
  static double Poisson_Deviance(arma::mat x, arma::vec y, 
                                 double intercept, arma::vec betas);

  // Virtual destructor
  ~CV_WEN();
};

#endif // CV_WEN_hpp




