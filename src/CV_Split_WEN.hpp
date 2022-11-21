/*
 * ===========================================================
 * File Type: HPP
 * File Name: CV_Split_WEN.hpp
 * Package Name: SplitGLM
 *
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

#ifndef CV_Split_WEN_hpp
#define CV_Split_WEN_hpp

#include <RcppArmadillo.h>

#include "config.h"

#include "CV_WEN.hpp"
#include "Split_WEN.hpp"

class CV_Split_WEN{

  private:

  // Variables supplied by user
  arma::mat x; 
  arma::vec y;
  arma::uword type;
  arma::uword G;
  arma::uword include_intercept;
  double alpha_s;
  double alpha_d;
  arma::uword n_lambda_sparsity;
  arma::uword n_lambda_diversity;
  double tolerance;
  arma::uword max_iter;
  arma::uword n_folds;
  // Variables created inside class
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (does not include intercept term)
  arma::vec lambda_sparsity_grid;
  arma::vec lambda_diversity_grid;
  double eps;
  arma::mat intercepts;
  arma::cube betas;
  arma::mat cv_errors_sparsity_mat;
  arma::mat cv_errors_diversity_mat;
  arma::vec cv_errors_sparsity;
  arma::vec cv_errors_diversity;
  double cv_opt_old;
  double cv_opt_new;
  arma::uword index_sparsity_opt;
  double lambda_sparsity_opt;
  arma::uword index_diversity_opt;
  double lambda_diversity_opt;
  arma::uword n_threads;

  // Private function to initialize the weighted elastic models (each fold)
  void Initialize();
  
  // Method to get diversity penalty parameter that kills all interactions
  double Get_Lambda_Diversity_Max();
  // Method to get the grid of lambda_sparsity
  void Compute_Lambda_Sparsity_Grid();
  // Method to get the grid of lambda_diversity
  void Compute_Lambda_Diversity_Grid();

  // Function that checks if there are interactions between groups in the matrix of betas
  bool Check_Interactions_Beta(arma::mat beta);
  // Function to returns a vector with ones corresponding to the betas that have interactions.
  arma::uvec Check_Interactions(arma::cube & betas);

  // Private function to create the folds
  arma::uvec Set_Diff(const arma::uvec & big, const arma::uvec & small);

  // Private function to compute the CV-MSPE over the folds
  double (*Compute_Deviance)(arma::mat x, arma::vec y,
                             arma::vec intercept, arma::mat betas);

  public:

  // Constructor - with data
  CV_Split_WEN(arma::mat & x, arma::vec & y,
               arma::uword & type,
               arma::uword & G, arma::uword & include_intercept,
               double & alpha_s, double & alpha_d,
               arma::uword & n_lambda_sparsity, arma::uword & n_lambda_diversity,
               double & tolerance, arma::uword & max_iter,
               arma::uword & n_folds,
               arma::uword & n_threads);

  // Functions to set new data
  void Set_X(arma::mat & x);
  void Set_Y(arma::vec & y);

  // Method to set alpha_s to new value and return current alpha_s
  void Set_Alpha_S(double alpha_s);
  double Get_Alpha_S();
  // Method to get the grid of lambda sparsity
  arma::vec Get_Lambda_Sparsity_Grid();
  // Method to get the grid of lambda diversity
  arma::vec Get_Lambda_Diversity_Grid();

  // Functions to set maximum number of iterations and tolerance
  void Set_Max_Iter(arma::uword & max_iter);
  void Set_Tolerance(double & tolerance);

  // Cross-validation - Sparsity
  arma::vec Get_CV_Error_Sparsity();
  // Cross-validation - Diversity
  arma::vec Get_CV_Error_Diversity();
  
  // Optimal penalty parameter - Sparsity
  double Get_lambda_sparsity_opt();
  // Optimal penalty parameter - Diversity
  double Get_lambda_diversity_opt();

  // Methods to return coefficients
  arma::cube Get_Coef();
  arma::mat Get_Intercept();

  // Optimal sparsity parameter
  arma::uword Get_Optimal_Index_Sparsity();
  // Optimal diversity parameter
  arma::uword Get_Optimal_Index_Diversity();

  // Computing the solutions over a grid for folds. Grid is either for the sparsity or the diverity (one of them is fixed)
  void Compute_CV_Grid(arma::uvec & sample_ind, arma::uvec & fold_ind,
                       bool & diversity_search);
  // Function for the initial CV Error with no diversity
  void Get_CV_Sparsity_Initial();
  // Coordinate descent algorithms for coefficients
  void Compute_CV_Betas();
  // Coordinate descent algorithms for coefficients (Full Diveristy)
  void Compute_CV_Betas_Full_Diversity();

  // Static functions for the deviance
  static double Linear_Deviance(arma::mat x, arma::vec y,
                                arma::vec intercept, arma::mat betas);
  static double Logistic_Deviance(arma::mat x, arma::vec y,
                                  arma::vec intercept, arma::mat betas);
  static double Gamma_Deviance(arma::mat x, arma::vec y,
                               arma::vec intercept, arma::mat betas);
  static double Poisson_Deviance(arma::mat x, arma::vec y,
                                 arma::vec intercept, arma::mat betas);

  // Virtual destructor
  ~CV_Split_WEN();
};

#endif // CV_WEN_hpp




