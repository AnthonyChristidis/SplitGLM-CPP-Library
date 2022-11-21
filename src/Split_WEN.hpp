/*
* ===========================================================
* File Type: HPP
* File Name: Split_WEN.hpp
* Package Name: SplitGLM
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef Split_WEN_hpp
#define Split_WEN_hpp

#include <RcppArmadillo.h>

#include "config.h"

class Split_WEN{

  private:

  // Variables supplied by user
  arma::mat x;
  arma::vec y;
  arma::uword type;
  arma::uword G;
  arma::uword include_intercept;
  double alpha_s;
  double alpha_d;
  double lambda_sparsity;
  double lambda_diversity;
  double tolerance;
  arma::uword max_iter;
  // Variables created inside class
  arma::rowvec mu_x;
  arma::rowvec sd_x;
  arma::mat x_std;
  arma::mat x_std_2;
  double mu_y;
  arma::uword n; // Number of samples
  arma::uword p; // Number of variables (does not include intercept term)
  arma::vec intercept;
  arma::vec intercept_scaled;
  arma::mat betas;
  arma::mat betas_scaled;
  arma::mat expected_val;
  arma::mat weights;
  arma::mat residuals;
  arma::mat new_betas;
  arma::vec new_intercept;
  arma::vec xj_y; // Convenience vector for soft-thresholding
  double w_xj2; // Convenience variable for soft-thresholding
  arma::mat active_set;
  const double EQUAL_TOLERANCE = 1e-5;

  // Function to initial the object characteristics
  void Initialize();
  // Functions for the computation of coefficients
  void Adjust_Expected_Weights(arma::uword & group);
  void (*Compute_Expected_Weights)(arma::uword & group,
                                   arma::mat & x,
                                   arma::vec & intercept, arma::mat & betas,
                                   arma::mat & expected_val, arma::mat & weights);
  void Adjust_Residuals(arma::uword & group);
  double Soft(double z, double gamma);
  arma::vec Beta_Weights_Abs(arma::uword & group);
  arma::vec Beta_Weights_Sq(arma::uword & group);

  // Function to copmute the diversity penalty
  double Sparsity_Penalty();
  double Diversity_Penalty();
  
  // Function to compare the active sets for convergence
  arma::uword Compare_Active_Set();
  
  public:

  // Constructor - with data
  Split_WEN(arma::mat x, arma::vec y,
                             arma::uword & type,
                             arma::uword & G, arma::uword & include_intercept,
                             double alpha_s, double alpha_d, 
                             double lambda_sparsity, double lambda_diversity,
                             double tolerance, arma::uword max_iter);

  // Functions to set new data
  void Set_X(arma::mat & x);
  void Set_Y(arma::vec & y);

  // Functions to set maximum number of iterations and tolerance
  void Set_Max_Iter(arma::uword & max_iter);
  void Set_Tolerance(double & tolerance);

  // Functions for cycling over variables (CD iterations)
  void Cycle_Full_Set(arma::uword & group);
  // Functions for cycling over variables (CD iterations) - Active Set
  void Cycle_Active_Set(arma::uword & group);

  // Coordinate descent algorithms for coefficients
  void Compute_Coef();
  void Compute_Coef_Active();

  // Methods to return coefficients
  arma::mat Get_Coef();
  arma::vec Get_Intercept();
  arma::mat Get_Coef_Scaled();
  arma::vec Get_Intercept_Scaled();

  // Method to set alpha_s to new value and return current alpha_s
  void Set_Alpha_S(double alpha_s);
  double Get_Alpha_S();
  // Method to set lambda_sparsity to new value and return current lambda_sparsity
  void Set_Lambda_Sparsity(double lambda_sparsity);
  double Get_Lambda_Sparsity();
  // Method to set lambda_diversity to new value and return current lambda_diversity
  void Set_Lambda_Diversity(double lambda_diversity);
  double Get_Lambda_Diversity();

  // Functions to return expected values and weights
  arma::mat Get_Expected();
  arma::mat Get_Weights();

  // Function to get objective function value
  double Get_Objective_Value();

  // Function to scale back coefficients to original scale
  void Scale_Coefficients();
  void Scale_Intercept();
 
  // Static functions for expected values
  static void Linear_Update(arma::uword & group,
                            arma::mat & x,
                            arma::vec & intercept, arma::mat & betas,
                            arma::mat & expected_val, arma::mat & weights);
  static void Linear_Update_Intercept(arma::uword & group,
                                      arma::mat & x,
                                      arma::vec & intercept, arma::mat & betas,
                                      arma::mat & expected_val, arma::mat & weights);
  static void Logistic_Update(arma::uword & group,
                              arma::mat & x,
                              arma::vec & intercept, arma::mat & betas,
                              arma::mat & expected_val, arma::mat & weights);
  static void Logistic_Update_Intercept(arma::uword & group,
                                        arma::mat & x,
                                        arma::vec & intercept, arma::mat & betas,
                                        arma::mat & expected_val, arma::mat & weights);
  static void Gamma_Update(arma::uword & group,
                           arma::mat & x,
                           arma::vec & intercept, arma::mat & betas,
                           arma::mat & expected_val, arma::mat & weights);
  static void Gamma_Update_Intercept(arma::uword & group,
                                     arma::mat & x,
                                     arma::vec & intercept, arma::mat & betas,
                                     arma::mat & expected_val, arma::mat & weights);
  static void Poisson_Update(arma::uword & group,
                             arma::mat & x,
                             arma::vec & intercept, arma::mat & betas,
                             arma::mat & expected_val, arma::mat & weights);
  static void Poisson_Update_Intercept(arma::uword & group,
                                       arma::mat & x,
                                       arma::vec & intercept, arma::mat & betas,
                                       arma::mat & expected_val, arma::mat & weights);

  // Destructor
  ~Split_WEN();
};

#endif // WEN_hpp




