/*
* ===========================================================
* File Type: HPP
* File Name: WEN.hpp
* Package Name: SplitGLM
* 
  * Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/
  
#ifndef WEN_hpp
#define WEN_hpp

#include <RcppArmadillo.h>
  
#include "config.h"

class WEN{
  
  private: 
    
  // Variables supplied by user
  arma::mat x; 
  arma::vec y;
  arma::uword type;
  arma::uword include_intercept;
  double alpha;
  double lambda_sparsity;
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
  arma::uword beta_size; // Number of coefficients
  double intercept;
  double intercept_scaled;
  arma::vec betas;
  arma::vec betas_scaled;
  arma::vec expected_val;
  arma::vec weights;
  arma::vec residuals;
  arma::vec new_betas;
  double new_intercept;
  arma::vec xj_y; // Convenience vector for soft-thresholding
  double w_xj2; // Convenience variable for soft-thresholding
  const double EQUAL_TOLERANCE = 1e-5;
  
  // Function to initial the object characteristics
  void Initialize();
  // Functions for the computation of coefficients
  void Adjust_Expected_Weights();
  void (*Compute_Expected_Weights)(arma::mat & x, double & intercept, arma::vec & betas, 
                                   arma::vec & expected_val, arma::vec & weights);
  void Adjust_Residuals();
  double Soft(double z, double gamma);
  arma::uword Compare_Active_Set(arma::uvec & active_set);
  
  public:
     
  // Constructor - with data
  WEN(arma::mat x, arma::vec y,
      arma::uword & type, arma::uword & include_intercept, 
      double alpha, double lambda_sparsity,
      double tolerance, arma::uword max_iter);
    
  // Functions to set new data
  void Set_X(arma::mat & x);
  void Set_Y(arma::vec & y);
  
  // Functions to set maximum number of iterations and tolerance
  void Set_Max_Iter(arma::uword & max_iter);
  void Set_Tolerance(double & tolerance);
  
  // Functions for cycling over variables (CD iterations)
  void Cycle_Full_Set();
  void Cycle_Active_Set(arma::uvec & active_set);
  
  // Coordinate descent algorithms for coefficients
  void Compute_Coef();
  void Compute_Coef_Active();
  
  // Methods to return coefficients
  arma::vec Get_Coef();
  double Get_Intercept();
  arma::vec Get_Coef_Scaled();
  double Get_Intercept_Scaled();
  
  // Method to set alpha to new value and return current alpha
  void Set_Alpha(double alpha);
  double Get_Alpha();
  // Method to set lambda to new value and return current lambda
  void Set_Lambda_Sparsity(double lambda_sparsity);
  double Get_Lambda_Sparsity();
  
  // Functions to return expected values and weights
  arma::vec Get_Expected();
  arma::vec Get_Weights();
  
  // Function to get objective function value
  double Get_Objective_Value();
  
  // Function to scale back coefficients to original scale
  void Scale_Coefficients();
  void Scale_Intercept();
  
  // Static functions for expected values
  static void Linear_Update(arma::mat & x, double & intercept, arma::vec & betas, 
                            arma::vec & expected_val, arma::vec & weights);
  static void Linear_Update_Intercept(arma::mat & x, double & intercept, arma::vec & betas, 
                                      arma::vec & expected_val, arma::vec & weights);
  static void Logistic_Update(arma::mat & x, double & intercept, arma::vec & betas, 
                              arma::vec & expected_val, arma::vec & weights);
  static void Logistic_Update_Intercept(arma::mat & x, double & intercept, arma::vec & betas, 
                                        arma::vec & expected_val, arma::vec & weights);
  static void Gamma_Update(arma::mat & x, double & intercept, arma::vec & betas, 
                           arma::vec & expected_val, arma::vec & weights);
  static void Gamma_Update_Intercept(arma::mat & x, double & intercept, arma::vec & betas, 
                                     arma::vec & expected_val, arma::vec & weights);
  static void Poisson_Update(arma::mat & x, double & intercept, arma::vec & betas, 
                             arma::vec & expected_val, arma::vec & weights);
  static void Poisson_Update_Intercept(arma::mat & x, double & intercept, arma::vec & betas, 
                                       arma::vec & expected_val, arma::vec & weights);

  // Destructor
  ~WEN();
};

#endif // WEN_hpp




