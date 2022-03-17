#pragma once

#include "my_math.h"


// CGM for a quadratic functional
// A - a symmetric positive definite matrix
// b - a vector of values
// eps - calculation accuracy
boost::numeric::ublas::vector<double> CGM(const boost::numeric::ublas::matrix<double>& A, 
	const boost::numeric::ublas::vector<double>& b, const double& eps);


// One-dimensional optimization - the golden section method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
template <typename T>
double GoldenSection(double(*F)(const boost::numeric::ublas::vector<T>&),
					 const boost::numeric::ublas::vector<T>& args,
					 const boost::numeric::ublas::vector<T>& p);


// One-dimensional optimization - the Newton-Raphson method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
double Newton_Raphson(Function F, const boost::numeric::ublas::vector<double>& args, 
	const boost::numeric::ublas::vector<double>& p, const double& eps);


// One-dimensional optimization - the Secant method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
double Secant(Function F, const boost::numeric::ublas::vector<double>& args,
	const boost::numeric::ublas::vector<double>& p, const double& eps);


// Gradient descent for a nonlinear system of functions
// spaceSize - the dimension of the function
// f - system of functions
// eps - calculation accuracy
boost::numeric::ublas::vector<double> Gradient_descent(int spaceSize, Vec_Function f, const double& eps);


// Nonlinear CGM - Fletcher-Rieves Method
// spaceSize - the dimension of the function
// F - pointer to the minimized function
// eps - calculation accuracy
boost::numeric::ublas::vector<double> FletcherRievesMethod(int spaceSize, Function F, const double& eps);


// Nonlinear CGM - Polak-Ribiere Method
// spaceSize - the dimension of the function
// F - pointer to the minimized function
// eps - calculation accuracy
boost::numeric::ublas::vector<double> PolakRibiereMethod(int spaceSize, Function F, const double& eps);


// Gradient descent for a complex functions
// spaceSize - the dimension of the function
// f - system of functions
// eps - calculation accuracy
boost::numeric::ublas::vector<c_double> C_Gradient_descent(int spaceSize, C_Function f, double eps);