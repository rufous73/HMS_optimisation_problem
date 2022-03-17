#pragma once

#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <utility>

using Function = double(*)(const boost::numeric::ublas::vector<double>&);
using Vec_Function = boost::numeric::ublas::vector<double>(*)(const boost::numeric::ublas::vector<double>&);


// Ñalculating the derivative of a function of one variable
// f - function pointer
// arg - the point at which the derivative is calculated
// step - step of the calculation method
double Derivative(double (*f)(const double&), const double& arg, const double& step);


// Calculating the gradient of a function
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::vector<double> Gradient(Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step);


// Calculation of the Jacobian of a system of functions
// f - an array of pointers to system of functions
// args - the point at which the jacobian is calculated
// step - the step of the calculation method
boost::numeric::ublas::matrix<double> Jacobian(Vec_Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step);


// Calculation of the Hessian of a function
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> Hessian(Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step);


// Numerical calculation of the integral: the rectangle method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Rectangual(const double& a, const double& b, const int& n, double (*f)(const double&));


// Numerical calculation of the integral: the trapecian method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Trapecian(const double& a, const double& b, const int& n, double (*f)(const double&));


// Numerical calculation of the integral: the Simpson method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Simpson(const double& a, const double& b, const int& n, double (*f)(const double&));


// Numerical calculation of the integral: the Simpson method for current task
// a - lower bound
// b - upper bound
// n - number of steps
// x1 - spatial coordinate of the Ox axis
// n1 - number of the summation elements
// f - function pointer
double Integral_Simpson_1(const double& a, const double& b, int n, const double& x1, int n1,
	double (*f)(const double&, const double&, int));


using c_double = std::complex<double>;
using C_Function = double(*)(const boost::numeric::ublas::vector<c_double>&);
using C_Vec_Function = boost::numeric::ublas::vector<double>(*)(const boost::numeric::ublas::vector<c_double>&);


// Calculating the complex gradient
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::vector<c_double> C_Gradient(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// The first item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_11(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// The second item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_12(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// The third item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_13(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// The fourth item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_14(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// Calculation of the holomorfic Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<c_double> C_Hessian_1(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// Calculation of the complex Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<c_double> C_Hessian_2(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);


// Calculation of the holomorfic and complex Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
std::pair<boost::numeric::ublas::matrix<c_double>, boost::numeric::ublas::matrix<c_double>> H1_H2(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step);
