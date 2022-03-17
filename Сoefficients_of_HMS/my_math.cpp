#include "my_math.h"

// Ñalculating the derivative of a function of one variable
// f - function pointer
// arg - the point at which the derivative is calculated
// step - step of the calculation method
double Derivative(double (*f)(const double&), const double& arg, const double& step) {
	return (f(arg + step) - f(arg)) / step;
}


// Calculating the gradient of a function
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::vector<double> Gradient(Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step) {
	using namespace boost::numeric::ublas;

	int varCount = args.size();
	vector<double> gradient(varCount);
	for (int i = 0; i < varCount; i++) {
		gradient[i] = (f(args + step * unit_vector<double>(varCount, i)) - f(args)) / step;
	}

	return gradient;
}


// Calculation of the Jacobian of a system of functions
// f - an array of pointers to system of functions
// args - the point at which the jacobian is calculated
// step - the step of the calculation method
boost::numeric::ublas::matrix<double> Jacobian(Vec_Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step) {
	using namespace boost::numeric::ublas;

	vector<double> curFValue = f(args); // the value of the function at the current point
	int funcCount = curFValue.size();
	int varCount = args.size();

	vector<vector<double>> res(varCount);
	matrix<double> jacobian(funcCount, varCount);
	for (int i = 0; i < varCount; i++) {
		res[i] = (f(args + step * unit_vector<double>(varCount, i)) - curFValue) / step;
	}

	for (int i = 0; i < funcCount; i++) {
		for (int j = 0; j < varCount; j++) {
			jacobian(i, j) = res[j][i];
		}
	}

	return jacobian;
}


// Calculation of the Hessian of a function
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> Hessian(Function f,
	const boost::numeric::ublas::vector<double>& args, const double& step) {
	using namespace boost::numeric::ublas;

	int varCount = args.size();
	matrix<double> hessian(varCount, varCount);
	vector<double> fPlus(varCount);
	vector<double> fMinus(varCount);

	double curFValue = f(args); // the value of the function at the current point 
	for (int i = 0; i < varCount; i++) {
		// In the case of a derivative of one variable, we take the second central derivative as an approximation 
		fPlus[i] = f(args + step * unit_vector<double>(varCount, i));
		fMinus[i] = f(args - step * unit_vector<double>(varCount, i));

		hessian(i, i) = (fPlus[i] - 2 * curFValue + fMinus[i]) / (step * step);
	}

	for (int i = 0; i < varCount; i++) {
		// Since the Hessian is symmetric, we calculate only half of it
		for (int j = i + 1; j < varCount; j++) {
			double x = f(args + step * (unit_vector<double>(varCount, i) + unit_vector<double>(varCount, j)));
			hessian(i, j) = (x - fPlus[i] - fPlus[j] + curFValue) / (step * step);
		}
	}

	for (int i = 0; i < varCount; i++) {
		for (int j = 0; j < i; j++) {
			hessian(i, j) = hessian(j, i);
		}
	}

	return hessian;
}


// Numerical calculation of the integral: the rectangle method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Rectangual(const double& a, const double& b, const int& n, double (*f)(const double&)) {
	double step = (b - a) / n;
	std::vector<double> x(n + 1);
	std::vector<double> y(n + 1);
	double res = 0.0;

	for (size_t i = 0; i < x.size(); i++) {
		x[i] = a + i * step;
		y[i] = f(x[i]);
	}

	for (size_t i = 0; i < x.size() - 1; i++) {
		res += (x[i + 1] - x[i]) * y[i];
	}

	return res;
}


// Numerical calculation of the integral: the trapecian method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Trapecian(const double& a, const double& b, const int& n, double (*f)(const double&)) {
	double step = (b - a) / n;
	std::vector<double> x(n + 1);
	std::vector<double> y(n + 1);
	double res = 0.0;

	for (int i = 0; i <= n; i++) {
		x[i] = a + i * step;
		y[i] = f(x[i]);
	}

	for (size_t i = 0; i < x.size() - 1; i++) {
		res += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2;
	}

	return res;
}


// Numerical calculation of the integral: the Simpson method
// a - lower bound
// b - upper bound
// n - number of steps
// f - function pointer
double Integral_Simpson(const double& a, const double& b, const int& n, double (*f)(const double&)) {
	double step = (b - a) / n;
	std::vector<double> x(n + 1);
	std::vector<double> y(n + 1);
	double res = 0.0;

	for (int i = 0; i <= n; i++) {
		x[i] = a + i * step;
		y[i] = f(x[i]);
	}

	for (size_t i = 0; i < x.size() - 2; i += 2) {
		res += (x[i + 1] - x[i]) * (y[i] + 4 * y[i + 1] + y[i + 2]) / 3;
	}

	return res;
}


// Numerical calculation of the integral: the Simpson method for current task
// a - lower bound
// b - upper bound
// n - number of steps
// x1 - spatial coordinate of the Ox axis
// n1 - number of the summation elements
// f - function pointer
double Integral_Simpson_1(const double& a, const double& b, int n, const double& x1, int n1, 
	double (*f)(const double&, const double&, int)) {
	double step = (b - a) / n;
	std::vector<double> x(n + 1);
	std::vector<double> y(n + 1);
	double res = 0.0;

	for (int i = 0; i <= n; i++) {
		x[i] = a + i * step;
		y[i] = f(x[i], x1, n1);
	}

	for (size_t i = 0; i < x.size() - 2; i += 2) {
		res += (x[i + 1] - x[i]) * (y[i] + 4 * y[i + 1] + y[i + 2]) / 3;
	}

	return res;
}


// Calculating the complex gradient
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::vector<c_double> C_Gradient(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	int varCount = args.size();
	c_double c_step = step * 1.0i;
	vector<c_double> gradient(varCount);
	double F_value = f(args);
	double real_part, img_part; // real and imaginary parts of gradient 
	for (int i = 0; i < varCount; i++) {
		real_part = (f(args + step * unit_vector<double>(varCount, i)) - F_value) / step;
		img_part = (f(args + c_step * unit_vector<c_double>(varCount, i)) - F_value) / step;
		gradient[i] = 0.5 * (real_part + 1.0i * img_part);
	}

	return gradient;
}


// The first item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_11(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;

	int varCount = args.size();
	matrix<double> hessian(varCount, varCount);
	vector<double> fPlus(varCount);
	vector<double> fMinus(varCount);

	double curFValue = f(args); // the value of the function at the current point 
	std::cout << "Calculation of dioganal of H11\n";
	for (int i = 0; i < varCount; i++) {
		// In the case of a derivative of one variable, we take the second central derivative as an approximation 
		fPlus[i] = f(args + step * unit_vector<double>(varCount, i));
		fMinus[i] = f(args - step * unit_vector<double>(varCount, i));

		hessian(i, i) = (fPlus[i] - 2 * curFValue + fMinus[i]) / (step * step);
		//std::cout << i << ",";
	}

	std::cout << "Calculation of the upper triangle of H11\n";

	for (int i = 0; i < varCount; i++) {
		// Since the Hessian is symmetric, we calculate only half of it
		for (int j = i + 1; j < varCount; j++) {
			double x = f(args + step * (unit_vector<double>(varCount, i) + unit_vector<double>(varCount, j)));
			hessian(i, j) = (x - fPlus[i] - fPlus[j] + curFValue) / (step * step);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	std::cout << "Calculation of the lower triangle of H11\n";

	for (int i = 0; i < varCount; i++) {
		for (int j = 0; j < i; j++) {
			hessian(i, j) = hessian(j, i);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	//std::cout << std::endl;

	return hessian;
}


// The second item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_12(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	int varCount = args.size();
	matrix<double> hessian(varCount, varCount);
	vector<double> fPlus(varCount);
	vector<double> fMinus(varCount);

	c_double c_step = step * 1.0i;
	double curFValue = f(args); // the value of the function at the current point 
	std::cout << "Calculation of dioganal of H12\n";
	for (int i = 0; i < varCount; i++) {
		// In the case of a derivative of one variable, we take the second central derivative as an approximation 
		fPlus[i] = f(args + c_step * unit_vector<c_double>(varCount, i));
		fMinus[i] = f(args - c_step * unit_vector<c_double>(varCount, i));

		hessian(i, i) = (fPlus[i] - 2 * curFValue + fMinus[i]) / (step * step);

		//std::cout << i << ",";
	}

	std::cout << "Calculation of the upper triangle of H12\n";

	for (int i = 0; i < varCount; i++) {
		// Since the Hessian is symmetric, we calculate only half of it
		for (int j = i + 1; j < varCount; j++) {
			double x = f(args + c_step * (unit_vector<c_double>(varCount, i) + unit_vector<c_double>(varCount, j)));
			hessian(i, j) = (x - fPlus[i] - fPlus[j] + curFValue) / (step * step);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	std::cout << "Calculation of the lower triangle of H12\n";

	for (int i = 0; i < varCount; i++) {
		for (int j = 0; j < i; j++) {
			hessian(i, j) = hessian(j, i);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	//std::cout << std::endl;

	return hessian;
}


// The third item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_13(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	int varCount = args.size();
	matrix<double> hessian(varCount, varCount);
	vector<double> fPlus(varCount);
	vector<double> c_fPlus(varCount);

	c_double c_step = step * 1.0i;
	double curFValue = f(args);

	std::cout << "Calculation of fPlus & c_fPlus\n";

	for (int i = 0; i < varCount; i++) {
		fPlus[i] = f(args + step * unit_vector<double>(varCount, i));
		c_fPlus[i] = f(args + c_step * unit_vector<c_double>(varCount, i));
		//std::cout << i << ",";
	}

	std::cout << "Calculation of elements of H13\n";

	for (int i = 0; i < varCount; i++) {
		for (int j = 0; j < varCount; j++) {
			double x = f(args + step * unit_vector<double>(varCount, i) + c_step * unit_vector<c_double>(varCount, j));
			hessian(i, j) = (x - fPlus[i] - c_fPlus[j] + curFValue) / (step * step);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	//std::cout << std::endl;

	return hessian;
}


// The fourth item for calculating the complex and holomorphic hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<double> C_Hessian_14(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	int varCount = args.size();
	matrix<double> hessian(varCount, varCount);
	vector<double> fPlus(varCount);
	vector<double> c_fPlus(varCount);

	c_double c_step = step * 1.0i;
	double curFValue = f(args);
	std::cout << "Calculation of fPlus & c_fPlus\n";
	for (int i = 0; i < varCount; i++) {
		fPlus[i] = f(args + step * unit_vector<double>(varCount, i));
		c_fPlus[i] = f(args + c_step * unit_vector<c_double>(varCount, i));
		//std::cout << i << ",";
	}

	std::cout << "Calculation of elements of H14\n";

	for (int i = 0; i < varCount; i++) {
		for (int j = 0; j < varCount; j++) {
			double x = f(args + c_step * unit_vector<c_double>(varCount, i) + step * unit_vector<double>(varCount, j));
			hessian(i, j) = (x - fPlus[j] - c_fPlus[i] + curFValue) / (step * step);
			//std::cout << "(" << i << "," << j << "),";
		}
	}

	//std::cout << std::endl;

	return hessian;
}


// Calculation of the holomorfic Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<c_double> C_Hessian_1(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	matrix<double> H11 = C_Hessian_11(f, args, step);
	matrix<double> H12 = C_Hessian_12(f, args, step);
	matrix<double> H13 = C_Hessian_13(f, args, step);
	matrix<double> H14 = C_Hessian_14(f, args, step);

	matrix<c_double> img_part = H13 + H14;

	return 0.25 * (H11 - H12 - 1.0i * img_part);
}


// Calculation of the complex Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
boost::numeric::ublas::matrix<c_double> C_Hessian_2(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	matrix<double> H11 = C_Hessian_11(f, args, step);
	matrix<double> H12 = C_Hessian_12(f, args, step);
	matrix<double> H13 = C_Hessian_13(f, args, step);
	matrix<double> H14 = C_Hessian_14(f, args, step);

	matrix<c_double> img_part = H13 - H14;

	return 0.25 * (H11 + H12 + 1.0i * img_part);
}

// Calculation of the holomorfic and complex Hessian
// f - function pointer
// args - the point at which the derivative is calculated
// step - step of the calculation method
std::pair<boost::numeric::ublas::matrix<c_double>, boost::numeric::ublas::matrix<c_double>> H1_H2(C_Function f,
	const boost::numeric::ublas::vector<c_double>& args, const double& step) {
	using namespace boost::numeric::ublas;
	using namespace std::complex_literals;

	std::cout << "Calculation of H11\n";
	matrix<double> H11 = C_Hessian_11(f, args, step);
	std::cout << "Calculation of H12\n";
	matrix<double> H12 = C_Hessian_12(f, args, step);
	std::cout << "Calculation of H13\n";
	matrix<double> H13 = C_Hessian_13(f, args, step);
	std::cout << "Calculation of H14\n";
	matrix<double> H14 = C_Hessian_14(f, args, step);

	matrix<c_double> img_part1 = H13 + H14;
	matrix<c_double> img_part2 = H13 - H14;

	matrix<c_double> H1 = 0.25 * (H11 - H12 - 1.0i * img_part1);
	matrix<c_double> H2 = 0.25 * (H11 + H12 + 1.0i * img_part2);

	return std::make_pair(H1, H2);
}
