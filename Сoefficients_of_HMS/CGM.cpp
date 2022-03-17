#include "CGM.h"


// CGM for a quadratic functional
// A - a symmetric positive definite matrix
// b - a vector of values
// eps - calculation accuracy
boost::numeric::ublas::vector<double> CGM(const boost::numeric::ublas::matrix<double>& A, 
	const boost::numeric::ublas::vector<double>& b, const double& eps) {
	using namespace boost::numeric::ublas;

	vector<double> x = zero_vector<double>(A.size1()); // initial approximation
	vector<double> r = b - prec_prod(A, x); // resiaul vector
	vector<double> p = r; // direction vector
	double rSquare = inner_prod(r, r);
	int numIter = 0;
	while (rSquare > eps) {
		numIter++;
		vector<double> temp = prec_prod(A, p);
		double alpha = rSquare / inner_prod(temp, p);
		x = x + alpha * p;

		vector<double> rNew = r - alpha * temp;
		double rNewSquare = inner_prod(rNew, rNew);
		double beta = rNewSquare / rSquare;
		r = rNew;
		rSquare = rNewSquare;
		p = r + beta * p;
	}

	std::cout << "Number of iterations: " << numIter << std::endl;
	return x;
}


// One-dimensional optimization - the golden section method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
template <typename T>
double GoldenSection(double(*F)(const boost::numeric::ublas::vector<T>&), 
					 const boost::numeric::ublas::vector<T>& args,
					 const boost::numeric::ublas::vector<T>& p) {
	const double EPS = 1e-8;
	double a = 0;
	double b = 1e5;
	double x0 = a + 0.5 * (3 - sqrt(5.0)) * (b - a);
	double x1 = b - 0.5 * (3 - sqrt(5.0)) * (b - a);

	while (abs(b - a) > EPS) {
		if (F(args - x0 * p) < F(args - x1 * p)) {
			b = x1;
			x1 = x0;
			x0 = a + 0.5 * (3 - sqrt(5.0)) * (b - a);
		}
		else {
			a = x0;
			x0 = x1;
			x1 = b - 0.5 * (3 - sqrt(5.0)) * (b - a);
		}
	}

	return (a + b) / 2;
}


// One-dimensional optimization - the Newton-Raphson method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
double Newton_Raphson(Function F, const boost::numeric::ublas::vector<double>& args,
	const boost::numeric::ublas::vector<double>& p, const double& eps) {
	using namespace boost::numeric::ublas;

	double alpha; // result of optimization
	double numenator = inner_prod(Gradient(F, args, eps), p);

	matrix<double> transposed_p(1, args.size());
	for (size_t j = 0; j < args.size(); j++) {
		transposed_p(0, j) = p[j];
	}

	matrix<double> prod_of_p_H = prec_prod(transposed_p, Hessian(F, args, eps));
	vector<double> v_prod_of_p_H(args.size());
	for (size_t i = 0; i < args.size(); i++) {
		v_prod_of_p_H[i] = prod_of_p_H(0, i);
	}
	double denumenator = inner_prod(v_prod_of_p_H, p);
	alpha = -numenator / denumenator;

	return alpha;
}


// One-dimensional optimization - the Secant method
// f - function pointer
// args - the point at which optimization is performed
// p - direction vector
double Secant(Function F, const boost::numeric::ublas::vector<double>& args,
	const boost::numeric::ublas::vector<double>& p, const double& eps) {
	using namespace boost::numeric::ublas;

	double alpha; // result of optimization
	double numenator = inner_prod(Gradient(F, args, eps), p);
	double denumenator = inner_prod(Gradient(F, args + eps * p, eps), p) - numenator;
	alpha = -eps * (numenator / denumenator);

	return alpha;
}


// Gradient descent for a nonlinear system of functions
// spaceSize - the dimension of the function
// f - system of functions
// eps - calculation accuracy
boost::numeric::ublas::vector<double> Gradient_descent(int spaceSize, Vec_Function f, const double& eps) {
	using namespace boost::numeric::ublas;

	//std::cout << "Gradient descent calculation started\n";

	vector<double> x(spaceSize, 0.0); // initial approximation
	matrix<double> jacobi = Jacobian(f, x, 1e-16); // Jacobian of the system
	matrix<double> t_jacobi = trans(jacobi); // transposed Jacobian
	vector<double> G = f(x);
	vector<double> p = prec_prod(t_jacobi, G); // direction vector
	// Size of step calculation: eta = (v * F) / (v * v), v = J * p
	vector<double> v = prec_prod(jacobi, p);
	double eta = inner_prod(v, G) / inner_prod(v, v);

	//int numIter = 0;
	do {
		//std::cout << "Step " << numIter << ": x = " << x << ", eta = " << eta << std::endl;

		//numIter++;
		x = x - eta * p;
		G = f(x);
		jacobi = Jacobian(f, x, 1e-16);
		t_jacobi = trans(jacobi);
		p = prec_prod(t_jacobi, G);
		v = prec_prod(jacobi, p);
		eta = inner_prod(v, G) / inner_prod(v, v);
	} while (norm_2(p) > eps);

	//std::cout << "Gradient descent calculation over\n";

	return x;
}


// Nonlinear CGM - Fletcher-Rieves Method
// spaceSize - the dimension of the function
// F - pointer to the minimized function
// eps - calculation accuracy
boost::numeric::ublas::vector<double> FletcherRievesMethod(int spaceSize, Function F, const double& eps) {
	using namespace boost::numeric::ublas;

	vector<double> x = zero_vector<double>(spaceSize); // initial approximation
	vector<double> p = -Gradient(F, x, eps); // direction vector
	double gradSquare = inner_prod(p, p);

	int numIter = 0;
	do {
		numIter++;
		double alpha, beta, newGradSquare;
		vector<double> newGrad;

		// Calculation of the minimum F(x + alpha * p) using the one-dimensional optimization method
		alpha = GoldenSection(F, x, p);
		x = x + alpha * p;

		newGrad = -Gradient(F, x, eps);
		newGradSquare = inner_prod(newGrad, newGrad);

		if (numIter % (spaceSize) == 0) {
			beta = 0; // Update
		}
		else {
			beta = newGradSquare / gradSquare; // Fletcher-Rieves Method
		}

		p = newGrad + beta * p;
		gradSquare = newGradSquare;
	} while (gradSquare > eps);

	return x;
}


// Nonlinear CGM - Polak-Ribiere Method
// spaceSize - the dimension of the function
// F - pointer to the minimized function
// eps - calculation accuracy
boost::numeric::ublas::vector<double> PolakRibiereMethod(int spaceSize, Function F, const double& eps) {
	using namespace boost::numeric::ublas;

	vector<double> x = zero_vector<double>(spaceSize); // initial approximation
	vector<double> grad = -Gradient(F, x, eps);
	vector<double> p = grad; // direction vector
	double gradSquare = inner_prod(p, p);

	int numIter = 0;
	do {
		numIter++;
		double alpha, beta, newGradSquare;
		vector<double> newGrad;

		// Calculation of the minimum F(x + alpha * p) using the one-dimensional optimization method
		alpha = GoldenSection(F, x, p);
		x = x + alpha * p;

		newGrad = -Gradient(F, x, eps);
		newGradSquare = inner_prod(newGrad, newGrad);

		if (numIter % (spaceSize) == 0) {
			beta = 0; // Update
		}
		else {
			beta = (newGradSquare - inner_prod(newGrad, grad)) / gradSquare; // Polak-Ribiere Method
			if (beta < 0) {
				beta = 0; // Update
			}
		}

		p = newGrad + beta * p;
		grad = newGrad;
		gradSquare = newGradSquare;
	} while (gradSquare > eps);

	return x;
}


// Gradient descent for a complex functions
// spaceSize - the dimension of the function
// f - system of functions
// eps - calculation accuracy
boost::numeric::ublas::vector<c_double> C_Gradient_descent(int spaceSize, C_Function f, double eps) {
	using namespace boost::numeric::ublas;

	std::cout << "Complex Gradient descent optimization started\n\n";

	vector<c_double> x(spaceSize, 0.0); // initial approximation
	vector<c_double> s = C_Gradient(f, x, eps);
	vector<c_double> p = -s; // direction vector
	

	double residual = std::abs(norm_1(s));
	int numIter = 0;

	std::cout << "Step " << numIter << ":\n" << "x = " << x << '\n' << "residual = " << residual << "\nf = " << f(x) << "\n\n";

	do {
		numIter++;
		std::cout << "Calculation of H1 & H2:\n";
		std::pair<matrix<c_double>, matrix<c_double>> H1H2 = H1_H2(f, x, eps);
		std::cout << "H1 & H2 calculated\n\n";

		// for calculation alpa
		matrix<c_double> t_p(1, spaceSize);
		for (int j = 0; j < spaceSize; j++) {
			t_p(0, j) = p[j];
		}

		matrix<c_double> pH1 = prec_prod(t_p, H1H2.first);
		vector<c_double> v_pH1(spaceSize);
		for (int j = 0; j < spaceSize; j++) {
			v_pH1[j] = pH1(0, j);
		}

		matrix<c_double> pH2 = prec_prod(t_p, H1H2.second);
		vector<c_double> v_pH2(spaceSize);
		for (int j = 0; j < spaceSize; j++) {
			v_pH2[j] = pH2(0, j);
		}

		c_double alpha, beta;
		vector<c_double> new_s;
		alpha = inner_prod(conj(s), s) / std::real(inner_prod(v_pH2, conj(p)) + inner_prod(v_pH1, p));
		x = x + alpha * p;
		/*vector<c_double> H2p = prec_prod(conj(H1H2.second), p);
		vector<c_double> H1p = prec_prod(conj(H1H2.first), conj(p));
		new_s = s + alpha * (H2p + H1p);*/

		new_s = C_Gradient(f, x, eps);

		if (numIter % (spaceSize) == 0) {
			beta = 0.0; //Update
		}
		else {
			beta = inner_prod(conj(new_s), new_s) / inner_prod(conj(s), s);
		}
		
		p = -new_s + beta * p;

		s = new_s;
		residual = std::abs(norm_1(s));

		std::cout << "Step " << numIter << ":\n" << "x = " << x << '\n' << "residual = " << residual 
			<< "\nf = " << f(x) << "\nalpha = " << alpha << "\nbeta = " << beta << "\n\n";

		if (numIter == 65) {
			eps = 1.0e-10;
		}

		if (numIter == 98) {
			eps = 1.0e-12;
		}

		if (numIter == 134) {
			eps = 1.0e-14;
		}

		if (numIter == 160) {
			//eps = 1.0e-15;
			break;
		}

	} while (residual > 1.0e-3);

	std::cout << "Complex Gradient descent optimization over\n\n";

	return x;
}
