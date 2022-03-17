#include "CGM.h"


const double g_pi = 3.14159265358979323846; // number pi
const double g_c = 299792458; // m/s - speed of light
double g_frequency = 10.0e9; // Hz - central frequency
double g_omega = 2 * g_pi * g_frequency; // 1/s - circular frequency
double g_lambda = g_c / g_frequency; // m - central wavelength
double g_k = 2 * g_pi / g_lambda; // 1/m - wavenumber (free-space)
double g_kc = 2.1 * g_k; // 1/m - wavenumber that determine the center spatial frequency
double g_kw = 1.1 * g_k; // 1/m - wavenumber that determine the range of added spectrum
double g_L = g_lambda / 6.5; // m - distance between two consecutive basis-functions
double g_eta = 120 * g_pi; // Ohm - impedance of free-space
double g_Y = 1 / g_eta; // S - admittance


// Radius-vector
// x - spatial coordinate of the Ox axis
boost::numeric::ublas::vector<double> Radius_vector(const double& x) {
    using namespace boost::numeric::ublas;

    // To set the incident wave, the beginning of the Y coordinate axis was shifted to the point ds
    // In the rest of the work, the beginning of the Y coordinate is in the position of the metasurface
    double ds = g_lambda / 3;
    vector<double> rad_vector(2);
    rad_vector[0] = x;
    rad_vector[1] = ds;

    return rad_vector;
}


// Electric field of the incident wave
// x - spatial coordinate of the Ox axis
c_double Ex_inc(const double& x) {
    using namespace boost::numeric::ublas;

    vector<double> rad_vector = Radius_vector(x);
    vector<double> v_k = g_k * unit_vector<double>(2, 2); // direction vector (0, k)
    double Eo = 500.0 / norm_2(rad_vector); // magnitude of Ex_inc
    double t = g_lambda / (3 * g_c); // the time it took for the wave to reach the metasurface t = ds/3 = lambda/3c
    double theta = -(g_omega * t - inner_prod(v_k, rad_vector)); // the phase of field
    
    return std::polar(Eo, theta);
}


// Magnetic field of the incident wave 
// x - spatial coordinate of the Ox axis
c_double Hz_inc(const double& x) {
    using namespace boost::numeric::ublas;

    vector<double> rad_vector = Radius_vector(x);
    vector<double> v_k = g_k * unit_vector<double>(2, 2); // direction vector (0, k)
    double Ho = 500.0 / (g_eta * norm_2(rad_vector)); // magnitude of Hz_inc
    double t = g_lambda / (3 * g_c); // the time it took for the wave to reach the metasurface t = ds/3 = lambda/3c
    double theta = -(g_omega * t - inner_prod(v_k, rad_vector)); // the phase of field

    return std::polar(Ho, theta);
}


// Integrand expression
// kx - integration variable
// x - spatial coordinate of the Ox axis
// n - number of the summation elements
double Integrand(const double& kx, const double& x, int n) {
    return sqrt(kx * kx - g_k * g_k) * cos(kx * (x - n * g_L));
}


// Electric field of the surface wave
// x - spatial coordinate of the Ox axis
// An - weight variable
c_double Ex_sw(const double& x, const c_double& An) {
    using namespace std::complex_literals;

    int N = 19; // 2N + 1 - numbers of unit cells
    double a = g_kc - g_kw; // lower bound of integral
    double b = g_kc + g_kw; // upper bound of integral
    c_double Ex = 0.0;
    for (int n = -N; n <= N; n++) {
        Ex += ((-1.0i * g_eta * An) / (2 * g_kw * g_k)) * Integral_Simpson_1(a, b, 100, x, n, Integrand);
    }

    return Ex;
}


// sinc-function
// x - spatial coordinate of the Ox axis
// n - number of the summation elements
double Sinc(const double& x, int n) {
    if (x == (n * g_L)) {
        return 1.0;
    }
    else {
        return sin(g_kw * (x - (n * g_L))) / (x - (n * g_L));
    }
}


// Magnetic field of the surfase wave 
// x - spatial coordinate of the Ox axis
// An - weight variable
c_double Hz_sw(const double& x, const c_double& An) {
    int N = 19; // 2N + 1 - numbers of unit cells
    c_double Hz = 0.0;
    for (int n = -N; n <= N; n++) {
        Hz += An * Sinc(x, n) * cos(g_kc * (x - n * g_L));
    }

    return Hz;
}


// Vecor of the power mismatch
// An - vector of weight variables
boost::numeric::ublas::vector<double> G(const boost::numeric::ublas::vector<c_double>& An) {
    using namespace boost::numeric::ublas;

    int N = 19; // 2N + 1 - numbers of unit cells
    std::vector<double> x(2 * N + 1); // array of spatial coordinates x
    vector<double> G_res(2 * N + 1);
    double P_out = 700; // W/m^2 - output power
    int n = -N;
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = n * g_L;
        G_res[i] = std::real((Ex_inc(x[i]) + Ex_sw(x[i], An[i])) *
            std::conj(Hz_inc(x[i]) + Hz_sw(x[i], An[i]))) - P_out;
        n++;
    }

    return G_res;
}


// The function whose minimum needs to be found (G^2)
// An - vector of weight variables
double GTG(const boost::numeric::ublas::vector<c_double>& An) {
    using namespace boost::numeric::ublas;

    vector<double> G_vec = G(An);
    return 0.5 * inner_prod(G_vec, G_vec);
}



int main()
{
    using namespace boost::numeric::ublas;
    using namespace std::complex_literals;

    int N = 19; // 2N + 1 - numbers of unit cells
    std::cout << "Calculation of An with " << 2 * N + 1 << " unit cells:" << std::endl << std::endl;
    std::cout.setf(std::ios_base::scientific);
    std::cout.precision(20);
    vector<c_double> An = C_Gradient_descent(2 * N + 1, GTG, 1.7e-8);
    std::cout << "Value of An is " << An << std::endl << std::endl;
    std::cout << "Value of G with calculated An is " << G(An) << std::endl;

    return 0;
}