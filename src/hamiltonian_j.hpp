#ifndef HAMILTONIANJ_HPP
#define HAMILTONIANJ_HPP

#include "Physis_J.hpp"

using namespace std;


static void precompute_derivatives(
    const Physis_J& sys,
    const std::vector<dcomplex>& fH0,
    std::vector<dcomplex>& fx,
    std::vector<dcomplex>& fxx);

double VHTL(double q, double mu);
double VHO(double q, double mu);
double VYUK(double q, double mu);


double compute_f_r_pc(
    int r,
    double p,
    double z,
    double pm,
    double pM,
    int Np,        // radial grid points
    int Ntheta,     // angular grid points
    double mu,
    int mode,
    string vertex);

double factorial(int r);

vector<dcomplex> Hamiltonian_J(const Physis_J& sys, 
    const vector<dcomplex>& fH0,
    vector<double> taylor_coeffs_0,
    vector<double> taylor_coeffs_1,
    vector<double> taylor_coeffs_2);

#endif