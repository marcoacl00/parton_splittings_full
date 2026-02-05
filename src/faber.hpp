#ifndef FABER_HPP
#define FABER_HPP

#include <iostream>
#include <complex>
#include <vector>
#include "Physis_J.hpp"
#include "Physis.hpp"

using dcomplex = std::complex<double>; 

void faber_params3D_ker(const Physis_J& sys, double &lambda_F, dcomplex &gamma_0, dcomplex &gamma_1);
void faber_params3D_full(const Physis& sys, double &lambda_F, dcomplex &gamma_0, dcomplex &gamma_1);

dcomplex coeff(int m, double dt1, const dcomplex& gamma_0, const dcomplex& gamma_1, double lambF);

std::vector<dcomplex> faber_expandJ(const Physis_J& sys, double ht, const dcomplex& gamma0, const dcomplex& gamma1, double one_lamb,
                                    const std::vector<dcomplex>& coeff_array, std::vector <double> taylor_coeff0, std::vector <double> taylor_coeff1, std::vector <double> taylor_coeff2);

std::vector<dcomplex> faber_expand_full(const Physis& sis_f, double ht, const dcomplex& gamma0_f, const dcomplex& gamma1_f, double one_lamb_1, const std::vector<dcomplex>& coeff_array_f);

#endif



