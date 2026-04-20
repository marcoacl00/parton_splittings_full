#ifndef DERIVATIVES_HPP
#define DERIVATIVES_HPP

#include "Physis.hpp"

using namespace std;

struct InterpCoeffs {
    dcomplex f;
    dcomplex fk;
    dcomplex fl;
    dcomplex fkk;
    dcomplex fll;
    dcomplex flk;
};


void precompute_derivatives_3d(
    const Physis& sys,
    const std::vector<dcomplex>& fH0,
    vector<InterpCoeffs>& coeffs
);




#endif
