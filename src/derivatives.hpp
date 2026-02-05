#ifndef DERIVATIVES_HPP
#define DERIVATIVES_HPP

#include "Physis.hpp"

using namespace std;

static void precompute_derivatives(
    const Physis& sys,
    const std::vector<dcomplex>& fH0,
    std::vector<dcomplex>& fk,
    std::vector<dcomplex>& fkk,
    std::vector<dcomplex>& fl,
    std::vector<dcomplex>& fll,
    std::vector<dcomplex>& fp,
    std::vector<dcomplex>& fpp,
    std::vector<dcomplex>& flk,
    std::vector<dcomplex>& fpl,
    std::vector<dcomplex>& fpk
);


#endif
