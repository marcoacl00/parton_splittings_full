#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP

#include "Physis.hpp"
#include "hamiltonian_j.hpp"
#include "derivatives.hpp"


using namespace std;


/*double compute_f_r_pc(
    int r,
    double p,
    double z,
    double pm,
    double pM,
    int Np,        // radial grid points
    int Ntheta,     // angular grid points
    double mu,
    int mode,
    string vertex);*/


vector<dcomplex> Hamiltonian_qqbar(const Physis& sys, 
    const vector<dcomplex>& fH0);

vector<dcomplex> Hamiltonian_qqbar_GPU(const Physis& sys, 
    const vector<dcomplex>& fH0);

vector<dcomplex> apply_hamiltonian(const Physis& sys, const vector<dcomplex>& fH0);

inline std::vector<dcomplex> apply_hamiltonian(const Physis& sys, const std::vector<dcomplex>& fH0) {

    //return Hamiltonian_qqbar_GPU(sys, fH0);
    return Hamiltonian_qqbar(sys, fH0);
}


#endif