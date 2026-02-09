#include <complex>
#include <vector>
#include <cmath>
#include "Physis_J.hpp"
#include "Physis.hpp"
#include "hamiltonian.hpp"
#include "hamiltonian_j.hpp"
#include "faber.hpp"
#include "bessel-library.hpp"

dcomplex I(0.0, 1.0);

// Faber expansion coefficients (mth)
dcomplex coeff(int m, double dt1, 
                const dcomplex& gamma_0, const dcomplex& gamma_1, double lambF)
{
    dcomplex sqrt_g1 = sqrt(gamma_1);
    dcomplex base = -I / sqrt_g1;
    dcomplex sqrt_term = pow(base, m);

    dcomplex jv_arg_c = 2.0 * lambF * dt1 * sqrt_g1;

    dcomplex jvArg = 2.0 * lambF * dt1 * sqrt_g1;
    dcomplex jv = bessel::cyl_j(m, jvArg, false, false);
   
    dcomplex exp_arg = exp(-I * lambF * dt1 * gamma_0);

    return sqrt_term * exp_arg * jv;
}



void faber_params3D_ker(const Physis_J& sys, double &lambda_F, dcomplex &gamma_0, dcomplex &gamma_1)
{
    double Lp = sys.Lp();
    double mu = sys.mu();
    int Np = sys.Np();
    double omega = sys.omega();
    double delta_p = Lp / (Np - 1);
    double qtilde = sys.qtilde();
    std::cout << "delta_p: " << delta_p / 5.067 << " GeV \n";

    double lam_re_max = Lp*Lp / (2.0 * omega);
    double lam_re_min = 0.0;
    double lam_im_max = 0.25*qtilde/(delta_p*delta_p) ;
    double lam_im_min = -0.25*qtilde/(delta_p*delta_p) ;

    double c = (lam_re_max - lam_re_min)/2.0;
    double l = (lam_im_max - lam_im_min)/2.0;

    lambda_F = pow( pow(l, 2.0/3.0) + pow(c, 2.0/3.0), 3.0/2.0 ) / 2.0;
    double one_lamb = 1.0 / lambda_F;

    double csc = c / lambda_F;
    double lsc = l / lambda_F;

    gamma_0 = dcomplex((lam_re_max + lam_re_min) * 0.5 * one_lamb, 
                        (lam_im_min + lam_im_max) * 0.5 * one_lamb);
    // replicate python expression (approx)
    gamma_1 = dcomplex(0.25 * ((pow(csc,2.0/3.0) + pow(lsc,2.0/3.0)) 
                            * (pow(csc,4.0/3.0) - pow(lsc,4.0/3.0))), 0.0);

}

void faber_params3D_full(const Physis& sys, double &lambda_F, dcomplex &gamma_0, dcomplex &gamma_1)
{
    double Lk = sys.Lk();
    double Ll = sys.Ll();
    double mu = sys.mu();
    int Nl = sys.Nl();
    int Nk = sys.Nk();
    double omega = sys.omega();
    double delta_k = Lk / (Nk - 1);
    double delta_l = Ll / (Nl - 1);
    double qtilde = sys.qtilde();
    

    double lam_re_max = Lk*Ll / omega;
    double lam_re_min = -Lk*Ll / omega;
    double lam_im_max = 0.0;
    double lam_im_min = - 4.0 * qtilde * 1.0 / 8.0 /(delta_k*delta_k) - qtilde/(delta_l*delta_l);

    double c = (lam_re_max - lam_re_min)/2.0;
    double l = (lam_im_max - lam_im_min)/2.0;

    lambda_F = pow( pow(l, 2.0/3.0) + pow(c, 2.0/3.0), 3.0/2.0 ) / 2.0;
    double one_lamb = 1.0 / lambda_F;

    double csc = c / lambda_F;
    double lsc = l / lambda_F;

    gamma_0 = dcomplex((lam_re_max + lam_re_min) * 0.5 * one_lamb, 
                        (lam_im_min + lam_im_max) * 0.5 * one_lamb);
    // replicate python expression (approx)
    gamma_1 = dcomplex(0.25 * ((pow(csc,2.0/3.0) + pow(lsc,2.0/3.0)) 
                            * (pow(csc,4.0/3.0) - pow(lsc,4.0/3.0))), 0.0);

}


// faber exoansion function 
vector<dcomplex> faber_expandJ(const Physis_J& sys, double ht, 
                                const dcomplex& gamma0, const dcomplex& gamma1, 
                                double one_lamb,
                                const vector<dcomplex>& coeff_array, 
                                vector<double> taylor_coeff0, 
                                vector<double> taylor_coeff1, 
                                vector<double> taylor_coeff2)
{
    if (coeff_array.size() < 3) {
        throw std::runtime_error("faber_expand3D: coeff_array.size() must be >= 3");
    }

    // Initialize fH_0, fH_1, fH_2
    vector<dcomplex> fH_0 = sys.Fsol();
    const size_t n = fH_0.size();
    
    vector<dcomplex> fH_1 = Hamiltonian_J(sys, fH_0, taylor_coeff0, taylor_coeff1, taylor_coeff2);
    
    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        fH_1[k] = fH_1[k] * one_lamb - gamma0 * fH_0[k];
    }
    
    vector<dcomplex> fH_2 = Hamiltonian_J(sys, fH_1, taylor_coeff0, taylor_coeff1, taylor_coeff2);
    
    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        fH_2[k] = fH_2[k] * one_lamb - gamma0 * fH_1[k] - 2.0 * gamma1 * fH_0[k];
    }
    
    // Initialize Uf_est 
    vector<dcomplex> Uf_est(n);
    
    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        Uf_est[k] = coeff_array[0] * fH_0[k] + 
                    coeff_array[1] * fH_1[k] + 
                    coeff_array[2] * fH_2[k];
    }
    
    // Main iteration loop
    for (size_t k = 3; k < coeff_array.size(); ++k) {
        // Rotate: fH_0 <- fH_1 <- fH_2 <- new
        fH_0.swap(fH_1);
        fH_1.swap(fH_2);
        
        fH_2 = Hamiltonian_J(sys, fH_1, taylor_coeff0, taylor_coeff1, taylor_coeff2);
        
        const dcomplex coeff_k = coeff_array[k];
        const dcomplex neg_gamma0 = -gamma0;
        const dcomplex neg_gamma1 = -gamma1;
        
        // Merge all operations into single parallel region
        //#pragma omp parallel for schedule(static) if(n > 1000)
        for (size_t j = 0; j < n; ++j) {
            // Apply transformations to fH_2
            fH_2[j] = fH_2[j] * one_lamb + neg_gamma0 * fH_1[j] + neg_gamma1 * fH_0[j];
            
            // Accumulate into Uf_est
            Uf_est[j] += coeff_k * fH_2[j];
        }
    }
    
    return Uf_est;
}

vector<dcomplex> faber_expand_full(const Physis& sis_f, double ht, 
                                const dcomplex& gamma0_f, const dcomplex& gamma1_f, 
                                double one_lamb_1, 
                                const vector<dcomplex>& coeff_array_f)
{
    if (coeff_array_f.size() < 3) {
        throw std::runtime_error("faber_expand_full: coeff_array_f.size() must be >= 3");
    }

    // Initialize fH_0, fH_1, fH_2
    vector<dcomplex> fH_0 = sis_f.Fsol();
    const size_t n = fH_0.size();

    vector<dcomplex> fH_1 = apply_hamiltonian(sis_f, fH_0);

    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        fH_1[k] = fH_1[k] * one_lamb_1 - gamma0_f * fH_0[k];
    }
    
    vector<dcomplex> fH_2 = apply_hamiltonian(sis_f, fH_1);
    
    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        fH_2[k] = fH_2[k] * one_lamb_1 - gamma0_f * fH_1[k] - 2.0 * gamma1_f * fH_0[k];
    }
    
    // Initialize Uf_est 
    vector<dcomplex> Uf_est(n);
    
    //#pragma omp parallel for schedule(static) if(n > 1000)
    for (size_t k = 0; k < n; ++k) {
        Uf_est[k] = coeff_array_f[0] * fH_0[k] + 
                    coeff_array_f[1] * fH_1[k] + 
                    coeff_array_f[2] * fH_2[k];
    }
    
    // Main iteration loop
    for (size_t k = 3; k < coeff_array_f.size(); ++k) {
        // Rotate: fH_0 <- fH_1 <- fH_2 <- new
        fH_0.swap(fH_1);
        fH_1.swap(fH_2);
        fH_2 = apply_hamiltonian(sis_f, fH_1);
        const dcomplex coeff_k = coeff_array_f[k];
        const dcomplex neg_gamma0 = -gamma0_f;
        const dcomplex neg_gamma1 = -gamma1_f;  
        // Merge all operations into single parallel region
        //#pragma omp parallel for schedule(static) if(n > 1000)
        for (size_t j = 0; j < n; ++j) {
            // Apply transformations to fH_2
            fH_2[j] = fH_2[j] * one_lamb_1 + neg_gamma0 * fH_1[j] + neg_gamma1 * fH_0[j];
            // Accumulate into Uf_est
            Uf_est[j] += coeff_k * fH_2[j];
        }
    }
    return Uf_est;
}