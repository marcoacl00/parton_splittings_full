#include "hamiltonian.hpp"
#include "spline.h"

using namespace std;
using dcomplex = complex<double>;

const dcomplex I(0.0, 1.0);
//const double sqrt_e = 1.648721270700128;
const double e = 2.718281828459045;
const double PI = 3.141592653589793;


double VHTL_eff(double q, double mu)
{
    // yukawa potential (extra q factor due to polar coordinates jacobian) q / pow((q*q + mu*mu), 2)
    // extra 4 factor comes from simplifications of the convolution
    // htl 1 / (q*(q*q + (inv_sqrt_e * mu)*(inv_sqrt_e * mu)))
    return  1.0 / (4.0*q*(4.0*q*q + (e * mu*mu))); //q / pow((q*q + mu*mu), 2); 
}

double VHO_eff(double q, double mu)
{   double eps = mu;
    double exp = std::exp(-q*q / (2 * eps*eps));

    double fac = PI / 4.0 * 1.0 / (2.0 * PI * pow(eps, 4)) * (q*q  / (eps*eps)  - 2.0);


    return exp * fac;
}

double VYUK_eff(double q, double mu)
{   double eps = mu;
    // yukawa potential (extra q factor due to polar coordinates jacobian) q / pow((q*q + mu*mu), 2)
    // htl 1 / (q*(q*q + (inv_sqrt_e * mu)*(inv_sqrt_e * mu)))
 
    return  q / pow((4.0*q*q + mu*mu), 2);
}

struct GaussLegendre {
    vector<double> nodes;
    vector<double> weights;
   
    
GaussLegendre(int n) {
    // Initialize with precomputed values
    compute_gauss_legendre(n, nodes, weights);
}
    
private:
    void compute_gauss_legendre(int n, vector<double>& x, vector<double>& w) {
        x.resize(n);
        w.resize(n);
        
        // Compute roots of Legendre polynomial using Newton's method
        for (int i = 0; i < (n + 1) / 2; ++i) {
            // Initial guess
            double z = cos(M_PI * (i + 0.75) / (n + 0.5));
            double z1, pp;
            
            // Newton iteration
            do {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) {
                    double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
                }
                // Derivative
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                z1 = z;
                z = z1 - p1 / pp;
            } while (std::abs(z - z1) > 1e-15);
            
            x[i] = -z;
            x[n - 1 - i] = z;
            w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
            w[n - 1 - i] = w[i];
        }
    }
};


// Global precomputed Gauss-Legendre quadrature points (initialize once)
static const GaussLegendre GL_RADIAL(10);  



vector<dcomplex> Hamiltonian_qqbar(const Physis& sys, const vector<dcomplex>& fH0){

    // extract all relevant parameters from sys once
    int Nk = sys.Nk();
    int Nl = sys.Nl();
    int Npsi = sys.NPsi();
    double Lk = sys.Lk();
    double Ll = sys.Ll();
    double qtilde = sys.qtilde();
    double omega = sys.omega();
    double mu = sys.mu();
    double z = sys.z();
    vector<double> K_array = sys.K();
    vector<double> L_array = sys.L();
    vector<double> psi_array = sys.Psi();
    vector<double> cos_psi_array = sys.Cos_Psi();
    bool is_large_Nc = (sys.Ncmode() == "LNc");

    int Nsig = 2;

    // define output vector, fill with zeros
    vector<dcomplex> HF(fH0.size(), dcomplex(0.0, 0.0));

    double delta_k = Lk / (Nk - 1);
    double delta_l = Ll / (Nl - 1);
    double delta_psi = PI / (Npsi - 1);


    double CF = (is_large_Nc) ? 1.5 : 4.0/3.0;
    double CA = 3.0;

    // integration  
    double pmin = sys.pmin(); 
    double pmax = sys.pmax();

    // Precompute grid parameters for interpolation
    const double psi_min = psi_array.front();
    const double psi_max = psi_array.back();
    const double k_min = K_array.front();
    const double k_max = K_array.back();
    const double l_min = L_array.front();
    const double l_max = L_array.back();
    
    const double inv_dpsi = 1.0 / delta_psi;
    const double inv_dk = 1.0 / delta_k;
    const double inv_dl = 1.0 / delta_l;


    // ---- Gauss-Legendre setup --- 
    const int n_radial = GL_RADIAL.nodes.size();
    const int n_angular = 12; // can be adjusted for accuracy, 8 is a

    
    // map Gauss-Legendre nodes from [-1,1] to integration domains
    // For radial: [pmin, pmax]
    double sp = mu;
    double split =  10.0 * mu;

    vector<double> p1_nodes, p1_weights; // region [pmin, split]
    vector<double> p2_nodes, p2_weights; // region [split, pmax]
    p1_nodes.resize(n_radial);
    p1_weights.resize(n_radial);
    p2_nodes.resize(n_radial);
    p2_weights.resize(n_radial);
    vector<double> theta_nodes(n_angular), theta_weights(n_angular);
    vector<double> cos_theta(n_angular), sin_theta(n_angular);

    // Map separately for two radial subdomains. If split is outside [pmin,pmax],
    // one of the regions becomes the full interval.
    double a1 = 0.1*mu;
    double b1 = split; 
    double a2 = split;
    double b2 = 12.0*mu;

    double mid1 = 0.5 * (b1 + a1);
    double half1 = 0.5 * (b1 - a1);
    double mid2 = 0.5 * (b2 + a2);
    double half2 = 0.5 * (b2 - a2);

    for (int i = 0; i < n_radial; ++i) {
        // region 1: [a1, b1]
        p1_nodes[i] = mid1 + half1 * GL_RADIAL.nodes[i];
        p1_weights[i] = half1 * GL_RADIAL.weights[i];

        //cout << "p1: " << p1_nodes[i] << " w1: " << p1_weights[i] << endl;

        // region 2: [a2, b2]
        p2_nodes[i] = mid2 + half2 * GL_RADIAL.nodes[i];
        p2_weights[i] = half2 * GL_RADIAL.weights[i];

        // cout << "p1: " << p1_nodes[i] << " w1: " << p1_weights[i] << " | p2: " << p2_nodes[i] << " w2: " << p2_weights[i] << endl;
    }

    //compute chebyshev nodes for angular integration 
    for (int j = 0; j < n_angular; ++j) {
        theta_nodes[j] = M_PI * (2*j+1)/(2*n_angular); // Chebyshev nodes in [0, pi]
        cos_theta[j] = std::cos(theta_nodes[j]);
        sin_theta[j] = std::sin(theta_nodes[j]);
        //cout << "theta: " << theta_nodes[j] << " cos: " << cos_theta[j] << " sin: " << sin_theta[j] << endl;
    }

    std::vector<double> theta_reverse(32);
    std::vector<double> cos_theta_reverse(32);

    for(int i=0; i<32; i++){
        double dth = M_PI / double(32);
        double theta = M_PI - i * dth;
        theta_reverse[i]= theta;
        cos_theta_reverse[i]= std::cos(theta);
    }
    
    // this is much more efficient than evaluating acos 
    tk::spline acos_spline(cos_theta_reverse, theta_reverse);


    // some other precomputed constants
    double z2 = z * z;
    double one_z = 1.0 - z;
    double one_z_2 = one_z * one_z;
    double two_z_minus_1 = 2.0 * z - 1.0;
    double two_z_minus_1_2 = two_z_minus_1 * two_z_minus_1;

    // extract potential mode
    int mode = sys.mode();

    // lambda to choose potential based on mode
    // 0 = yukawa, 1 = HTL, 2 = HO
    auto V = [&mode](double p, double mu) -> double {
        if (mode == 0) {
            return VYUK_eff(p, mu);
        } else if (mode == 1) {
            return VHTL_eff(p, mu);
        }
            else{return VHO_eff(p, mu);}
        };

    // precompute derivatives
    std::vector<dcomplex> fk, fkk, fl, fll, flk, fp, fpp, fpk, fpl;
    precompute_derivatives_3d(sys, fH0, fk, fkk, fl, fll, fp, fpp, flk, fpl, fpk);



    // sampler with 2nd order Taylor expansion around nearest grid point, with hard domain checks
    auto get_fval = [&](int sig, double psi, double k, double l, int ip_, int ik_, int il_) -> dcomplex {
        // --- clamp psi to domain ---
        psi = std::clamp(psi, psi_min, psi_max);
        k   = std::clamp(k, k_min, k_max);
        l   = std::clamp(l, l_min, l_max);

        // --- angular interpolation (linear) ---
        // Map psi to index in psi_array
        // For simplicity, assume psi_array is sorted ascending
        ip_ = std::clamp(static_cast<int>((psi - psi_min) * inv_dpsi), 0, Npsi - 2);
        double t_psi = (psi - psi_array[ip_]) / (psi_array[ip_ + 1] - psi_array[ip_]);

        // --- k,l indices ---
        ik_ = std::clamp(static_cast<int>((k - k_min) * inv_dk), 0, Nk - 2);
        il_ = std::clamp(static_cast<int>((l - l_min) * inv_dl), 0, Nl - 2);

        // --- load Taylor expansion coefficients at psi_array[ip_] ---
        dcomplex f000 = fH0[sys.idx(sig, ip_, il_, ik_)];
        dcomplex f_k   = fk[sys.idx(sig, ip_, il_, ik_)];
        dcomplex f_l   = fl[sys.idx(sig, ip_, il_, ik_)];
        dcomplex f_kk  = fkk[sys.idx(sig, ip_, il_, ik_)];
        dcomplex f_ll  = fll[sys.idx(sig, ip_, il_, ik_)];
        dcomplex f_kl  = flk[sys.idx(sig, ip_, il_, ik_)];

        // --- offsets ---
        double t_k = k - K_array[ik_];
        double t_l = l - L_array[il_];

        // --- Taylor expansion in k,l only ---
        dcomplex fval_kl = f000
            + f_k * t_k
            + f_l * t_l
            + 0.5 * f_kk * t_k * t_k
            + 0.5 * f_ll * t_l * t_l
            + f_kl  * t_k * t_l;

        // --- linear interpolation in psi ---
        // Take next psi grid point
        dcomplex fval_kl_next = fH0[sys.idx(sig, ip_ + 1, il_, ik_)]
            + fk[sys.idx(sig, ip_ + 1, il_, ik_)] * t_k
            + fl[sys.idx(sig, ip_ + 1, il_, ik_)] * t_l
            + 0.5 * fkk[sys.idx(sig, ip_ + 1, il_, ik_)] * t_k * t_k
            + 0.5 * fll[sys.idx(sig, ip_ + 1, il_, ik_)] * t_l * t_l
            + flk[sys.idx(sig, ip_ + 1, il_, ik_)] * t_k * t_l;

        // Linear interpolation in ψ
        dcomplex fval = (1.0 - t_psi) * fval_kl + t_psi * fval_kl_next;

        return fval;
    };




    // string vertex = sys.vertex(); <- this is not necessary since the function is already for qqbar

    dcomplex prefac = - 4.0 * dcomplex(0, 1) * qtilde / (2.0 * PI); //temporarily with the 4 factor from the jacobian

    if (!is_large_Nc) {
        throw invalid_argument("Nc mode not implemented yet in Hamiltonian_qqbar");
    }

    // ============ sig = 0 ============
    #pragma omp parallel for collapse(3)
    for (int ip = 0; ip < Npsi; ++ip) {
        for (int il = 0; il < Nl; ++il) {
            for (int ik = 0; ik < Nk; ++ik) {

                double psi = psi_array[ip];
                double cos_psi = cos_psi_array[ip];
                double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi * cos_psi));
                double k = K_array[ik];
                double k2 = k * k;
                double l = L_array[il];
                double l2 = l * l;
                double kl = k * l;
                auto current_idx = sys.idx(0, ip, il, ik);

                dcomplex f_ = fH0[current_idx];

                HF[current_idx] = (k * l) / omega * cos_psi * f_;

                dcomplex sum_integral(0.0, 0.0);

                auto integrand_sig0 = [&](double p, double cos_th, double sin_th) {
                    double p2 = p * p;
                    double kp = k * p;
                    double lp = l * p;
                    //double Vp = p * V(2.0 * p, mu);
                    //cout << Vp << endl;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    // Compute distances
                    double Rp_m_k = std::sqrt(k2 + p2 - 2*kp*cos_th);
                    double Rp_p_k = std::sqrt(k2 + p2 + 2*kp*cos_th);
                    double Rp_m_l =std::sqrt(l2 + p2 - 2*lp*cos_theta_minus_psi);
                    //double Rp_m_l = (il > 0) ? l - cos_theta_minus_psi * p + p2 * (1 - cos_theta_minus_psi*cos_theta_minus_psi)/(2*l): p;
                    
                    // cout << "Rp_m_k: " << Rp_m_k << " Rp_p_k: " << Rp_p_k << " Rp_m_l: " << Rp_m_l << endl;

                    // ----- Compute angles ------
                    
                    // p-k, l-p samples
                    double numer_pmk_lmp = (p2 - kp * cos_th - lp * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_lmp = numer_pmk_lmp / (Rp_m_k * Rp_m_l);
                    cos_val_pmk_lmp = std::clamp(cos_val_pmk_lmp, -1.0 + 1e-14, 1.0 - 1e-14);
                    double sin_val_pmk_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_pmk_lmp * cos_val_pmk_lmp));

                    double angle_pmk_lmp = atan2(sin_val_pmk_lmp, cos_val_pmk_lmp); // use atan2 for better numerical stability

                    // k+p, l-p samples
                    double numer_ppk_lmp = -(p2 + kp * cos_th - lp * cos_theta_minus_psi) + kl * cos_psi;
                    double cos_val_ppk_lmp = numer_ppk_lmp / (Rp_p_k * Rp_m_l);
                    cos_val_ppk_lmp = std::clamp(cos_val_ppk_lmp, -1.0 + 1e-14, 1.0 - 1e-14);
                    double sin_val_ppk_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_ppk_lmp * cos_val_ppk_lmp));
                    double angle_ppk_lmp = atan2(sin_val_ppk_lmp, cos_val_ppk_lmp); // use atan2 for better numerical stability

                    // Sample with domain checks
                    bool in_domain_1 = (Rp_m_k >= k_min && Rp_m_k <= k_max + delta_k && Rp_m_l >= l_min && Rp_m_l <= l_max + delta_l);
                    bool in_domain_2 = (Rp_p_k >= k_min && Rp_p_k <= k_max + delta_k && Rp_m_l >= l_min && Rp_m_l <= l_max + delta_l);

                    // (only if both samples are valid)
                    dcomplex S0_contrib(0.0, 0.0);

                    // double t_psi_1 = (angle_ppk_lmp - psi_array[ip]);
                    // double t_psi_2 = (angle_pmk_lmp - psi_array[ip]);
                    // double t_k_1 = (Rp_p_k - K_array[ik]) ;
                    // double t_k_2 = (Rp_m_k - K_array[ik]);
                    // double t_l_1 = (Rp_m_l - L_array[il]);

                    dcomplex f0_kmp_lmp = get_fval(0, angle_pmk_lmp, Rp_m_k, Rp_m_l, ip, ik, il);
                    dcomplex f0_kpp_lmp = get_fval(0, angle_ppk_lmp, Rp_p_k, Rp_m_l, ip, ik, il);
                    S0_contrib += 2.0 * CF * (2.0 * f_ - f0_kmp_lmp - f0_kpp_lmp);

                    // NaN checks
                    // dcomplex f_k = fp[current_idx];
                    // dcomplex f_l = fl[current_idx];
                    // dcomplex f_kk = fkk[current_idx];
                    // dcomplex f_ll = fll[current_idx];
                    // dcomplex f_psipsi = fpp[current_idx];

                    //S0_contrib += -2.0 * CF * (f_k * (t_k_1 + t_k_2) + 2.0 * f_l * t_l_1 + f_kk * (t_k_1 * t_k_1 + t_k_2 + t_k_2) + 2.0 * f_ll * t_l_1 * t_l_1 + f_psipsi * (t_psi_1 * t_psi_1 + t_psi_2 * t_psi_2));
                    //} 

                    return S0_contrib;
                };

              

                //Integrate over region 1: [a1,b1]
                // Simpson 1/3 over region 1: [a1,b1]
                if (half1 > 0.0) {
                    // choose an even number of subintervals for Simpson (at least 2)
                    int Nsim = std::max(2, 2 * n_radial);                    

                    // double h =  lim  / static_cast<double>(Nsim-1);

                    dcomplex radial_acc(0.0, 0.0);
                    double h = (b1 - a1) / static_cast<double>(Nsim);
                    for (int ii = 0; ii <= n_radial; ++ii) {

                        double p = a1 + ii * (b1 - a1) / Nsim; // uniform sampling for Simpson's rule
                        //double p = p1_nodes[ii];

                        int coeff = (ii == 0 || ii == Nsim) ? 1 : ((ii % 2 == 1) ? 4 : 2);

                        int Ntheta = n_angular; // number of Chebyshev points
                        dcomplex integrated_in_theta = 0.0;

                        // loop over Chebyshev nodes once
                        for (int j = 0; j < Nsim; ++j) {
                            double xj = cos_theta[j]; // Chebyshev node in [-1,1]
                            double sqrt_term = sin_theta[j]; // sqrt(1 - xj^2)

                            // ---- first half: theta in [0, pi] ----
                            double cos_th1 = xj;
                            double sin_th1 = sqrt_term;
                            integrated_in_theta += integrand_sig0(p, cos_th1, sin_th1);

                            // ---- second half: theta in [pi, 2pi] ----
                            double cos_th2 = -xj;
                            double sin_th2 = -sqrt_term;
                            integrated_in_theta += integrand_sig0(p, cos_th2, sin_th2);
                        }

                        // multiply by pi/Ntheta (Chebyshev weight)
                        integrated_in_theta *= M_PI / Ntheta;

                        double Vp = V(2.0 * p, sp);
                        //radial_acc = p1_weights[ii] * Vp * integrated_in_theta ;
                        radial_acc += h/3.0 *  p * Vp * integrated_in_theta * double(coeff); // include jacobian p and potential V(p)
                    }

                    sum_integral += radial_acc; 

                    // nan check
                    //cout << "ip: " << ip << " il: " << il << " ik: " << ik << " radial_acc: " << sum_integral << endl;
                    
                }

                // Integrate over region 2: [a2,b2]
                if (false) {
                    int Nsim = std::max(2, n_radial); // choose an even number of subintervals for Simpson (at least 2)
                    dcomplex radial_acc(0.0, 0.0);
                    double h = (b2 - a2) / static_cast<double>(Nsim);

                    for (int ii = 0; ii <= Nsim; ++ii) {
                        double p = a2 + ii * (b2 - a2) / Nsim; // uniform sampling for Simpson's rule
                        //double p = p2_nodes[ii];
                        int coeff = (ii == 0 || ii == Nsim) ? 1 : ((ii % 2 == 1) ? 4 : 2);
                        double Vp = p * V(2.0 * p, mu);
                        dcomplex integrated_in_theta = 0.0;
                        for (int j = 0; j < n_angular; ++j) {
                            double xj = cos_theta[j]; // Chebyshev node in [-1,1]
                            double sqrt_term = sin_theta[j]; // sqrt(1 - xj^2)

                            // ---- first half: theta in [0, pi] ----
                            double cos_th1 = xj;
                            double sin_th1 = sqrt_term;
                            integrated_in_theta += integrand_sig0(p, cos_th1, sin_th1);

                            // ---- second half: theta in [pi, 2pi] ----
                            double cos_th2 = -xj;
                            double sin_th2 = -sqrt_term;
                            integrated_in_theta += integrand_sig0(p, cos_th2, sin_th2);

                        }
                        sum_integral += integrated_in_theta * Vp * p * h / 3.0 * double(coeff); // include jacobian p and potential V(p)
                    }
                }

                HF[current_idx] += prefac * sum_integral;

                /*dcomplex laplacian_l;
                if (il == 0){
                    laplacian_l = 2.0 * fll[current_idx]; // at l=0, the function is psi-independent, so the laplacian reduces to the second derivative in l
                }
                else{
                    laplacian_l = fll[current_idx] + (1.0/l) * fl[current_idx] + (1.0)/(l*l) * fpp[current_idx];
                }




                // //compute derivatives on the fly here
                dcomplex laplacian_k = fkk[current_idx] + (1.0/k) * fk[current_idx] ;

                HF[sys.idx(0, ip, il, ik)] += I * qtilde / 4.0 * 0.5 * CF * (laplacian_k + laplacian_l);*/


            }}
        }

            // Enforce psi-independence at l = 0 by averaging
            for (int ik = 0; ik < Nk; ++ik) {
                // compute ψ-average at l = 0
                dcomplex avg = 0.0;
                for (int ip = 0; ip < Npsi; ++ip) {
                    avg += HF[sys.idx(0, ip, 0, ik)];
                }
                avg /= static_cast<double>(Npsi);

                // enforce ψ-invariance
                for (int ip = 0; ip < Npsi; ++ip) {
                    HF[sys.idx(0, ip, 0, ik)] = avg;
                }
            }
        

        // ============ sig = 1 ============
        #pragma omp parallel for collapse(3)
        for (int ip = 0; ip < Npsi; ++ip) {
            for (int il = 0; il < Nl; ++il) {
                for (int ik = 0; ik < Nk; ++ik) {
                    double psi = psi_array[ip];
                    double cos_psi = cos_psi_array[ip];
                    double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi * cos_psi));
                    double k = K_array[ik];
                    double k2 = k * k;
                    double l = L_array[il];
                    double l2 = l * l;
                    double kl = k * l;

                    const dcomplex f_ = fH0[sys.idx(1, ip, il, ik)];
                    
                    // BOUNDARY CONDITION: At l=0, set kinetic term to 0 to ensure psi-independence
            
                    HF[sys.idx(1, ip, il, ik)] = k * l * cos_psi / omega * f_;

                    dcomplex sum_integral(0.0, 0.0);

                    // Helper lambda for M10 and M11 contributions
                    auto integrand_sig1 = [&](double p, double cos_th, double sin_th) {
                        double p2 = p * p;
                        double pl = p * l;
                        double pk = p * k;
                        double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                        // M10 contribution
                        double Rp_m_k = std::sqrt(k2 + p2 - 2*pk*cos_th);
                        double Rp_p_k = std::sqrt(k2 + p2 + 2*pk*cos_th);
                        double Rp_m_l = (il == 0) ? p : std::sqrt(l2 + p2 - 2*pl*cos_theta_minus_psi);
                        //double R_l_m_2z_p = std::sqrt(l2 + (2*z-1)*(2*z-1)*p2 - 2.0 * pl*(2*z-1)*cos_theta_minus_psi);
                        //double R_l_m_2_1z_p =std::sqrt(l2 + (2*z-1)*(2*z-1)*p2 + 2.0 * pl*(2*z-1)*cos_theta_minus_psi);
                        double R_k_m_2z_p = std::sqrt(k2 + (2*z-1)*(2*z-1)*p2 - 2.0 * pk*(2*z-1)*cos_th);
                        double R_k_m_2_1z_p = std::sqrt(k2 + (2*z-1)*(2*z-1)*p2 + 2.0 * pk*(2*z-1)*cos_th);

                        // Angle calculations
                        
                        //k-p, l-p (sigma_0)
                        double numer_pmk_lmp = (p2 - pk * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
                        double cos_val_pmk_lmp = std::clamp(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-12), -1.0, 1.0);
                        //double angle_pmk_lmp = acos_spline(cos_val_pmk_lmp);
                        double sin_val_pmk_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_pmk_lmp * cos_val_pmk_lmp));
                        double angle_pmk_lmp = atan2(sin_val_pmk_lmp, cos_val_pmk_lmp); // use atan2 for better numerical stability

                        //k+p, l-p (sigma_0)
                        double numer_ppk_lmp = -(p2 + pk * cos_th - pl * cos_theta_minus_psi) + kl * cos_psi;
                        double cos_val_ppk_lmp = std::clamp(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-12), -1.0, 1.0);
                        //double angle_ppk_lmp = acos_spline(cos_val_ppk_lmp);
                        double sin_val_ppk_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_ppk_lmp * cos_val_ppk_lmp));
                        double angle_ppk_lmp = atan2(sin_val_ppk_lmp, cos_val_ppk_lmp); // use atan2 for better numerical stability

                        // k - (2z-1)p, l - p (sigma_zs)
                        // numerator = -k p Cos[\[Theta]] + p (-1 + 2 z) (p - l Cos[\[Theta] - \[Psi]]) + k l Cos[\[Psi]]
                        double numer_k_m_2z_p_lmp = (-pk * cos_th + p * two_z_minus_1 * (p - pl * cos_theta_minus_psi) + kl * cos_psi);
                        double cos_val_k_m_2z_p_lmp = std::clamp(numer_k_m_2z_p_lmp / (R_k_m_2z_p * Rp_m_l + 1e-12), -1.0, 1.0);
                        //double angle_k_m_2z_p_lmp = acos_spline(cos_val_k_m_2z_p_lmp);
                        double sin_val_k_m_2z_p_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_k_m_2z_p_lmp * cos_val_k_m_2z_p_lmp));
                        double angle_k_m_2z_p_lmp = atan2(sin_val_k_m_2z_p_lmp, cos_val_k_m_2z_p_lmp); // use atan2 for better numerical stability

                        // k + (2z - 1)p, l - p (sigma_zs)
                        //numerator = -k p Cos[\[Theta]] - p (-1 + 2 z) (p - l Cos[\[Theta] - \[Psi]]) + k l Cos[\[Psi]]

                        double numer_k_m_2_1z_p_lmp = (-pk * cos_th - p * two_z_minus_1 * (p - pl * cos_theta_minus_psi) + kl * cos_psi);
                        double cos_val_k_m_2_1z_p_lmp = std::clamp(numer_k_m_2_1z_p_lmp / (R_k_m_2_1z_p * Rp_m_l + 1e-12), -1.0, 1.0);
                        //double angle_k_m_2_1z_p_lmp = acos_spline(cos_val_k_m_2_1z_p_lmp);
                        double sin_val_k_m_2_1z_p_lmp = std::sqrt(std::max(0.0, 1.0 - cos_val_k_m_2_1z_p_lmp * cos_val_k_m_2_1z_p_lmp));
                        double angle_k_m_2_1z_p_lmp = atan2(sin_val_k_m_2_1z_p_lmp, cos_val_k_m_2_1z_p_lmp); // use atan2 for better numerical stability

                        
                        // Sample values with domain checks
                        // bool d1 = (Rp_m_k >= k_min && Rp_m_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
                        // bool d2 = (Rp_p_k >= k_min && Rp_p_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
                        // bool d3 = (R_k_m_2z_p >= k_min && R_k_m_2z_p <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
                        // bool d4 = (R_k_m_2_1z_p >= k_min && R_k_m_2_1z_p <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);

                        dcomplex M10_contrib(0.0, 0.0);
                        dcomplex f0_kmp_lmp = get_fval(0, angle_pmk_lmp, Rp_m_k, Rp_m_l, ip, ik, il);
                        dcomplex f0_kpp_lmp = get_fval(0, angle_ppk_lmp, Rp_p_k, Rp_m_l, ip, ik, il);
                        dcomplex f0_pmk_l2zp = get_fval(0, angle_k_m_2z_p_lmp, R_k_m_2z_p, Rp_m_l, ip, ik, il);
                        dcomplex f0_pmk_l2_1zp = get_fval(0, angle_k_m_2_1z_p_lmp, R_k_m_2_1z_p, Rp_m_l, ip, ik, il);


                        M10_contrib += CA * (f0_pmk_l2zp + f0_pmk_l2_1zp - f0_kmp_lmp - f0_kpp_lmp); // minus sign correct


                        // M11 contribution
                        double R_k_zp = std::sqrt(k2 + 4*z2*p2 - 4*pk*z*cos_th);
                        double R_k_1zp = std::sqrt(k2 + 4*one_z_2*p2 - 4*pk*one_z*cos_th);

                        double numer_k_zp_l = (k * cos_psi - 2*z*p*cos_theta_minus_psi);
                        double cos_val_k_zp_l = std::clamp(numer_k_zp_l / (R_k_zp), -1.0, 1.0);

                        double angle_k_zp_l = (il > 0) ? acos_spline(cos_val_k_zp_l) : 0.0;

                        double numer_k_1zp_l = (k * cos_psi - 2*one_z*p*cos_theta_minus_psi);
                        double cos_val_k_1zp_l = std::clamp(numer_k_1zp_l / (R_k_1zp), -1.0, 1.0);

                        double angle_k_1zp_l = (il > 0) ? acos_spline(cos_val_k_1zp_l) : 0.0;

                        //bool d5 = (R_k_zp >= k_min && R_k_zp <= k_max);
                        //bool d6 = (R_k_1zp >= k_min && R_k_1zp <= k_max);

                        dcomplex M11_contrib(0.0, 0.0);
                        //if (d5 && d6) {
                        dcomplex f1_k_zp_l = get_fval(1, angle_k_zp_l, R_k_zp, l, ip, ik, il);
                        dcomplex f1_k_1zp_l = get_fval(1, angle_k_1zp_l, R_k_1zp, l, ip, ik, il);
                        M11_contrib += 2.0 * CF * (2.0 * f_ - f1_k_zp_l -  f1_k_1zp_l);
                        //}

                        /*if (d6) {
                            dcomplex f1_k_1zp_l = get_fval(1, angle_k_1zp_l, R_k_1zp, l);
                            M11_contrib += 2.0 * CF * (f_ - f1_k_1zp_l);
                        }*/

                        return M10_contrib + M11_contrib;
                    };

        

                if (half1 > 0.0) {
                    // choose an even number of subintervals for Simpson (at least 2)
                    int Nsim = std::max(2, 3 * n_radial);                    

                    // double h =  lim  / static_cast<double>(Nsim-1);

                    dcomplex radial_acc(0.0, 0.0);
                    double h = (b1 - a1) / static_cast<double>(Nsim);
                    for (int ii = 0; ii <= Nsim; ++ii) {

                        double p = a1 + ii * h; // uniform sampling for Simpson's rule
                        
                        int coeff = (ii == 0 || ii == Nsim) ? 1 : ((ii % 2 == 1) ? 4 : 2);

                        int Ntheta = n_angular; // number of Chebyshev points
                        dcomplex integrated_in_theta = 0.0;

                        // loop over Chebyshev nodes once
                        for (int j = 0; j < Ntheta; ++j) {
                            double xj = cos_theta[j]; // Chebyshev node in [-1,1]
                            double sqrt_term = sin_theta[j]; // sqrt(1 - xj^2)

                            // ---- first half: theta in [0, pi] ----
                            double cos_th1 = xj;
                            double sin_th1 = sqrt_term;
                            integrated_in_theta += integrand_sig1(p, cos_th1, sin_th1);

                            // ---- second half: theta in [pi, 2pi] ----
                            double cos_th2 = -xj;
                            double sin_th2 = -sqrt_term;
                            integrated_in_theta += integrand_sig1(p, cos_th2, sin_th2);
                        }

                        // multiply by pi/Ntheta (Chebyshev weight)
                        integrated_in_theta *= M_PI / Ntheta;

                        double Vp = V(2.0 * p, sp);
                        radial_acc += h/3.0 *  p * Vp * integrated_in_theta * double(coeff); // include jacobian p and potential V(p)
                    }

                    sum_integral += radial_acc; 

                

            }

            if (false) {
                int Nsim = std::max(2, n_radial); // choose an even number of subintervals for Simpson (at least 2)
                dcomplex radial_acc(0.0, 0.0);
                double h = (b2 - a2) / static_cast<double>(Nsim);
                for (int ii = 0; ii <= Nsim; ++ii) {
                    double p = a2 + ii * h; // uniform sampling for Simpson's rule
                    int coeff = (ii == 0 || ii == Nsim) ? 1 : ((ii % 2 == 1) ? 4 : 2);
                    double Vp = V(2.0 * p, mu);
                    dcomplex integrated_in_theta = 0.0;
                    for (int j = 0; j < n_angular; ++j) {
                        double xj = cos_theta[j]; // Chebyshev node in [-1,1]
                        double sqrt_term = sin_theta[j]; // sqrt(1 - xj^2)

                        // ---- first half: theta in [0, pi] ----
                        double cos_th1 = xj;
                        double sin_th1 = sqrt_term;
                        integrated_in_theta += integrand_sig1(p, cos_th1, sin_th1);

                        // ---- second half: theta in [pi, 2pi] ----
                        double cos_th2 = -xj;
                        double sin_th2 = -sqrt_term;
                        integrated_in_theta += integrand_sig1(p, cos_th2, sin_th2);

                    }
                    sum_integral += integrated_in_theta * Vp * p * h / 3.0 * double(coeff); // include jacobian p and potential V(p)
                }
            }

                    HF[sys.idx(1, ip, il, ik)] += prefac * sum_integral;

                }
            }
        }

        // Enforce psi-independence at l = 0 by averaging
        for (int ik = 0; ik < Nk; ++ik) {
            // compute ψ-average at l = 0
            dcomplex avg = 0.0;
            for (int ip = 0; ip < Npsi; ++ip) {
                avg += HF[sys.idx(1, ip, 0, ik)];
            }
            avg /= static_cast<double>(Npsi);

            // enforce ψ-invariance
            for (int ip = 0; ip < Npsi; ++ip) {
                HF[sys.idx(1, ip, 0, ik)] = avg;
            }
        }

    return HF;
}
