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
{   double eps =mu;
    double exp = std::exp(-q*q / (2 * eps*eps));

    double fac = q *  1.0 / (4.0 * PI * PI * pow(eps, 4)) * (q*q  / (eps*eps)  - 2.0);


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
static const GaussLegendre GL_RADIAL(6);  
static const GaussLegendre GL_ANGULAR(32); 

inline dcomplex quad1D(
    double t,           // normalized coordinate in [0,1]
    const dcomplex& f0, // i-1
    const dcomplex& f1, // i
    const dcomplex& f2  // i+1
) {
    // Lagrange basis on {-1,0,1}
    double w0 = 0.5 * t * (t - 1.0);
    double w1 = 1.0 - t * t;
    double w2 = 0.5 * t * (t + 1.0);
    return f0 * w0 + f1 * w1 + f2 * w2;
}



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
    const int n_angular = GL_ANGULAR.nodes.size();

    
    // map Gauss-Legendre nodes from [-1,1] to integration domains
    // For radial: [pmin, pmax]
    double split = 1.0 * mu;

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
    double a1 = pmin;
    double b1 = std::min(split, pmax);
    double a2 = std::max(split, pmin);
    double b2 = pmax;

    double mid1 = 0.5 * (b1 + a1);
    double half1 = 0.5 * (b1 - a1);
    double mid2 = 0.5 * (b2 + a2);
    double half2 = 0.5 * (b2 - a2);

    for (int i = 0; i < n_radial; ++i) {
        // region 1: [a1, b1]
        p1_nodes[i] = mid1 + half1 * GL_RADIAL.nodes[i];
        p1_weights[i] = half1 * GL_RADIAL.weights[i];

        // region 2: [a2, b2]
        p2_nodes[i] = mid2 + half2 * GL_RADIAL.nodes[i];
        p2_weights[i] = half2 * GL_RADIAL.weights[i];

        // cout << "p1: " << p1_nodes[i] << " w1: " << p1_weights[i] << " | p2: " << p2_nodes[i] << " w2: " << p2_weights[i] << endl;
    }

    for (int j = 0; j < n_angular; ++j) {
        theta_nodes[j] = M_PI + M_PI * GL_ANGULAR.nodes[j];
        theta_weights[j] = M_PI * GL_ANGULAR.weights[j];
        cos_theta[j] = std::cos(theta_nodes[j]);
        sin_theta[j] = std::sin(theta_nodes[j]);
        //cout << "theta: " << theta_nodes[j] << " cos: " << cos_theta[j] << " sin: " << sin_theta[j] << endl;
    }

    std::vector<double> theta_reverse(32);
    std::vector<double> cos_theta_reverse(32);

    for(int i=0; i<32; i++){
        double dth = PI / double(32);
        double theta = PI - i * dth;
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


    // trilinear sampler over (psi, k, l) using sys.idx(sig, ip, ik, il)
    auto sample_triquadratic = [&](int sig, double psi, double k, double l) -> dcomplex {

    // --- hard domain check ---
    if (psi < psi_min || psi > psi_max ||
        k   < k_min   || k   > k_max   ||
        l   < l_min   || l   > l_max) {
        return dcomplex(0.0, 0.0);
    }

    // Find central indices
    int ip = int((psi - psi_min) * inv_dpsi);
    int ik = int((k   - k_min)   * inv_dk);
    int il = int((l   - l_min)   * inv_dl);

    // Clamp so ip-1, ip, ip+1 are valid
    ip = std::clamp(ip, 1, Npsi - 2);
    ik = std::clamp(ik, 1, Nk   - 2);
    il = std::clamp(il, 1, Nl   - 2);

    // Centered normalized coordinates in [-1,1]
    double tp = (psi - psi_array[ip]) / (psi_array[ip+1] - psi_array[ip]);
    double tk = (k   - K_array[ik])   / (K_array[ik+1]   - K_array[ik]);
    double tl = (l   - L_array[il])   / (L_array[il+1]   - L_array[il]);

    // Step 1: quadratic in psi → 3×3 values
    dcomplex g[3][3];
    for (int dl = -1; dl <= 1; ++dl) {
        for (int dk = -1; dk <= 1; ++dk) {
            const dcomplex f0 = fH0[sys.idx(sig, ip-1, il+dl, ik+dk)];
            const dcomplex f1 = fH0[sys.idx(sig, ip,   il+dl, ik+dk)];
            const dcomplex f2 = fH0[sys.idx(sig, ip+1, il+dl, ik+dk)];
            g[dl+1][dk+1] = quad1D(tp, f0, f1, f2);
        }
    }

    // Step 2: quadratic in l → 3 values
    dcomplex h[3];
    for (int dk = 0; dk < 3; ++dk) {
        h[dk] = quad1D(tl, g[0][dk], g[1][dk], g[2][dk]);
    }

    // Step 3: quadratic in k → final value
    return quad1D(tk, h[0], h[1], h[2]);
};



    // string vertex = sys.vertex(); <- this is not necessary since the function is already for qqbar

    dcomplex prefac = - 1.0 / 8.0 * dcomplex(0, 1) * qtilde;

        if (!is_large_Nc) {
        throw invalid_argument("Nc mode not implemented yet in Hamiltonian_qqbar");
    }

    // ============ sig = 0 ============
    #pragma omp parallel for
    for (int idx = 0; idx < Npsi * Nk * Nl; ++idx) {
        int ip = idx / (Nl * Nk);
        int il = (idx / Nk) % Nl;
        int ik = idx % Nk;

        double psi = psi_array[ip];
        double cos_psi = cos_psi_array[ip];
        double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi * cos_psi));
        double k = K_array[ik];
        double k2 = k * k;
        double l = L_array[il];
        double l2 = l * l;
        double kl = k * l;

        dcomplex f = fH0[sys.idx(0, ip, il, ik)];
        
        HF[sys.idx(0, ip, il, ik)] = k * l * cos_psi / omega * f;   

        dcomplex sum_integral(0.0, 0.0);

        auto process_integration_point_sig0 = [&](double p, double w_p, double cos_th, double sin_th, double w_th) {
            double p2 = p * p;
            double kp = k * p;
            double lp = l * p;
            double Vp = V(p, mu);
            double w_total = w_p * w_th;
            double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;


            // Compute distances
            double Rp_m_k = std::sqrt(k2 + p2 - 2*kp*cos_th);
            double Rp_p_k = std::sqrt(k2 + p2 + 2*kp*cos_th);
            double Rp_m_l = (il == 0) ? p : std::sqrt(l2 + p2 - 2*lp*cos_theta_minus_psi);

            // Compute angles
            
            
            // p-k, l-p samples
            double numer_pmk_lmp = (p2 - kp * cos_th - lp * cos_theta_minus_psi + kl * cos_psi);
            double cos_val_pmk_lmp = std::clamp(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-7), -1.0, 1.0);
            double angle_pmk_lmp = acos_spline(cos_val_pmk_lmp);
            
            // k+p, l-p samples
            double numer_ppk_lmp = -(p2 + kp * cos_th - lp * cos_theta_minus_psi) + kl * cos_psi;
            double cos_val_ppk_lmp = std::clamp(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-7), -1.0, 1.0);
            double angle_ppk_lmp = acos_spline(cos_val_ppk_lmp);
            
            // Sample with domain checks
            bool in_domain_1 = (Rp_m_k >= k_min && Rp_m_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
            bool in_domain_2 = (Rp_p_k >= k_min && Rp_p_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
            
            // Potential contribution (only if both samples are valid)
            dcomplex S0_contrib(0.0, 0.0);
            if (in_domain_1) {
                dcomplex f0_kmp_lmp = sample_triquadratic(0, angle_pmk_lmp, Rp_m_k, Rp_m_l);
                //dcomplex f0_kpp_lmp = sample_trilinear(0, angle_ppk_lmp, Rp_p_k, Rp_m_l);
                S0_contrib += 2.0 * CF * (f - f0_kmp_lmp);
            }

            if(in_domain_2) {
                dcomplex f0_kpp_lmp = sample_triquadratic(0, angle_ppk_lmp, Rp_p_k, Rp_m_l);
                S0_contrib += 2.0 * CF * (f - f0_kpp_lmp);
            }

            return S0_contrib * Vp * w_total;
        };

        // Integrate over region 1: [a1,b1]
        if (half1 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                for (int j = 0; j < n_angular; ++j) {
                    sum_integral += process_integration_point_sig0(
                        p1_nodes[i], p1_weights[i], 
                        cos_theta[j], sin_theta[j], theta_weights[j]
                    );
                }
            }
        }

        // Integrate over region 2: [a2,b2]
        if (half2 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                for (int j = 0; j < n_angular; ++j) {
                    sum_integral += process_integration_point_sig0(
                        p2_nodes[i], p2_weights[i], 
                        cos_theta[j], sin_theta[j], theta_weights[j]
                    );
                }
            }
        }
        HF[sys.idx(0, ip, il, ik)] += prefac * sum_integral;
    }

    // Enforce psi-independence at l = 0 by averaging
    for (int ik = 0; ik < Nk; ++ik) {
        dcomplex avg(0.0, 0.0);

        for (int ip = 0; ip < Npsi; ++ip) {
            avg += HF[sys.idx(0, ip, 0, ik)];
        }

        avg /= double(Npsi);

        for (int ip = 0; ip < Npsi; ++ip) {
            HF[sys.idx(0, ip, 0, ik)] = avg;
        }
    }

    // ============ sig = 1 ============
    #pragma omp parallel for
    for (int idx = 0; idx < Npsi * Nk * Nl; ++idx) {
        int ip = idx / (Nl * Nk);
        int il = (idx / Nk) % Nl;
        int ik = idx % Nk;

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
        auto process_integration_point = [&](double p, double w_p, double cos_th, double sin_th, double w_th) {
            double p2 = p * p;
            double pl = p * l;
            double pk = p * k;
            double Vp = V(p, mu);
            double w_total = w_p * w_th;
            double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

            // M10 contribution
            double Rp_m_k = std::sqrt(k2 + p2 - 2*pk*cos_th);
            double Rp_p_k = std::sqrt(k2 + p2 + 2*pk*cos_th);
            double Rp_m_l = (il == 0) ? p : std::sqrt(l2 + p2 - 2*pl*cos_theta_minus_psi);
            double R_l_m_2z_p = std::sqrt(l2 + (2*z-1)*(2*z-1)*p2 - 4.0*pl*(2*z-1)*cos_theta_minus_psi);
            double R_l_m_2_1z_p =std::sqrt(l2 + (2*z-1)*(2*z-1)*p2 + 4.0*pl*(2*z-1)*cos_theta_minus_psi);

            // Angle calculations
            
            //k-p, l-p (sigma_0)
            double numer_pmk_lmp = (p2 - pk * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
            double cos_val_pmk_lmp = std::clamp(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-12), -1.0, 1.0);
            double angle_pmk_lmp = acos_spline(cos_val_pmk_lmp);

            //k+p, l-p (sigma_0)
            double numer_ppk_lmp = -(p2 + pk * cos_th - pl * cos_theta_minus_psi) + kl * cos_psi;
            double cos_val_ppk_lmp = std::clamp(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-12), -1.0, 1.0);
            double angle_ppk_lmp = acos_spline(cos_val_ppk_lmp);

            // p - k, l - (2z-1) p (sigma_zs)
            // numerator = p (-1 + 2 z) (p - k Cos[\[Theta]]) - l p Cos[\[Theta] - \[Psi]] + k l Cos[\[Psi]]
            double numer_pmk_l2zp = (p2 * two_z_minus_1 - pk * two_z_minus_1 * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
            double cos_val_pmk_l2zp = std::clamp(numer_pmk_l2zp / (Rp_m_k * R_l_m_2z_p + 1e-12), -1.0, 1.0);
            double angle_pmk_l2zp = acos_spline(cos_val_pmk_l2zp);

            // p - k, l + (2z-1) p (sigma_zs)
            //numerator = p^2 - 2 p^2 z + k p (-1 + 2 z) Cos[\[Theta]] - l p Cos[\[Theta] - \[Psi]] + k l Cos[\[Psi]]

            double numer_pmk_l2_1zp = (p2 - 2*p2*z + pk * two_z_minus_1 * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
            double cos_val_pmk_l2_1zp = std::clamp(numer_pmk_l2_1zp / (Rp_m_k * R_l_m_2_1z_p + 1e-12), -1.0, 1.0);
            double angle_pmk_l2_1zp = acos_spline(cos_val_pmk_l2_1zp);

            // Sample values with domain checks
            bool d1 = (Rp_m_k >= k_min && Rp_m_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
            bool d2 = (Rp_p_k >= k_min && Rp_p_k <= k_max && Rp_m_l >= l_min && Rp_m_l <= l_max);
            bool d3 = (Rp_m_k >= k_min && Rp_m_k <= k_max && R_l_m_2z_p >= l_min && R_l_m_2z_p <= l_max);
            bool d4 = (Rp_m_k >= k_min && Rp_m_k <= k_max && R_l_m_2_1z_p >= l_min && R_l_m_2_1z_p <= l_max);

            dcomplex M10_contrib(0.0, 0.0);
            if (d1) {
                dcomplex f0_kmp_lmp = sample_triquadratic(0, angle_pmk_lmp, Rp_m_k, Rp_m_l);
                //dcomplex f0_kpp_lmp = sample_trilinear(0, angle_ppk_lmp, Rp_p_k, Rp_m_l);
                //dcomplex f0_pmk_l2zp = sample_trilinear(0, angle_pmk_l2zp, Rp_m_k, R_l_m_2z_p);
                //dcomplex f0_pmk_l2_1zp = sample_trilinear(0, angle_pmk_l2_1zp, Rp_m_k, R_l_m_2_1z_p);
                M10_contrib += CA * (f_ - f0_kmp_lmp); // minus sign ?????
            }

            if (d2) {
                dcomplex f0_kpp_lmp = sample_triquadratic(0, angle_ppk_lmp, Rp_p_k, Rp_m_l);
                M10_contrib += CA * (f_ - f0_kpp_lmp);
            }

            if (d3) {
                dcomplex f0_pmk_l2zp = sample_triquadratic(0, angle_pmk_l2zp, Rp_m_k, R_l_m_2z_p);
                M10_contrib += -CA * (f_ - f0_pmk_l2zp);
            }

            if (d4) {
                dcomplex f0_pmk_l2_1zp = sample_triquadratic(0, angle_pmk_l2_1zp, Rp_m_k, R_l_m_2_1z_p);
                M10_contrib += -CA * (f_ - f0_pmk_l2_1zp);
            }

            // M11 contribution
            double R_k_zp = std::sqrt(k2 + 4*z2*p2 - 4*pk*z*cos_th);
            double R_k_1zp = std::sqrt(k2 + 4*one_z_2*p2 - 4*pk*one_z*cos_th);

            double numer_k_zp_l = (k * cos_psi - 2*z*p*cos_theta_minus_psi);
            double cos_val_k_zp_l = std::clamp(numer_k_zp_l / (R_k_zp), -1.0, 1.0);

            double angle_k_zp_l = (il > 0) ? acos_spline(cos_val_k_zp_l) : 0.0;

            double numer_k_1zp_l = (k * cos_psi - 2*one_z*p*cos_theta_minus_psi);
            double cos_val_k_1zp_l = std::clamp(numer_k_1zp_l / (R_k_1zp), -1.0, 1.0);

            double angle_k_1zp_l = (il > 0) ? acos_spline(cos_val_k_1zp_l) : 0.0;

            bool d5 = (R_k_zp >= k_min && R_k_zp <= k_max);
            bool d6 = (R_k_1zp >= k_min && R_k_1zp <= k_max);

            dcomplex M11_contrib(0.0, 0.0);
            if (d5) {
                dcomplex f1_k_zp_l = sample_triquadratic(1, angle_k_zp_l, R_k_zp, l);
                //dcomplex f1_k_1zp_l = sample_trilinear(1, angle_k_1zp_l, R_k_1zp, l);
                M11_contrib += 2.0 * CF * (f_ - f1_k_zp_l );
            }

            if (d6) {
                dcomplex f1_k_1zp_l = sample_triquadratic(1, angle_k_1zp_l, R_k_1zp, l);
                M11_contrib += 2.0 * CF * (f_ - f1_k_1zp_l);
            }

            return Vp * (M10_contrib + M11_contrib) * w_total;
        };

        // Integrate over region 1: [a1,b1]
        if (half1 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                for (int j = 0; j < n_angular; ++j) {
                    sum_integral += process_integration_point(
                        p1_nodes[i], p1_weights[i], 
                        cos_theta[j], sin_theta[j], theta_weights[j]
                    );
                }
            }
        }

        // Integrate over region 2: [a2,b2]
        if (half2 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                for (int j = 0; j < n_angular; ++j) {
                    sum_integral += process_integration_point(
                        p2_nodes[i], p2_weights[i], 
                        cos_theta[j], sin_theta[j], theta_weights[j]
                    );
                }
            }
        }

        HF[sys.idx(1, ip, il, ik)] += prefac * sum_integral;
    }

    // Enforce psi-independence at l = 0 by averaging
    for (int ik = 0; ik < Nk; ++ik) {
        dcomplex avg(0.0, 0.0);
        for (int ip = 0; ip < Npsi; ++ip) {
            avg += HF[sys.idx(1, ip, 0, ik)];
        }
        avg /= double(Npsi);
        for (int ip = 0; ip < Npsi; ++ip) {
            HF[sys.idx(1, ip, 0, ik)] = avg;
        }
    }

    return HF;
}
