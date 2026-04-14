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

    inline double fast_acos(double x) {                                                                                                                                                                                                                         
        // uses the identity acos(x) = pi/2 - asin(x)
        // and a minimax approximation for asin near 0                                                                                                                                                                                                          
        double a = std::abs(x);                                                                                                                                                                                                                                 
        double r = ((-0.0187293 * a + 0.0742610) * a - 0.2121144) * a + 1.5707288;                                                                                                                                                                              
        r = r * std::sqrt(1.0 - a);                                                                                                                                                                                                                             
        return (x < 0.0) ? M_PI - r : r;                                                                                                                                                                                                                        
    }   


// Global precomputed Gauss-Legendre quadrature points (initialize once)
static const GaussLegendre GL_RADIAL(10);  

struct current_eval {
    int ik;
    int il;
    int ip;
    double k;
    double l;
    double psi;
    double cos_psi;
    double sin_psi;
    double p;
    double theta;
    double cos_theta;
    double sin_theta;
    double pk;
    double pl;
    double kl;
};

 

vector<dcomplex> Hamiltonian(const Physis& sys, const vector<dcomplex>& fH0){

    // extract all relevant parameters from sys once
    int Nk = sys.Nk();
    int Nl = sys.Nl();
    int Npsi = sys.NPsi();
    int Nsig = sys.Nsig();
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
    const bool is_large_Nc = (sys.Ncmode() == "LNc");
    const bool is_gamma_qqbar = (sys.vertex() == "gamma_qq");
    const bool is_qqg = (sys.vertex() == "q_qg");
    const bool is_ggg = (sys.vertex() == "g_gg");

    // extract potential mode
    int mode = sys.mode();

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
    const int n_angular = 12; // can be adjusted for accuracy

    
    // map Gauss-Legendre nodes from [-1,1] to integration domains
    // For radial: [pmin, pmax]
    double sp = mu;
    double split =  (mode == 2) ? 4.0 * mu : 10.0 * mu;

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
    double a1 = 0.0;
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

    struct Fcomp2D {dcomplex f_0, f_1;};
    struct Fcomp3D {dcomplex f_0, f_1, f_2;};

    // sampler with 2nd order Taylor expansion around nearest grid point, with hard domain checks
    auto get_fval = [&](double psi, double k, double l) -> Fcomp2D {
        // --- clamp to domain ---
            psi = std::clamp(psi, psi_min, psi_max);
            k   = std::clamp(k,   k_min,   k_max);
            l   = std::clamp(l,   l_min,   l_max);

            // --- compute indices ONCE ---
            int ip_ = std::clamp(static_cast<int>((psi - psi_min) * inv_dpsi), 0, Npsi - 2);
            int ik_ = std::clamp(static_cast<int>((k   - k_min)   * inv_dk),   0, Nk   - 2);
            int il_ = std::clamp(static_cast<int>((l   - l_min)   * inv_dl),   0, Nl   - 2);

            double t_psi = (psi - psi_array[ip_]) * inv_dpsi;
            double t_k   = k - K_array[ik_];
            double t_l   = l - L_array[il_];

            double tk2   = 0.5 * t_k * t_k;
            double tl2   = 0.5 * t_l * t_l;
            double tkl   =       t_k * t_l;

            // --- flat indices computed once, reused for both sigma ---
            int base0  = sys.idx(0, ip_,     il_, ik_);
            int base1  = sys.idx(1, ip_,     il_, ik_);
            int base0n = sys.idx(0, ip_ + 1, il_, ik_);
            int base1n = sys.idx(1, ip_ + 1, il_, ik_);

            // --- sigma 0, psi slice ---
            dcomplex v0 = fH0[base0]
                + fk [base0] * t_k  + fl [base0] * t_l
                + fkk[base0] * tk2  + fll[base0] * tl2
                + flk[base0] * tkl;

            // --- sigma 0, next psi slice ---
            dcomplex v0n = fH0[base0n]
                + fk [base0n] * t_k  + fl [base0n] * t_l
                + fkk[base0n] * tk2  + fll[base0n] * tl2
                + flk[base0n] * tkl;

            // --- sigma 1, psi slice ---
            dcomplex v1 = fH0[base1]
                + fk [base1] * t_k  + fl [base1] * t_l
                + fkk[base1] * tk2  + fll[base1] * tl2
                + flk[base1] * tkl;

            // --- sigma 1, next psi slice ---
            dcomplex v1n = fH0[base1n]
                + fk [base1n] * t_k  + fl [base1n] * t_l
                + fkk[base1n] * tk2  + fll[base1n] * tl2
                + flk[base1n] * tkl;

            double one_t = 1.0 - t_psi;
            return { one_t * v0 + t_psi * v0n,
                    one_t * v1 + t_psi * v1n };
        };

    // string vertex = sys.vertex(); <- this is not necessary since the function is already for qqbar

    dcomplex prefac = - 4.0 * dcomplex(0, 1) * qtilde / (2.0 * PI); //with the 4 factor from the jacobian

    // if (!is_large_Nc) {
    //     throw invalid_argument("Nc mode not implemented yet in Hamiltonian_qqbar");
    // }

    #pragma omp parallel for collapse(3)
    for (int ip = 0; ip < Npsi; ++ip) {
        for (int il = 0; il < Nl; ++il) {
            for (int ik = 0; ik < Nk; ++ik) {

                double psi     = psi_array[ip];
                double cos_psi = cos_psi_array[ip];
                double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi * cos_psi));
                double k  = K_array[ik];
                double k2 = k * k;
                double l  = L_array[il];
                double l2 = l * l;
                double kl = k * l;

                auto idx0 = sys.idx(0, ip, il, ik);
                auto idx1 = sys.idx(1, ip, il, ik);

                const dcomplex f0_ = fH0[idx0];
                const dcomplex f1_ = fH0[idx1];

                // kinetic terms
                HF[idx0] = (k * l) / omega * cos_psi * f0_;
                HF[idx1] = (k * l) / omega * cos_psi * f1_;


                dcomplex sum0(0.0, 0.0);
                dcomplex sum1(0.0, 0.0);

                // returns {contrib_sig0, contrib_sig1} for one (p, cos_th, sin_th) point
                auto integrand_qqbar = [&](double p, double cos_th, double sin_th)
                    -> std::pair<dcomplex, dcomplex>
                {
                    double p2  = p * p;
                    double pk  = p * k;
                    double pl  = p * l;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    // ---- distances (computed once, shared by both sigma) ----
                    double Rp_m_k = std::sqrt(k2 + p2 - 2.0*pk*cos_th);
                    double Rp_p_k = std::sqrt(k2 + p2 + 2.0*pk*cos_th);
                    double Rp_m_l = (il == 0) ? p
                                              : std::sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);

                    double R_k_zp      = std::sqrt(k2 + 4.0*z2*p2        - 4.0*pk*z*cos_th);
                    double R_k_1zp     = std::sqrt(k2 + 4.0*one_z_2*p2   - 4.0*pk*one_z*cos_th);
                    double R_k_m_2z_p  = std::sqrt(k2 + two_z_minus_1_2*p2 - 2.0*pk*two_z_minus_1*cos_th);
                    double R_k_m_2_1z_p= std::sqrt(k2 + two_z_minus_1_2*p2 + 2.0*pk*two_z_minus_1*cos_th);

                    // ---- angles (computed once, shared by both sigma) ----
                    // helper: angle = atan2(sqrt(1-c^2), c) with clamped c
                    auto angle_from_cos = [](double num, double denom) -> double {
                        double c = num / (denom + 1e-12); // add small number to avoid division by zero
                        return fast_acos(std::clamp(c, -1.0, 1.0)); // use fast_acos for efficiency
                    };

                    double angle_pmk_lmp = angle_from_cos(
                        p2 - pk*cos_th - pl*cos_theta_minus_psi + kl*cos_psi,
                        Rp_m_k * Rp_m_l);

                    double angle_ppk_lmp = angle_from_cos(
                        -(p2 + pk*cos_th - pl*cos_theta_minus_psi) + kl*cos_psi,
                        Rp_p_k * Rp_m_l);

                    double angle_k_m_2z_p = angle_from_cos(
                        -pk*cos_th + p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2z_p * Rp_m_l);

                    double angle_k_m_2_1z_p = angle_from_cos(
                        -pk*cos_th - p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2_1z_p * Rp_m_l);

                    double angle_k_zp_l  = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*z*p*cos_theta_minus_psi)     / (R_k_zp  + 1e-12), -1.0, 1.0)) : 0.0;

                    double angle_k_1zp_l = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi) / (R_k_1zp + 1e-12), -1.0, 1.0)) : 0.0;



                    // ---- sample f[0] at all geometry points ----
                    auto [f0_pmk_lmp,   f1_pmk_lmp]   = get_fval(angle_pmk_lmp,    Rp_m_k,      Rp_m_l);
                    auto [f0_ppk_lmp,   f1_ppk_lmp]   = get_fval(angle_ppk_lmp,    Rp_p_k,      Rp_m_l);
                    auto [f0_2zp_lmp,   f1_2zp_lmp]   = get_fval(angle_k_m_2z_p,   R_k_m_2z_p,  Rp_m_l);
                    auto [f0_2_1zp_lmp, f1_2_1zp_lmp] = get_fval(angle_k_m_2_1z_p, R_k_m_2_1z_p,Rp_m_l);
                    auto [f0_kzp,       f1_kzp]        = get_fval(angle_k_zp_l,     R_k_zp,      l);
                    auto [f0_k1zp,      f1_k1zp]       = get_fval(angle_k_1zp_l,    R_k_1zp,     l);



                    // ---- build Sigma combinations ----
                    // Sig0:  2f - f(k-p, l-p) - f(k+p, l-p)
                    dcomplex Sig0_f0  = 2.0*f0_ - f0_pmk_lmp - f0_ppk_lmp;
                    dcomplex Sig0_f1  = 2.0*f1_ - f1_pmk_lmp - f1_ppk_lmp;

                    // Sig_zsc: 2f - f(k-(2z-1)p, l-p) - f(k+(2z-1)p, l-p)
                    dcomplex Sigzsc_f0 = 2.0*f0_ - f0_2zp_lmp - f0_2_1zp_lmp;
                    dcomplex Sigzsc_f1 = 2.0*f1_ - f1_2zp_lmp - f1_2_1zp_lmp;

                    // Sig_plus:  f - f(k-2zp, l)
                    dcomplex Sigp_f0 = f0_ - f0_kzp;
                    dcomplex Sigp_f1 = f1_ - f1_kzp;

                    // Sig_minus: f - f(k-2(1-z)p, l)
                    dcomplex Sigm_f0 = f0_ - f0_k1zp;
                    dcomplex Sigm_f1 = f1_ - f1_k1zp;

                    // ---- assemble M-matrix contributions ----
                    // sig=0 output:  M00 (into HF[idx0]) + M01 (cross term from sig=1)
                    dcomplex contrib0 = 2.0*CF * Sig0_f0;                                 // M00, Sig0
                    
                    dcomplex contrib1 = 
                          CA * (f0_2zp_lmp + f0_2_1zp_lmp - f0_pmk_lmp - f0_ppk_lmp)  // M10
                          + 2.0*CF * (2.0*f1_ - f1_kzp - f1_k1zp);                   // M11


                    if(!is_large_Nc){
                        contrib0 += (1.0/CA) * (Sigzsc_f0 - Sigp_f0 - Sigm_f0) - (1.0/CA) * (Sigzsc_f1 - Sigp_f1 - Sigm_f1);    
                        contrib1 += (1.0/CA) * (Sigzsc_f1 - Sig0_f1);
                    }

                    return {contrib0, contrib1};
                };

                auto integrand_qqg = [&](double p, double cos_th, double sin_th)
                    -> std::vector<dcomplex>
                {
                    double p2  = p * p;
                    double pk  = p * k;
                    double pl  = p * l;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    // ---- distances (computed once, shared by both sigma) ----
                    double Rp_m_k = std::sqrt(k2 + p2 - 2.0*pk*cos_th);
                    //double Rp_m_k = p-k;
                    double Rp_p_k = std::sqrt(k2 + p2 + 2.0*pk*cos_th);
                    //double Rp_p_k = p+k;
                    double Rp_m_l = (il == 0) ? p : std::sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);
                    // double Rp_m_l = p-l;

                    double R_k_zp      = std::sqrt(k2 + 4.0*z2*p2        - 4.0*pk*z*cos_th);
                    // double R_k_zp      = k + p;
                    double R_k_1zp     = std::sqrt(k2 + 4.0*one_z_2*p2   - 4.0*pk*one_z*cos_th);

                    //double R_k_1zp     = k + p;
                    double R_k_m_2z_p  = std::sqrt(k2 + two_z_minus_1_2*p2 - 2.0*pk*two_z_minus_1*cos_th);
                    //double R_k_m_2z_p  = k + p;
                    double R_k_m_2_1z_p= std::sqrt(k2 + two_z_minus_1_2*p2 + 2.0*pk*two_z_minus_1*cos_th);
                    // double R_k_m_2_1z_p= k + p;

                    // // ---- angles (computed once, shared by both sigma) ----
                    // // helper: angle = atan2(sqrt(1-c^2), c) with clamped c
                     auto angle_from_cos = [](double num, double denom) -> double {
                        double c = num / (denom + 1e-12); // add small number to avoid division by zero
                        return fast_acos(std::clamp(c, -1.0, 1.0)); // use fast_acos for efficiency
                    };

                    // double angle_pmk_lmp = psi;
                    // double angle_ppk_lmp = psi;
                    // double angle_k_m_2z_p = psi;
                    // double angle_k_m_2_1z_p = psi;
                    // double angle_k_zp_l  = psi;
                    // double angle_k_1zp_l = psi;

                    double angle_pmk_lmp = angle_from_cos(
                        p2 - pk*cos_th - pl*cos_theta_minus_psi + kl*cos_psi,
                        Rp_m_k * Rp_m_l);

                    double angle_ppk_lmp = angle_from_cos(
                        -(p2 + pk*cos_th - pl*cos_theta_minus_psi) + kl*cos_psi,
                        Rp_p_k * Rp_m_l);

                    double angle_k_m_2z_p = angle_from_cos(
                        -pk*cos_th + p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2z_p * Rp_m_l);

                    double angle_k_m_2_1z_p = angle_from_cos(
                        -pk*cos_th - p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2_1z_p * Rp_m_l);

                    //double angle_k_zp_l  = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*z*p*cos_theta_minus_psi)     / (R_k_zp  + 1e-12), -1.0, 1.0)) : 0.0;

                    double angle_k_zp_l = angle_from_cos(k*cos_psi - 2.0*z*p*cos_theta_minus_psi, R_k_zp);

                    //double angle_k_1zp_l = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi) / (R_k_1zp + 1e-12), -1.0, 1.0)) : 0.0;

                    double angle_k_1zp_l = angle_from_cos(k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi, R_k_1zp);


                    // ---- sample f[0] at all geometry points ----
                    auto [f0_pmk_lmp,   f1_pmk_lmp]   = get_fval(angle_pmk_lmp,    Rp_m_k,      Rp_m_l);
                    auto [f0_ppk_lmp,   f1_ppk_lmp]   = get_fval(angle_ppk_lmp,    Rp_p_k,      Rp_m_l);
                    auto [f0_2zp_lmp,   f1_2zp_lmp]   = get_fval(angle_k_m_2z_p,   R_k_m_2z_p,  Rp_m_l);
                    auto [f0_2_1zp_lmp, f1_2_1zp_lmp] = get_fval(angle_k_m_2_1z_p, R_k_m_2_1z_p,Rp_m_l);
                    auto [f0_kzp,       f1_kzp]        = get_fval(angle_k_zp_l,     R_k_zp,      l);
                    auto [f0_k1zp,      f1_k1zp]       = get_fval(angle_k_1zp_l,    R_k_1zp,     l);



                    // ---- build Sigma combinations ----
                    // Sig0:  2f - f(k-p, l-p) - f(k+p, l-p)
                    dcomplex Sig0_f0  = 2.0*f0_ - f0_pmk_lmp - f0_ppk_lmp;
                    dcomplex Sig0_f1  = 2.0*f1_ - f1_pmk_lmp - f1_ppk_lmp;

                    // Sig_zsc: 2f - f(k-(2z-1)p, l-p) - f(k+(2z-1)p, l-p)
                    dcomplex Sigzsc_f0 = 2.0*f0_ - f0_2zp_lmp - f0_2_1zp_lmp;
                    dcomplex Sigzsc_f1 = 2.0*f1_ - f1_2zp_lmp - f1_2_1zp_lmp;

                    // Sig_plus:  f - f(k-2zp, l)
                    dcomplex Sigp_f0 = f0_ - f0_kzp;
                    dcomplex Sigp_f1 = f1_ - f1_kzp;

                    // Sig_minus: f - f(k-2(1-z)p, l)
                    dcomplex Sigm_f0 = f0_ - f0_k1zp;
                    dcomplex Sigm_f1 = f1_ - f1_k1zp;

                    // ---- assemble M-matrix contributions ----
                    // sig=0 output:  M00 (into HF[idx0]) + M01 (cross term from sig=1)
                    dcomplex contrib0 = CA * (Sigm_f0 + Sig0_f0);                                 // M00, Sig0
                    dcomplex contrib1 = CA * (-Sigzsc_f0 + Sig0_f0) + 2.0 * (CF * Sigp_f1 + CA * Sigm_f1); 

                    dcomplex contrib2 = 0.0;
                    

                    // if(!is_large_Nc){
                    //     // we will have a new component
                    //     // sample f[2] at all geometry points
                    //     dcomplex f2_pmk_lmp    = get_fval(2, angle_pmk_lmp,     Rp_m_k,          Rp_m_l, ip, ik, il);
                    //     dcomplex f2_ppk_lmp    = get_fval(2, angle_ppk_lmp,     Rp_p_k,          Rp_m_l, ip, ik, il);
                    //     dcomplex f2_2zp_lmp    = get_fval(2, angle_k_m_2z_p,    R_k_m_2z_p,      Rp_m_l, ip, ik, il);
                    //     dcomplex f2_2_1zp_lmp  = get_fval(2, angle_k_m_2_1z_p,  R_k_m_2_1z_p,    Rp_m_l, ip, ik, il);
                    //     dcomplex f2_kzp        = get_fval(2, angle_k_zp_l,      R_k_zp,          l,      ip, ik, il);
                    //     dcomplex f2_k1zp       = get_fval(2, angle_k_1zp_l,     R_k_1zp,         l,      ip, ik, il);

                    //     // Sig for the extra component
                    //     dcomplex Sig0_f2 = 2.0*f2_ - f2_pmk_lmp - f2_ppk_lmp;
                    //     dcomplex Sigzsc_f2 = 2.0*f2_ - f2_2zp_lmp - f2_2_1zp_lmp;
                    //     dcomplex Sigp_f2 = f2_ - f2_kzp;
                    //     dcomplex Sigm_f2 = f2_ - f2_k1zp;
                        
                    //     contrib0 += -Sigp_f0/CA + (Sigzsc_f2 - Sig0_f2);

                    //     contrib2 = (Sigzsc_f0 - Sig0_f0) + CA * (Sigzsc_f1 + Sig0_f1 -2.0 * (Sigp_f1 + Sigm_f1)) + (-(1/CA) * Sigp_f2 + CA * (Sigzsc_f2 + Sigm_f2)); //M20 + M21 + M22
                        
                    // }

                    return {contrib0, contrib1, contrib2};
                };

                auto integrand_ggg = [&](double p, double cos_th, double sin_th)
                    -> std::vector<dcomplex>
                {
                    double p2  = p * p;
                    double pk  = p * k;
                    double pl  = p * l;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    // ---- distances (computed once, shared by both sigma) ----
                    double Rp_m_k = std::sqrt(k2 + p2 - 2.0*pk*cos_th);
                    //double Rp_m_k = p-k;
                    double Rp_p_k = std::sqrt(k2 + p2 + 2.0*pk*cos_th);
                    //double Rp_p_k = p+k;
                    double Rp_m_l = (il == 0) ? p : std::sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);
                    // double Rp_m_l = p-l;

                    double R_k_zp      = std::sqrt(k2 + 4.0*z2*p2        - 4.0*pk*z*cos_th);
                    // double R_k_zp      = k + p;
                    double R_k_1zp     = std::sqrt(k2 + 4.0*one_z_2*p2   - 4.0*pk*one_z*cos_th);

                    //double R_k_1zp     = k + p;
                    double R_k_m_2z_p  = std::sqrt(k2 + two_z_minus_1_2*p2 - 2.0*pk*two_z_minus_1*cos_th);
                    //double R_k_m_2z_p  = k + p;
                    double R_k_m_2_1z_p= std::sqrt(k2 + two_z_minus_1_2*p2 + 2.0*pk*two_z_minus_1*cos_th);
                    // double R_k_m_2_1z_p= k + p;

                    // // ---- angles (computed once, shared by both sigma) ----
                    // // helper: angle = atan2(sqrt(1-c^2), c) with clamped c
                     auto angle_from_cos = [](double num, double denom) -> double {
                        double c = num / (denom + 1e-12); // add small number to avoid division by zero
                        return fast_acos(std::clamp(c, -1.0, 1.0)); // use fast_acos for efficiency
                    };

                    // double angle_pmk_lmp = psi;
                    // double angle_ppk_lmp = psi;
                    // double angle_k_m_2z_p = psi;
                    // double angle_k_m_2_1z_p = psi;
                    // double angle_k_zp_l  = psi;
                    // double angle_k_1zp_l = psi;

                    double angle_pmk_lmp = angle_from_cos(
                        p2 - pk*cos_th - pl*cos_theta_minus_psi + kl*cos_psi,
                        Rp_m_k * Rp_m_l);

                    double angle_ppk_lmp = angle_from_cos(
                        -(p2 + pk*cos_th - pl*cos_theta_minus_psi) + kl*cos_psi,
                        Rp_p_k * Rp_m_l);

                    double angle_k_m_2z_p = angle_from_cos(
                        -pk*cos_th + p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2z_p * Rp_m_l);

                    double angle_k_m_2_1z_p = angle_from_cos(
                        -pk*cos_th - p*two_z_minus_1*(p - pl*cos_theta_minus_psi) + kl*cos_psi,
                        R_k_m_2_1z_p * Rp_m_l);

                    //double angle_k_zp_l  = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*z*p*cos_theta_minus_psi)     / (R_k_zp  + 1e-12), -1.0, 1.0)) : 0.0;

                    double angle_k_zp_l = angle_from_cos(k*cos_psi - 2.0*z*p*cos_theta_minus_psi, R_k_zp);

                    //double angle_k_1zp_l = (il > 0) ? std::acos(std::clamp((k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi) / (R_k_1zp + 1e-12), -1.0, 1.0)) : 0.0;

                    double angle_k_1zp_l = angle_from_cos(k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi, R_k_1zp);


                    // ---- sample f[0] at all geometry points ----
                    auto [f0_pmk_lmp,   f1_pmk_lmp]   = get_fval(angle_pmk_lmp,    Rp_m_k,      Rp_m_l);
                    auto [f0_ppk_lmp,   f1_ppk_lmp]   = get_fval(angle_ppk_lmp,    Rp_p_k,      Rp_m_l);
                    auto [f0_2zp_lmp,   f1_2zp_lmp]   = get_fval(angle_k_m_2z_p,   R_k_m_2z_p,  Rp_m_l);
                    auto [f0_2_1zp_lmp, f1_2_1zp_lmp] = get_fval(angle_k_m_2_1z_p, R_k_m_2_1z_p,Rp_m_l);
                    auto [f0_kzp,       f1_kzp]        = get_fval(angle_k_zp_l,     R_k_zp,      l);
                    auto [f0_k1zp,      f1_k1zp]       = get_fval(angle_k_1zp_l,    R_k_1zp,     l);



                    // ---- build Sigma combinations ----
                    // Sig0:  2f - f(k-p, l-p) - f(k+p, l-p)
                    dcomplex Sig0_f0  = 2.0*f0_ - f0_pmk_lmp - f0_ppk_lmp;
                    dcomplex Sig0_f1  = 2.0*f1_ - f1_pmk_lmp - f1_ppk_lmp;

                    // Sig_zsc: 2f - f(k-(2z-1)p, l-p) - f(k+(2z-1)p, l-p)
                    dcomplex Sigzsc_f0 = 2.0*f0_ - f0_2zp_lmp - f0_2_1zp_lmp;
                    dcomplex Sigzsc_f1 = 2.0*f1_ - f1_2zp_lmp - f1_2_1zp_lmp;

                    // Sig_plus:  f - f(k-2zp, l)
                    dcomplex Sigp_f0 = f0_ - f0_kzp;
                    dcomplex Sigp_f1 = f1_ - f1_kzp;

                    // Sig_minus: f - f(k-2(1-z)p, l)
                    dcomplex Sigm_f0 = f0_ - f0_k1zp;
                    dcomplex Sigm_f1 = f1_ - f1_k1zp;

                    // ---- assemble M-matrix contributions ----
                    // sig=0 output:  M00 (into HF[idx0]) + M01 (cross term from sig=1)
                    dcomplex contrib0 = CA * (Sigm_f0 + Sigp_f0 + Sig0_f0);                                 // M00, Sig0
                    dcomplex contrib1 = CA * (-Sigzsc_f0 + Sig0_f0) + 2.0 * CA * (Sigp_f1 + Sigm_f1);

                    dcomplex contrib2 = 0.0;
                    

                    if(!is_large_Nc){
                        std::cerr << "Nc mode not implemented yet in ggg case" << std::endl;
                    }

                    return {contrib0, contrib1, contrib2};
                };

                // ---- quadrature over region 1: [a1, b1] ----
                if (half1 > 0.0) {
                    int Nsim = std::max(2, 2*n_radial);
                    // ensure Nsim is even for Simpson's rule
                    if (Nsim % 2 != 0) ++Nsim;
                    double h = (b1 - a1) / static_cast<double>(Nsim);

                    dcomplex racc0(0.0, 0.0), racc1(0.0, 0.0);

                    for (int ii = 0; ii <= Nsim; ++ii) {
                        double p    = a1 + ii * h;
                        int coeff   = (ii == 0 || ii == Nsim) ? 1 : (ii % 2 == 1 ? 4 : 2);
                        double Vp   = V(2.0 * p, sp);
                        double w    = h / 3.0 * p * Vp * double(coeff);

                        dcomplex tacc0(0.0, 0.0), tacc1(0.0, 0.0);

                        for (int j = 0; j < n_angular; ++j) {
                            double cj = cos_theta[j];
                            double sj = sin_theta[j];

                            // theta in [0, pi]
                            if(is_gamma_qqbar){
                                auto [d0, d1] = integrand_qqbar(p,  cj,  sj);
                                auto [e0, e1] = integrand_qqbar(p, -cj, -sj);
                                tacc0 += d0 + e0;
                                tacc1 += d1 + e1;
                            }

                            if(is_qqg){
                                vector<dcomplex> d = integrand_qqg(p,  cj,  sj);
                                vector<dcomplex> e = integrand_qqg(p, -cj, -sj);
                                tacc0 += d[0] + e[0];
                                tacc1 += d[1] + e[1];
                                if(!is_large_Nc){
                                    tacc0 += d[2] + e[2];
                                }
                            }

                            if(is_ggg){
                                vector<dcomplex> d = integrand_ggg(p,  cj,  sj);
                                vector<dcomplex> e = integrand_ggg(p, -cj, -sj);
                                tacc0 += d[0] + e[0];
                                tacc1 += d[1] + e[1];
                                // if(!is_large_Nc){
                                //     std::throw invalid_argument("Nc mode not implemented yet in ggg case");
                                // }
                            }

                            // theta in [pi, 2pi]  (cos -> -cos, sin -> -sin)
                        }

                        // Chebyshev weight pi/Ntheta
                        tacc0 *= M_PI / static_cast<double>(n_angular);
                        tacc1 *= M_PI / static_cast<double>(n_angular);

                        racc0 += w * tacc0;
                        racc1 += w * tacc1;
                    }

                    sum0 += racc0;
                    sum1 += racc1;
                }

                // ---- quadrature over region 2: [a2, b2] ----
                // if (false) {
                //     int Nsim = std::max(2, n_radial);
                //     if (Nsim % 2 != 0) ++Nsim;
                //     double h = (b2 - a2) / static_cast<double>(Nsim);

                //     dcomplex racc0(0.0, 0.0), racc1(0.0, 0.0);

                //     for (int ii = 0; ii <= Nsim; ++ii) {
                //         double p  = a2 + ii * h;
                //         int coeff = (ii == 0 || ii == Nsim) ? 1 : (ii % 2 == 1 ? 4 : 2);
                //         double Vp = V(2.0 * p, mu);
                //         double w  = h / 3.0 * p * Vp * double(coeff);

                //         dcomplex tacc0(0.0, 0.0), tacc1(0.0, 0.0);

                //         for (int j = 0; j < n_angular; ++j) {
                //             double cj = cos_theta[j];
                //             double sj = sin_theta[j];

                //             auto [d0, d1] = integrand_fused(p,  cj,  sj);
                //             tacc0 += d0; tacc1 += d1;

                //             auto [e0, e1] = integrand_fused(p, -cj, -sj);
                //             tacc0 += e0; tacc1 += e1;
                //         }

                //         tacc0 *= M_PI / static_cast<double>(n_angular);
                //         tacc1 *= M_PI / static_cast<double>(n_angular);

                //         racc0 += w * tacc0;
                //         racc1 += w * tacc1;
                //     }

                //     sum0 += racc0;
                //     sum1 += racc1;
                // }

                HF[idx0] += prefac * sum0;
                HF[idx1] += prefac * sum1;
            }
        }
    }

    // ---- enforce psi-independence at l = 0 for both sigma ----
    for (int sig = 0; sig < Nsig; ++sig) {
        for (int ik = 0; ik < Nk; ++ik) {
            dcomplex avg = 0.0;
            for (int ip = 0; ip < Npsi; ++ip)
                avg += HF[sys.idx(sig, ip, 0, ik)];
            avg /= static_cast<double>(Npsi);
            for (int ip = 0; ip < Npsi; ++ip)
                HF[sys.idx(sig, ip, 0, ik)] = avg;
        }
    }


    return HF;
}
