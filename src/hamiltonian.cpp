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
    return  1.0 / (q*q*(q*q + (e * mu*mu))); //q / pow((q*q + mu*mu), 2); 
}

double VHO_eff(double q, double mu)
{   double eps = mu;
    double exp = std::exp(-q*q / (2 * eps*eps));

    double fac = 0.25 * 1.0 / (2.0 * pow(eps, 4)) * (q*q  / (eps*eps)  - 2.0);


    return exp * fac;
}

double VYUK_eff(double q, double mu)
{   double eps = mu;
 
    return  1.0 / pow((q*q + mu*mu), 2);
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
static const GaussLegendre GL_RADIAL(20);

static constexpr int MAX_NSIG = 6;
static constexpr int N_GEOM   = 6;   // number of interpolated geometry points
using SigArr   = std::array<dcomplex, MAX_NSIG>;
using PointMsk = std::array<bool, MAX_NSIG>;
using SampMask = std::array<PointMsk, N_GEOM>;

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

inline double ramp(double x, double x0, double xmax) {
    if (x <= x0) return 0.0;
    double s = (x - x0) / (xmax - x0);
    if (s >= 1.0) s = 1.0;
    return s * s * s * s;  // quartic, C^1 at the join
}

double sponge_profile(double k, double l,
                      double Lk, double Ll,
                      double k0, double l0,
                      double gamma0) {
    // separate ramps in k and l, combined so corner is also damped
    double rk = ramp(k, k0, Lk);
    double rl = ramp(l, l0, Ll);
    return gamma0 * (rk + rl);  // or std::max(rk, rl) if you prefer
}



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

        // sanity check
    if (Nsig > MAX_NSIG) {
        throw std::invalid_argument("Nsig exceeds MAX_NSIG.");
    }

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
    const int n_angular = 6; // can be adjusted for accuracy

    
    // map Gauss-Legendre nodes from [-1,1] to integration domains
    // For radial: [pmin, pmax]
    double sp = mu;
    double split =  (mode == 2) ? 6.0 * mu : 60.0 * mu;

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
    double a1 = 0.0*mu;
    double b1 = split; 

    double mid1 = 0.5 * (b1 + a1);
    double half1 = 0.5 * (b1 - a1);
    

    for (int i = 0; i < n_radial; ++i) {
        // region 1: [a1, b1]
        p1_nodes[i] = mid1 + half1 * GL_RADIAL.nodes[i];
        p1_weights[i] = half1 * GL_RADIAL.weights[i];

        //cout << "p1: " << p1_nodes[i] << " w1: " << p1_weights[i] << endl;

        // region 2: [a2, b2
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

    // sampler with 2nd order Taylor expansion around nearest grid point, with hard domain checks
        // precomputed constants
    double z2              = z * z;
    double one_z           = 1.0 - z;
    double one_z_2         = one_z * one_z;
    double two_z_minus_1   = 2.0 * z - 1.0;
    double two_z_minus_1_2 = two_z_minus_1 * two_z_minus_1;
 
    auto V = [&mode](double p, double mu) -> double {
        if (mode == 0) return VYUK_eff(p, mu);
        else if (mode == 1) return VHTL_eff(p, mu);
        else               return VHO_eff(p, mu);
    };
 
    vector <InterpCoeffs> der_coeffs;


    precompute_derivatives_3d(sys, fH0, der_coeffs);
 
    dcomplex prefac = -2.0 * dcomplex(0, 1) * qtilde / (PI);
 
    // ==================================================================
    // Build the sampling mask from the M-matrix formulas.
    //
    // Rules (derived from inspecting each M_* body):
    //   - If M[...] reads Sig0  [s] -> need points 0 and 1 for sigma s
    //   - If M[...] reads Sigzsc[s] -> need points 2 and 3 for sigma s
    //   - If M[...] reads Sigp  [s] -> need point 4 for sigma s
    //   - If M[...] reads Sigm  [s] -> need point 5 for sigma s
    //
    // f_here[s] is always filled for all s < Nsig (it's one array lookup,
    // needed for the kinetic term).
    // ==================================================================
 
    SampMask mask{};  // all false by default
 
    auto need_Sig0   = [&](int s) { mask[0][s] = mask[1][s] = true; };
    auto need_Sigzsc = [&](int s) { mask[2][s] = mask[3][s] = true; };
    auto need_Sigp   = [&](int s) { mask[4][s] = true; };
    auto need_Sigm   = [&](int s) { mask[5][s] = true; };
 
    if (is_gamma_qqbar) {
        // M_qqbar:
        //   out[0]    = 2 CF Sig0[0]
        //   out[1]    = CA (Sig0[0] - Sigzsc[0]) + 2 CF (Sigp[1] + Sigm[1])
        //   FNc adds to out[0]: (1/CA) (Sigzsc[0..1] - Sigp[0..1] - Sigm[0..1])
        //   FNc adds to out[1]: (1/CA) (Sigzsc[1] - Sig0[1])
        need_Sig0(0);
        need_Sigzsc(0);
        need_Sigp(1);
        need_Sigm(1);
        if (!is_large_Nc) {
            need_Sigzsc(0); need_Sigzsc(1);
            need_Sigp  (0); need_Sigp  (1);
            need_Sigm  (0); need_Sigm  (1);
            need_Sig0  (1);
        }
    } else if (is_qqg) {
        // M_qqg:
        //   out[0] = CA (Sigm[0] + Sig0[0])
        //   out[1] = CA (-Sigzsc[0] + Sig0[0]) + 2 (CF Sigp[1] + CA Sigm[1])
        //   FNc: out[0] += -Sigp[0]/CA + (Sigzsc[2] - Sig0[2])
        //        out[2]  = (Sigzsc[0] - Sig0[0])
        //                + CA (Sigzsc[1] + Sig0[1] - 2(Sigp[1]+Sigm[1]))
        //                + (-Sigp[2]/CA + CA (Sigzsc[2] + Sigm[2]))
        need_Sig0(0);  need_Sigm(0);
        need_Sigzsc(0);
        need_Sigp(1);  need_Sigm(1);
        if (!is_large_Nc) {
            need_Sigp(0);
            need_Sigzsc(2); need_Sig0(2);
            need_Sig0(1);   need_Sigzsc(1);
            need_Sigp(2);   need_Sigm(2);
        }
    } else if (is_ggg) {
        // M_ggg:
        //   out[0] = CA (Sigm[0] + Sigp[0] + Sig0[0])
        //   out[1] = CA (-Sigzsc[0] + Sig0[0]) + 2 CA (Sigp[1] + Sigm[1])
        need_Sig0(0);  need_Sigp(0);  need_Sigm(0);
        need_Sigzsc(0);
        need_Sigp(1);  need_Sigm(1);
        if (!is_large_Nc) {
            // TODO: once ggg FNc (Nsig=6) formulas are written in M_ggg,
            // replace this conservative fallback with the exact mask.
            for (int s = 0; s < Nsig; ++s) {
                need_Sig0(s); need_Sigzsc(s); need_Sigp(s); need_Sigm(s);
            }
        }
    } else {
        throw std::invalid_argument("Unknown vertex type");
    }
 
    // ==================================================================
    // Mask-aware sampler. Fills only result[s] where point_mask[s] is true.
    // Unmasked entries stay zero (harmless: the M-matrix won't read them).
    // ==================================================================
    auto get_fval = [&](double psi, double k, double l,
                    const PointMsk& point_mask) -> SigArr {
        // ---- 1. Canonicalize angle to [0, π] ----
        psi = std::abs(std::remainder(psi, 2.0 * M_PI));

        // ---- 2. Out-of-range in (k, l) -> zero. No silent clamping. ----
        if (k < k_min || k > k_max || l < l_min || l > l_max) {
            return SigArr{};
        }

        // ---- 3. Detect degenerate axes and locate cells accordingly. ----
        // An axis is "degenerate" if it has fewer than 2 cells (i.e., N <= 1
        // means a single node, N == 2 means a single cell, N >= 2 means we can
        // interpolate). The interpolation uses N-1 cells, so we need N >= 2 to
        // have a cell at all. For N == 1, the axis collapses to a constant.
        //
        // We always need k to be interpolable (the radial axis matters for
        // physics), so we don't add a degenerate-k branch — assert it instead.

        const bool psi_degenerate = (Npsi <= 2);
        const bool l_degenerate   = (Nl   <= 2);

        assert(Nk >= 2 && "k axis must have at least 2 nodes for interpolation");

        // Locate the cell on each axis. For degenerate axes, ip_ or il_ stays at 0
        // and the corresponding parameter u or w is forced to 0 (so the "upper"
        // corner gets zero weight).
        int ip_ = 0;
        int il_ = 0;
        int ik_ = std::clamp(static_cast<int>(std::floor((k - k_min) * inv_dk)),
                            0, Nk - 2);

        double u = 0.0;
        double w = 0.0;
        const double v = (k - K_array[ik_]) * inv_dk;     // k axis is never degenerate

        if (!psi_degenerate) {
            ip_ = std::clamp(static_cast<int>(std::floor((psi - psi_min) * inv_dpsi)),
                            0, Npsi - 2);
            u = (psi - psi_array[ip_]) * inv_dpsi;
        }
        // else: ip_ = 0, u = 0 — the (ip_+1) corners will get zero weight,
        //       so we never actually read der_coeffs at an out-of-range index
        //       as long as Npsi >= 1 (which it always must be).

        if (!l_degenerate) {
            il_ = std::clamp(static_cast<int>(std::floor((l - l_min) * inv_dl)),
                            0, Nl - 2);
            w = (l - L_array[il_]) * inv_dl;
        }
        // else: il_ = 0, w = 0 — same logic.

        // ---- 4. Compute the trilinear weights as usual. ----
        // When u = 0 (degenerate psi), only the four corners with offset 0 in psi
        // get nonzero weight. Same for w. Math works out correctly without any
        // special-case branching in the weighted sum below.
        const double um = 1.0 - u, vm = 1.0 - v, wm = 1.0 - w;
        const double w000 = um * wm * vm;
        const double w100 = u  * wm * vm;
        const double w010 = um * wm * v;
        const double w110 = u  * wm * v;
        const double w001 = um * w  * vm;
        const double w101 = u  * w  * vm;
        const double w011 = um * w  * v;
        const double w111 = u  * w  * v;

        assert(std::abs(w000 + w100 + w010 + w110 + w001 + w101 + w011 + w111 - 1.0) < 1e-12);

        // ---- 5. Determine the safe upper-corner indices. ----
        // For degenerate axes, we still need a valid index for sys.idx, even
        // though that corner's weight is zero. Use the same node as the "lower"
        // corner — the read happens but contributes nothing.
        const int ip_next = psi_degenerate ? ip_ : (ip_ + 1);
        const int il_next = l_degenerate   ? il_ : (il_ + 1);
        // ik always has at least 2 nodes, so ik_ + 1 is always valid:
        const int ik_next = ik_ + 1;

        SigArr result{};
        for (int s = 0; s < Nsig; ++s) {
            if (!point_mask[s]) continue;

            const dcomplex f000 = der_coeffs[sys.idx(s, ip_,     il_,      ik_     )].f;
            const dcomplex f100 = der_coeffs[sys.idx(s, ip_next, il_,      ik_     )].f;
            const dcomplex f010 = der_coeffs[sys.idx(s, ip_,     il_,      ik_next )].f;
            const dcomplex f110 = der_coeffs[sys.idx(s, ip_next, il_,      ik_next )].f;
            const dcomplex f001 = der_coeffs[sys.idx(s, ip_,     il_next,  ik_     )].f;
            const dcomplex f101 = der_coeffs[sys.idx(s, ip_next, il_next,  ik_     )].f;
            const dcomplex f011 = der_coeffs[sys.idx(s, ip_,     il_next,  ik_next )].f;
            const dcomplex f111 = der_coeffs[sys.idx(s, ip_next, il_next,  ik_next )].f;

            result[s] = w000 * f000 + w100 * f100
                    + w010 * f010 + w110 * f110
                    + w001 * f001 + w101 * f101
                    + w011 * f011 + w111 * f111;
        }
        return result;
    };
 
    // ==================================================================
    // Sigma builder. For sigmas whose samples weren't filled, the
    // resulting Sig* entries will be wrong but are never read.
    // ==================================================================
    auto build_sigmas = [&](const std::array<SigArr, N_GEOM>& samples,
                            const SigArr& f_here,
                            SigArr& Sig0, SigArr& Sigzsc,
                            SigArr& Sigp, SigArr& Sigm) {
        for (int s = 0; s < Nsig; ++s) {
            Sig0  [s] = 2.0*f_here[s] - samples[0][s] - samples[1][s];
            Sigzsc[s] = 2.0*f_here[s] - samples[2][s] - samples[3][s];
            Sigp  [s] =     f_here[s] - samples[4][s];
            Sigm  [s] =     f_here[s] - samples[5][s];
        }
    };
 
    // ==================================================================
    // Geometry + sampling. Computes the 6 (angle, k, l) targets and
    // calls get_fval for each with its per-point mask.
    // ==================================================================
    auto sample_geometry = [&](double p, double cos_th, double sin_th,
                               int il, double cos_psi, double sin_psi,
                               double k, double k2, double l, double l2,
                               double kl, std::array<SigArr, N_GEOM>& samples)
    {
        double p2 = p * p;
        double pk = p * k;
        double pl = p * l;
        double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;
 
        double Rp_m_k = std::sqrt(k2 + p2 - 2.0*pk*cos_th);
        double Rp_p_k = std::sqrt(k2 + p2 + 2.0*pk*cos_th);
        double Rp_m_l = (il == 0) ? p
                                  : std::sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);
 
        double R_k_zp       = std::sqrt(k2 + 4.0*z2*p2          - 4.0*pk*z*cos_th);
        double R_k_1zp      = std::sqrt(k2 + 4.0*one_z_2*p2     - 4.0*pk*one_z*cos_th);
        double R_k_m_2z_p   = std::sqrt(k2 + two_z_minus_1_2*p2 - 2.0*pk*two_z_minus_1*cos_th);
        double R_k_m_2_1z_p = std::sqrt(k2 + two_z_minus_1_2*p2 + 2.0*pk*two_z_minus_1*cos_th);
 
        auto angle_from_cos = [](double num, double denom) -> double {
            double c = num / (denom + 1e-12);
            return fast_acos(std::clamp(c, -1.0, 1.0));
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
        double angle_k_zp_l  = angle_from_cos(
            k*cos_psi - 2.0*z*p*cos_theta_minus_psi, R_k_zp);
        double angle_k_1zp_l = angle_from_cos(
            k*cos_psi - 2.0*one_z*p*cos_theta_minus_psi, R_k_1zp);
 
        samples[0] = get_fval(angle_pmk_lmp,    Rp_m_k,       Rp_m_l, mask[0]); 
        samples[1] = get_fval(angle_ppk_lmp,    Rp_p_k,       Rp_m_l, mask[1]);
        samples[2] = get_fval(angle_k_m_2z_p,   R_k_m_2z_p,   Rp_m_l, mask[2]);
        samples[3] = get_fval(angle_k_m_2_1z_p, R_k_m_2_1z_p, Rp_m_l, mask[3]);
        samples[4] = get_fval(angle_k_zp_l,     R_k_zp,       l,      mask[4]);
        samples[5] = get_fval(angle_k_1zp_l,    R_k_1zp,      l,      mask[5]);
    };
 
    // ==================================================================
    // Vertex-specific M-matrix 
    // ==================================================================
 
    // gamma -> q qbar
    auto M_qqbar = [&](const SigArr& Sig0, const SigArr& Sigzsc,
                       const SigArr& Sigp, const SigArr& Sigm) -> SigArr {
        SigArr out{};
        out[0] = 2.0*CF * Sig0[0];
        out[1] = + 2.0*CF * (Sigp[1] + Sigm[1]); //CA * (Sig0[0] - Sigzsc[0])
               
               
        if (!is_large_Nc) {
            out[0] += (1.0/CA) * (Sigzsc[0] - Sigp[0] - Sigm[0])
                    - (1.0/CA) * (Sigzsc[1] - Sigp[1] - Sigm[1]);
            out[1] += (1.0/CA) * (Sigzsc[1] - Sig0[1]);
        }
        return out;
    };
 
    // q -> q g
    auto M_qqg = [&](const SigArr& Sig0, const SigArr& Sigzsc,
                     const SigArr& Sigp, const SigArr& Sigm) -> SigArr {
        SigArr out{};
        out[0] = CA * (Sigm[0] + Sig0[0]);
        out[1] = CA * (-Sigzsc[0] + Sig0[0])
               + 2.0 * (CF * Sigp[1] + CA * Sigm[1]);
        if (!is_large_Nc) {
            out[0] += -Sigp[0]/CA + 1/(CA*CA*(CA*CA-1))*(Sigzsc[2] - Sig0[2]);
            out[2]  = CA*CA*(CA*CA-1) * (Sigzsc[0] - Sig0[0])
                    + CA * (Sigzsc[1] + Sig0[1] - 2.0*(Sigp[1] + Sigm[1]))
                    + (-(1.0/CA) * Sigp[2] + CA * (Sigzsc[2] + Sigm[2]));
        }
        return out;
    };
 
    // g -> g g
    auto M_ggg = [&](const SigArr& Sig0, const SigArr& Sigzsc,
                     const SigArr& Sigp, const SigArr& Sigm) -> SigArr {
        SigArr out{};
        out[0] = CA * (Sigm[0] + Sigp[0] + Sig0[0]);
        out[1] = CA * (-Sigzsc[0] + Sig0[0]) + 2.0 * CA * (Sigp[1] + Sigm[1]);
        if (!is_large_Nc) {
            throw std::runtime_error("ggg FNc (Nsig=6) M-matrix not yet implemented");
        }
        return out;
    };
 
    // ==================================================================
    // Quadrature kernel. Templated on M-matrix lambda.
    // ==================================================================
    auto run_quadrature = [&](auto&& M_matrix) {
 
        #pragma omp parallel for collapse(3)
        for (int ip = 0; ip < Npsi; ++ip) {
            for (int il = 0; il < Nl; ++il) {
                for (int ik = 0; ik < Nk; ++ik) {
 
                    double cos_psi = cos_psi_array[ip];
                    double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi*cos_psi));
                    double k  = K_array[ik];
                    double k2 = k * k;
                    double l  = L_array[il];
                    double l2 = l * l;
                    double kl = k * l;
 
                    // f at (ip, il, ik) for all sigma (cheap, always needed)
                    SigArr f_here{};
                    for (int s = 0; s < Nsig; ++s)
                        f_here[s] = fH0[sys.idx(s, ip, il, ik)];
 
                    // kinetic term
                    double kinetic = 2.0 * (k * l) / omega * cos_psi;
                    for (int s = 0; s < Nsig; ++s)
                        HF[sys.idx(s, ip, il, ik)] = kinetic * f_here[s];

                    
                    SigArr sum{};
 
                    if (half1 > 0.0) {
                        int Nsim = std::max(2, n_radial);
                        if (Nsim % 2 != 0) ++Nsim;
                        double h = (b1 - a1) / static_cast<double>(Nsim);
                        double cheb_w = M_PI / static_cast<double>(n_angular);
 
                        SigArr racc{};
 
                        for (int ii = 0; ii <= Nsim; ++ii) {
                            double p    = p1_nodes[ii];
                            // int coeff   = (ii == 0 || ii == Nsim) ? 1
                                        //: (ii % 2 == 1 ? 4 : 2);
                            double Vp   = V(2.0 * p, sp);
                            double w    = p1_weights[ii] * p * Vp;
 
                            SigArr tacc{};
 
                            for (int j = 0; j < n_angular; ++j) {
                                double cj = cos_theta[j];
                                double sj = sin_theta[j];
 
                                std::array<SigArr, N_GEOM> samples;
                                SigArr Sig0, Sigzsc, Sigp, Sigm;
 
                                // +theta
                                sample_geometry(p, cj, sj, il,
                                                cos_psi, sin_psi,
                                                k, k2, l, l2, kl, samples);
                                build_sigmas(samples, f_here,
                                             Sig0, Sigzsc, Sigp, Sigm);
                                SigArr d = M_matrix(Sig0, Sigzsc, Sigp, Sigm);
 
                                // -theta
                                sample_geometry(p, -cj, -sj, il,
                                                cos_psi, sin_psi,
                                                k, k2, l, l2, kl, samples);
                                build_sigmas(samples, f_here,
                                             Sig0, Sigzsc, Sigp, Sigm);
                                SigArr e = M_matrix(Sig0, Sigzsc, Sigp, Sigm);
 
                                for (int s = 0; s < Nsig; ++s)
                                    tacc[s] += d[s] + e[s];
                        }
                         for (int s = 0; s < Nsig; ++s)
                                racc[s] += w * cheb_w * tacc[s];
                        }
 
                        for (int s = 0; s < Nsig; ++s)
                            sum[s] += racc[s];
                    }
 
                    for (int s = 0; s < Nsig; ++s)
                        HF[sys.idx(s, ip, il, ik)] += prefac * sum[s];
                }
            }
        }
    };

    // auto run_quadrature = [&](auto&& M_matrix) {

    //     // ---- Radial quadrature setup ----
    //     // Composite midpoint on two sub-regions: [0, b1] and [b1, pmax].
    //     // Midpoint pairs naturally with the trilinear interpolant (both O(h²)),
    //     // avoids evaluating at cell boundaries, and trivially supports Richardson
    //     // extrapolation if you want to upgrade accuracy later.
    //     //
    //     // The two regions exist because the integrand V(2p)·p·f typically has
    //     // different characteristic scales near and far from p ~ mu. You can use
    //     // different resolutions in each: more nodes near the peak, fewer in the tail.

    //     const double a1 = 0.0;     // inner region start
    //     const double b1 = split;   // inner region end (= 6*mu or 20*mu, set above)

    //     // Resolution per region. Tune these if needed; N1 controls accuracy near
    //     // the dominant scale, N2 controls the slowly-converging tail.
    //     const int N1 = 32;   // inner region nodes

    //     const double h1 = (b1 - a1)   / static_cast<double>(N1);

    //     const double cheb_w = M_PI / static_cast<double>(n_angular);

    //     // ---- Helper: one full angular sweep at a given p, accumulating into tacc ----
    //     // Captures everything via [&] from the enclosing scope.
    //     auto angular_sweep = [&](double p, double cos_psi, double sin_psi,
    //                             int il, double k, double k2,
    //                             double l, double l2, double kl,
    //                             const SigArr& f_here,
    //                             SigArr& tacc) {
    //         for (int j = 0; j < n_angular; ++j) {
    //             const double cj = cos_theta[j];
    //             const double sj = sin_theta[j];

    //             std::array<SigArr, N_GEOM> samples;
    //             SigArr Sig0, Sigzsc, Sigp, Sigm;

    //             // +theta
    //             sample_geometry(p, cj, sj, il,
    //                             cos_psi, sin_psi,
    //                             k, k2, l, l2, kl, samples);
    //             build_sigmas(samples, f_here, Sig0, Sigzsc, Sigp, Sigm);
    //             SigArr d = M_matrix(Sig0, Sigzsc, Sigp, Sigm);

    //             // -theta
    //             sample_geometry(p, -cj, -sj, il,
    //                             cos_psi, sin_psi,
    //                             k, k2, l, l2, kl, samples);
    //             build_sigmas(samples, f_here, Sig0, Sigzsc, Sigp, Sigm);
    //             SigArr e = M_matrix(Sig0, Sigzsc, Sigp, Sigm);

    //             for (int s = 0; s < Nsig; ++s)
    //                 tacc[s] += d[s] + e[s];
    //         }
    //     };

    //     // ---- Main loop over outer grid points ----
    //     #pragma omp parallel for collapse(3)
    //     for (int ip = 0; ip < Npsi; ++ip) {
    //         for (int il = 0; il < Nl; ++il) {
    //             for (int ik = 0; ik < Nk; ++ik) {

    //                 const double cos_psi = cos_psi_array[ip];
    //                 const double sin_psi = std::sqrt(std::max(0.0, 1.0 - cos_psi*cos_psi));
    //                 const double k  = K_array[ik];
    //                 const double k2 = k * k;
    //                 const double l  = L_array[il];
    //                 const double l2 = l * l;
    //                 const double kl = k * l;

    //                 // f at (ip, il, ik) for all sigma -- needed for both the kinetic
    //                 // term and as the f(x, x) value in the Sigma builder.
    //                 SigArr f_here{};
    //                 for (int s = 0; s < Nsig; ++s)
    //                     f_here[s] = fH0[sys.idx(s, ip, il, ik)];

    //                 // Kinetic term
    //                 const double kinetic = 2.0 * (k * l) / omega * cos_psi;
    //                 for (int s = 0; s < Nsig; ++s)
    //                     HF[sys.idx(s, ip, il, ik)] = kinetic * f_here[s];

    //                 SigArr sum{};

    //                 // ===== Region 1: [a1, b1], composite midpoint =====
    //                 if (b1 > a1) {
    //                     SigArr racc{};

    //                     for (int ii = 0; ii < N1; ++ii) {
    //                         const double p  = a1 + (ii + 0.5) * h1;   // midpoint of subinterval
    //                         const double Vp = V(2.0 * p, sp);
    //                         const double w_p = h1 * p * Vp;            // midpoint weight × Jacobian

    //                         SigArr tacc{};
    //                         angular_sweep(p, cos_psi, sin_psi, il,
    //                                     k, k2, l, l2, kl, f_here, tacc);

    //                         for (int s = 0; s < Nsig; ++s)
    //                             racc[s] += w_p * cheb_w * tacc[s];
    //                     }

    //                     for (int s = 0; s < Nsig; ++s)
    //                         sum[s] += racc[s];
    //                 }

    //                 // ===== Region 2: [b1, pmax], composite midpoint =====
    //                 // This was silently dropped in the original code.

    //                 for (int s = 0; s < Nsig; ++s)
    //                     HF[sys.idx(s, ip, il, ik)] += prefac * sum[s];
    //             }
    //         }
    //     }
    // };
 
    if      (is_gamma_qqbar) run_quadrature(M_qqbar);
    else if (is_qqg)         run_quadrature(M_qqg);
    else if (is_ggg)         run_quadrature(M_ggg);
 
    // ---- enforce psi-independence at l = 0 for all sigma ----
    // for (int sig = 0; sig < Nsig; ++sig) {
    //     for (int ik = 0; ik < Nk; ++ik) {
    //         dcomplex avg = 0.0;
    //         for (int ip = 0; ip < Npsi; ++ip)
    //             avg += HF[sys.idx(sig, ip, 0, ik)];
    //         avg /= static_cast<double>(Npsi);
    //         for (int ip = 0; ip < Npsi; ++ip)
    //             HF[sys.idx(sig, ip, 0, ik)] = avg;
    //     }
    // }
 
    return HF;
}