#include "Physis_J.hpp"
#include "hamiltonian_j.hpp"

using namespace std;
using dcomplex = complex<double>;

const dcomplex I(0.0, 1.0);
const double sqrt_e = 1.648721270700128;
const double e = std::exp(1.0);
const double PI = 3.141592653589793;


double VHTL(double q, double mu)
{
    // yukawa potential (extra q factor due to polar coordinates jacobian) q / pow((q*q + mu*mu), 2)
    // htl 1 / (q*(q*q + (inv_sqrt_e * mu)*(inv_sqrt_e * mu)))
    return  1/PI * 1 / (q*(q*q + (e * mu*mu))); //q / pow((q*q + mu*mu), 2); 
}

double VHO(double q, double mu) { 
    double eps = mu; 
    double x = q / eps;
    double exp = std::exp(-x*x / 2.0);
    double fac = 0.25 * 1.0 / (2.0 * PI * pow(eps, 3)) * (x*x*x - 2.0 * x); 

    return exp * fac; }

double VYUK(double q, double mu)
{   
    // yukawa potential (extra q factor due to polar coordinates jacobian) q / pow((q*q + mu*mu), 2)
    // htl 1 / (q*(q*q + (inv_sqrt_e * mu)*(inv_sqrt_e * mu)))
 

    return  1/PI * q / pow((q*q + mu*mu), 2);
}

double factorial(int r)
{
    double f = 1.0;
    for (int i = 2; i <= r; ++i) f *= i;
    return f;
}

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
    string vertex)
    {
    const double dpt = (pM - pm) / Np;
    const double dth = 2.0 * PI / Ntheta;

    double result = 0.0;
    double rfact = factorial(r);


    auto V = [&mode](double p, double mu) -> double {
        if (mode == 0) {
            return VYUK(p, mu);
        } else if (mode == 1) {
            return VHTL(p, mu);}
            else{return VHO(p, mu);}
        };




    for (int i = 0; i < Np; ++i)
    {
        // midpoint rule in p~
        double ptilde = pm + (i + 0.5) * dpt;
        double Vval = V(ptilde, mu);

        for (int j = 0; j < Ntheta; ++j)
        {
            // midpoint rule in theta
            double theta = (j + 0.5) * dth;

            if (vertex == "q_qg") {
            // g(z) = 1
                {
                    double gz = 1.0;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += Vval * angular_part;
                    }
                }

                // g(z) = z with color factor (2*CF/CA - 1)
                {
                    double CF = 4.0/3.0;
                    double CA = 3.0;
                    double gz = z;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += (2.0 * CF / CA - 1.0) * Vval * angular_part;
                    }
                }

                // g(z) = 1 - z
                {
                    double gz = 1.0 - z;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += Vval * angular_part;
                    }
                } 
            } 
            
            else if (vertex == "g_gg"){
            // g(z) = 1
                {
                    double gz = 1.0;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += Vval * angular_part;
                    }
                }

                // g(z) = z with color factor (2*CF/CA - 1)
                {
                    double CF = 4.0/3.0;
                    double CA = 3.0;
                    double gz = z;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += Vval * angular_part;
                    }
                }

                // g(z) = 1 - z
                {
                    double gz = 1.0 - z;
                    double Rval = std::sqrt(p*p + ptilde*ptilde*gz*gz - 2.0*p*ptilde*gz*std::cos(theta));
                    if (Rval != 0.0) {
                    double numerator = std::pow(Rval - p, r);
                    double angular_part =
                        (numerator / (rfact * Rval)) *
                        (p - ptilde * gz * std::cos(theta));
                    result += Vval * angular_part;
                    }
                } 
            } 

            

            else {
                double Rval = std::sqrt(p*p + ptilde*ptilde - 2.0*p*ptilde*std::cos(theta));
                if (Rval == 0.0) continue;

                double numerator = std::pow(Rval - p, r);
                double angular_part =
                    (numerator / (rfact * Rval)) *
                    (p - ptilde * std::cos(theta));

                result +=  Vval * angular_part;
            }
        }
    }

    result *= dpt * dth;
    return result;
}


static void precompute_derivatives(
    const Physis_J& sys,
    const std::vector<dcomplex>& fH0,
    std::vector<dcomplex>& fx,
    std::vector<dcomplex>& fxx)
    {
        const int Np = sys.Np();
        const double delta_p = sys.Lp() / (Np - 1);
        const double inv2    = 1.0 / (2.0 * delta_p);
        const double inv2sq  = 1.0 / (delta_p * delta_p);
        
        // Check if the first grid point is at p = 0 (radial symmetry applies)
        const bool grid_at_origin = (std::abs(sys.P().front()) < 1e-14);
        
        fx.assign(fH0.size(), dcomplex(0.0, 0.0));
        fxx.assign(fH0.size(), dcomplex(0.0, 0.0));
        
        #pragma omp parallel for
        for (int ix = 0; ix < Np; ++ix) {
            if (ix > 0 && ix < Np - 1) {
                // Interior: 3-point central, O(Δp²)
                fx[ix]  = ( fH0[ix+1] - fH0[ix-1] ) * inv2;
                fxx[ix] = ( fH0[ix+1] - 2.0*fH0[ix] + fH0[ix-1] ) * inv2sq;
            }
            else if (ix == 0) {
                if (grid_at_origin) {
                    // Radial symmetry: f'(0) = 0, mirror gives O(Δp²) second derivative
                    fx[ix]  = dcomplex(0.0, 0.0);
                    if (Np >= 2)
                        fxx[ix] = 2.0 * ( fH0[1] - fH0[0] ) * inv2sq;
                    else
                        fxx[ix] = dcomplex(0.0, 0.0);
                } else {
                    // No symmetry: 3-point one-sided, O(Δp²)
                    if (Np >= 3) {
                        fx[ix]  = ( -3.0*fH0[0] + 4.0*fH0[1] - fH0[2] ) * inv2;
                        if (Np >= 4)
                            fxx[ix] = ( 2.0*fH0[0] - 5.0*fH0[1] + 4.0*fH0[2] - fH0[3] ) * inv2sq;
                        else
                            fxx[ix] = ( fH0[0] - 2.0*fH0[1] + fH0[2] ) * inv2sq;  // O(Δp)
                    } else {
                        fx[ix]  = ( fH0[1] - fH0[0] ) / delta_p;
                        fxx[ix] = dcomplex(0.0, 0.0);
                    }
                }
            }
            else { // ix == Np - 1
                if (Np >= 3) {
                    fx[ix]  = ( 3.0*fH0[Np-1] - 4.0*fH0[Np-2] + fH0[Np-3] ) * inv2;
                    if (Np >= 4)
                        fxx[ix] = ( 2.0*fH0[Np-1] - 5.0*fH0[Np-2] + 4.0*fH0[Np-3] - fH0[Np-4] ) * inv2sq;
                    else
                        fxx[ix] = ( fH0[Np-1] - 2.0*fH0[Np-2] + fH0[Np-3] ) * inv2sq;  // O(Δp)
                } else {
                    fx[ix]  = ( fH0[Np-1] - fH0[Np-2] ) / delta_p;
                    fxx[ix] = dcomplex(0.0, 0.0);
                }
            }
        }
    }

// Gauss-Legendre Quadrature struct


struct GaussLaguerre {
    vector<double> nodes;
    vector<double> weights;
    
    GaussLaguerre(int n) {
        compute_gauss_laguerre(n, nodes, weights);
    }
    
private:
    void compute_gauss_laguerre(int n, vector<double>& x, vector<double>& w) {
        x.resize(n);
        w.resize(n);
        
        // Newton iteration on Laguerre polynomial roots
        for (int i = 0; i < n; ++i) {
            // Initial guess (Stroud & Secrest)
            double z;
            if (i == 0) {
                z = 3.0 / (1.0 + 2.4 * n);
            } else if (i == 1) {
                z = x[0] + 15.0 / (1.0 + 2.5 * n);
            } else {
                double ai = i - 1;
                z = x[i-1] + (1.0 + 2.55*ai)/(1.9*ai) * (x[i-1] - x[i-2]);
            }
            
            // Newton iteration
            for (int iter = 0; iter < 50; ++iter) {
                // Evaluate L_n(z) and L_{n-1}(z) using recurrence
                double L1 = 1.0;      // L_0
                double L0 = 1.0 - z;  // L_1
                for (int k = 1; k < n; ++k) {
                    double Lk = ((2.0*k + 1.0 - z)*L0 - k*L1) / (k + 1.0);
                    L1 = L0;
                    L0 = Lk;
                }
                // L0 = L_n(z), L1 = L_{n-1}(z)
                // Derivative: n*(L_n - L_{n-1}) / (... ) ; use:
                //   L_n'(z) = (n*L_n(z) - n*L_{n-1}(z)) / z
                double Lp = n * (L0 - L1) / z;
                
                double z1 = z;
                z = z1 - L0 / Lp;
                
                if (std::abs(z - z1) < 1e-14 * z) break;
            }
            
            x[i] = z;
            
            // Weight: w_i = 1 / (z · [L_n'(z)]²) · (n!)² / n   — simplified:
            //   w_i = z / ((n+1)² · [L_{n+1}(z)]²)
            // Compute L_{n+1}(z) using one more recurrence step:
            double L1 = 1.0;
            double L0 = 1.0 - z;
            for (int k = 1; k <= n; ++k) {
                double Lk = ((2.0*k + 1.0 - z)*L0 - k*L1) / (k + 1.0);
                L1 = L0;
                L0 = Lk;
            }
            // Now L0 = L_{n+1}(z)
            w[i] = z / ((n + 1.0) * (n + 1.0) * L0 * L0);
        }
    }
};


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
static const GaussLegendre GL_RADIAL(30);  // 12-point for radial integration
static const GaussLegendre GL_ANGULAR(40); // 32-point for angular integration


vector<dcomplex> Hamiltonian_J(const Physis_J& sys, 
    const vector<dcomplex>& fH0,
    vector<double> taylor_coeffs_0,
    vector<double> taylor_coeffs_1,
    vector<double> taylor_coeffs_2) {

    // Extract constants once
    const int Np = sys.Np();
    const double Lp = sys.Lp();
    const double qtilde = sys.qtilde();
    const double omega = sys.omega();
    const double mu = sys.mu();
    const double z = sys.z();
    const int mode = sys.mode();
    const string& vertex = sys.vertex();
    const auto& Pgrid = sys.P();
    const string& Nc_mode = sys.Ncmode();
    const bool is_large_Nc = (Nc_mode == "LNc");
    
    vector<dcomplex> HF(fH0.size(), dcomplex(0.0, 0.0));

    const double CF = (is_large_Nc) ? 1.5 : 4.0/3.0;
    const double CA = 3.0;

    // Precompute derivatives 
    std::vector<dcomplex> fx, fxx;
    precompute_derivatives(sys, fH0, fx, fxx);

    // Integration parameters
    double pmin = sys.pmin(); 
    double pmax = sys.pmax();

    // Function pointer for potential
    double (*V_func)(double, double);
    if (mode == 0) {
        V_func = VYUK;
    } else if (mode == 1) {
        V_func = VHTL;
    } else {
        V_func = VHO;
    }

    const double C = (vertex == "gamma_qq") ? CF : 0.5 * CA;
    
    // Precompute grid spacing for interpolation
    const double dx = Pgrid[1] - Pgrid[0];
    const double inv_dx = 1.0 / dx;
    const double Pgrid_front = Pgrid.front();
    const double Pgrid_back = Pgrid.back();
    
    // Precompute last taylor coefficient
    const double taylor_back = taylor_coeffs_0.back();
    
    // Branch prediction: determine vertex type once
    const bool is_gamma_qq = (vertex == "gamma_qq");
    const bool is_asymmetric = (vertex == "q_qg" || vertex == "g_gg");
    
    if (!is_gamma_qq && !is_asymmetric) {
        throw runtime_error("Insert a valid vertex");
    }
    

    
    // Split domain parameters
    const double split = 1.4142;
    
    // (A)symmetric factor for q_qg/g_gg vertices
    const double asym_fac = (vertex == "q_qg") ? 2.0 * CF / CA - 1.0 : 1.0;
    const double z_complement = 1.0 - z;
    const double z_sq = z * z;
    const double z_comp_sq = z_complement * z_complement;

    // Gauss-Legendre quadrature setup
    const int n_radial = GL_RADIAL.nodes.size();
    const int n_angular = GL_ANGULAR.nodes.size();


    
    // Map Gauss-Legendre nodes from [-1,1] to integration domains
    // For radial: two regions [pmin, split] and [split, pmax]
    const double r1_mid = 0.5 * (split + pmin);
    const double r1_half = 0.5 * (split - pmin);
    const double r2_mid = 0.5 * (pmax + split);
    const double r2_half = 0.5 * (pmax - split);
    
    // For angular: [0, 2π]
    const double ang_mid = M_PI;
    const double ang_half = M_PI;
    
    // Precompute mapped nodes and weights
    vector<double> pt1_nodes(n_radial), pt1_weights(n_radial);
    vector<double> pt2_nodes(n_radial), pt2_weights(n_radial);
    vector<double> theta_nodes(n_angular), theta_weights(n_angular);
    vector<double> cos_theta(n_angular);
    
    for (int i = 0; i < n_radial; ++i) {
        pt1_nodes[i] = r1_mid + r1_half * GL_RADIAL.nodes[i];
        pt1_weights[i] = r1_half * GL_RADIAL.weights[i];

        pt2_nodes[i] = r2_mid + r2_half * GL_RADIAL.nodes[i];
        pt2_weights[i] = r2_half * GL_RADIAL.weights[i];
    }
    
    for (int j = 0; j < n_angular; ++j) {
        theta_nodes[j] = ang_mid + ang_half * GL_ANGULAR.nodes[j];
        theta_weights[j] = ang_half * GL_ANGULAR.weights[j];
        cos_theta[j] = std::cos(theta_nodes[j]);
    }



    // lambda function for interpolation
    auto sample_f = [&](double x) -> dcomplex {
        if (x <= Pgrid_front) return fH0.front();
        if (x >= Pgrid_back)  return fH0.back();

        const double xn = (x - Pgrid_front) * inv_dx;
        int j = static_cast<int>(xn + 0.5);   // nearest grid point
        if (j < 0)         j = 0;
        if (j > Np - 1)    j = Np - 1;

        const double h = x - Pgrid[j];        // signed, |h| ≤ dx/2

        return fH0[j] + h * fx[j] + 0.5 * h*h * fxx[j];
    };
    // allocate and fill real/imag arrays for spline construction
    std::vector<double> f_real(Np), f_imag(Np);
    
    for (int i = 0; i < Np; ++i) {
        f_real[i] = std::real(fH0[i]);
        f_imag[i] = std::imag(fH0[i]);   
    }

    tk::spline j_r(Pgrid, f_real);
    tk::spline j_i(Pgrid, f_imag);

    #pragma omp parallel for schedule(dynamic, 8) if(Np > 64)
    for (int ix = 0; ix < Np; ++ix) {
        // Kinetic term

        const double p = Pgrid[ix];
        const double p_sq = p * p;
        const double factor = p_sq / (2.0 * omega);
        HF[ix] = fH0[ix] * factor;

        // Taylor expansion contribution
        const dcomplex conv = taylor_coeffs_0[ix] * fH0[ix] 
                            + taylor_coeffs_1[ix] * fx[ix] 
                            + taylor_coeffs_2[ix] * fxx[ix];
        
        const dcomplex vterm = taylor_back * fH0[ix] - conv;
        //const dcomplex vterm = 0;
        HF[ix] += -I * qtilde * C * vterm;

        // Integration contribution

        const dcomplex f = fH0[ix];
        dcomplex sum_corr(0.0, 0.0);

        if (ix == 0) continue;


        if (is_gamma_qq) {
            const int N_p = 61;       // radial points (must be odd)
            
            
            auto simpson_weight = [](int i, int N) -> double {
                if (i == 0 || i == N - 1) return 1.0;
                return (i % 2 == 1) ? 4.0 : 2.0;
            };

            // Identify which radial node (if any) coincides with the external p.
            // Pgrid is uniform with spacing h_p1 starting at pmin, so node i is at pmin + i*h_p1.
            double h_p1 = (pmax - pmin) / (N_p - 1);
            const double pp_diag_real = (p - pmin) / h_p1;
            
            // Region 1: [pmin, pmax]  — skip i == i_diag if present
            for (int i = 0; i < N_p; ++i) {


                const double pt = pmin + i * h_p1;
                const double pt_sq = pt * pt;
                const double Vval = V_func(pt, mu);
                const double w_r = simpson_weight(i, N_p);
                const double radial_weight = (w_r * h_p1 / 3.0);

                for (int j = 0; j < n_angular; ++j) {
                    const double cos_t = cos_theta[j];
                    const double w_th = theta_weights[j];
                    const double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                    if (Rval == 0.0) continue;
                    const dcomplex fpt = sample_f(Rval);
                    const double ang_kernel = (p - pt * cos_t);
                    dcomplex integ_corr = Vval * ang_kernel * (f - fpt) * radial_weight * w_th;
                    sum_corr += 1.0/p * integ_corr;
                }
            }


            HF[ix] += -I * qtilde * C * sum_corr;

        } else { // q_qg or g_gg

                // Replicate the gamma_qq integration pattern (Simpson over radial, GL over angle),
                // but keep the three g(z) contributions and apply the color/asymmetry factor
                // to the z-term (as in the original vertex logic).
                const int N_p = 161; // radial Simpson points (must be odd)
                auto simpson_weight = [](int i, int N) -> double {
                    if (i == 0 || i == N - 1) return 1.0;
                    return (i % 2 == 1) ? 4.0 : 2.0;
                };
                double h_p1 = (pmax - pmin) / (N_p - 1);

                for (int i = 0; i < N_p; ++i) {
                    const double pt = pmin + i * h_p1;
                    const double pt_sq = pt * pt;
                    const double Vval = V_func(pt, mu);
                    const double w_r = simpson_weight(i, N_p);
                    const double radial_weight = (w_r * h_p1 / 3.0);

                    for (int j = 0; j < n_angular; ++j) {
                        const double cos_t = cos_theta[j];
                        const double w_th = theta_weights[j];

                        // g(z) = 1
                        {
                            double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                            if (Rval > 0.0) {
                                const dcomplex fpt = sample_f(Rval);
                                const double ang_kernel = (p - pt * cos_t);
                                dcomplex integ_corr = Vval * ang_kernel * (f - fpt) * radial_weight * w_th;
                                sum_corr += (1.0 / p) * integ_corr;
                            }
                        }

                        // g(z) = z  (apply asymmetry/color factor here)
                        {
                            double Rval = std::sqrt(p_sq + pt_sq * z_sq - 2.0 * p * pt * z * cos_t);
                            if (Rval > 0.0) {
                                const dcomplex fpt = sample_f(Rval);
                                const double ang_kernel = (p - pt * z * cos_t);
                                dcomplex integ_corr = asym_fac * Vval * ang_kernel * (f - fpt) * radial_weight * w_th;
                                sum_corr += (1.0 / p) * integ_corr;
                            }
                        }

                        // g(z) = 1 - z
                        {
                            double Rval = std::sqrt(p_sq + pt_sq * z_comp_sq - 2.0 * p * pt * z_complement * cos_t);
                            if (Rval > 0.0) {
                                const dcomplex fpt = sample_f(Rval);
                                const double ang_kernel = (p - pt * z_complement * cos_t);
                                dcomplex integ_corr = Vval * ang_kernel * (f - fpt) * radial_weight * w_th;
                                sum_corr += (1.0 / p) * integ_corr;
                            }
                        }
                    }
                }


            HF[ix] += -I * qtilde * C * sum_corr;
        }
    }
    
    return HF;
}
