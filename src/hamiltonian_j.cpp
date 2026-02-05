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
    return  1 / (PI) * 1 / (q*(q*q + (e * mu*mu))); //q / pow((q*q + mu*mu), 2); 
}

double VHO(double q, double mu)
{   double eps = mu;
    double exp = std::exp(-q*q / (2 * eps*eps));
    double fac = 0.25 * 1.0 / (2.0 * PI *pow(eps, 4)) * (q*q*q / (eps*eps)  - 2.0 * q);


    return exp * fac;
}

double VYUK(double q, double mu)
{   double eps = mu;
    // yukawa potential (extra q factor due to polar coordinates jacobian) q / pow((q*q + mu*mu), 2)
    // htl 1 / (q*(q*q + (inv_sqrt_e * mu)*(inv_sqrt_e * mu)))
 

    return  1 / (PI) * q / pow((q*q + mu*mu), 2);
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

    int Np = sys.Np();
    double delta_p = sys.Lp() / (Np - 1);
    double inv2 = 1.0 / (2.0 * delta_p);
    double inv2sq = 1.0 / (delta_p * delta_p);

    // allocate vectors sized as fH0
    fx.assign(fH0.size(), dcomplex(0.0,0.0));
    fxx.assign(fH0.size(), dcomplex(0.0,0.0));


    // compute central differences; handle edges with one-sided
    #pragma omp parallel for 
        for (int ix = 0; ix < Np; ++ix) {

            // fx: d/dx (along ix)
            if (ix > 0 && ix < Np - 1) {
                fx[ix] = ( fH0[ ix+1 ] - fH0[ ix-1 ] ) * inv2;
                fxx[ix] = ( fH0[ ix+1 ] - 2.0 * fH0[ix] + fH0[ ix-1 ] ) * inv2sq;

            } else if (ix == 0) {
                // forward difference for fx, second derivative via forward formula
                fx[ix] = ( fH0[ ix+1 ] - fH0[ix] ) / delta_p;
                if (Np > 2)
                    fxx[ix] = ( fH0[ ix+2 ] - 2.0 * fH0[ ix+1 ] + fH0[ix] ) * inv2sq;
                else fxx[ix] = dcomplex(0.0,0.0);
            } else { // ix == Np-1
                fx[ix] = ( fH0[ix] - fH0[ ix-1 ] ) / delta_p;
                if (Np > 2)
                    fxx[ix] = ( fH0[ix] - 2.0 * fH0[ ix-1 ] + fH0[ ix-2 ] ) * inv2sq;
                else fxx[ix] = dcomplex(0.0,0.0);
            }

        }
    }

// Gauss-Legendre Quadrature struct
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
static const GaussLegendre GL_RADIAL(3);  // 3-point for radial integration
static const GaussLegendre GL_ANGULAR(3); // 3-point for angular integration


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
    
    vector<dcomplex> HF(fH0.size(), dcomplex(0.0, 0.0));

    const double CF = 4.0 / 3.0;
    const double CA = 3.0;

    // Precompute derivatives 
    std::vector<dcomplex> fx, fxx;
    precompute_derivatives(sys, fH0, fx, fxx);

    // Integration parameters
    const double pmin = sys.pmin(); 
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
    
    // Adjust pmax if needed
    const bool do_integration = (pmin < Pgrid_back);
    if (do_integration && pmax > Pgrid_back) {
        pmax = Pgrid_back;
    }
    
    // Split domain parameters
    const double split = 4.0 * mu;
    
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
        if (x >= Pgrid_back) return fH0.back();
        
        int j = int((x - Pgrid_front) * inv_dx);
        j = std::min(std::max(j, 0), Np - 2);
        
        double t = (x - Pgrid[j]) * inv_dx;
        return fH0[j] * (1.0 - t) + fH0[j + 1] * t;
    };

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
        HF[ix] += -I * qtilde * C * vterm;

        // Integration contribution
        if (!do_integration || pmin >= pmax) continue;

        const dcomplex f = fH0[ix];
        dcomplex sum_corr(0.0, 0.0);

        if (is_gamma_qq) {
            // Gauss-Legendre integration over two radial regions
            
            // Region 1: [pmin, split]
            for (int i = 0; i < n_radial; ++i) {
                const double pt = pt1_nodes[i];
                const double pt_sq = pt * pt;
                const double Vval = V_func(pt, mu);
                const double w_p = pt1_weights[i];
                
                for (int j = 0; j < n_angular; ++j) {
                    const double cos_t = cos_theta[j];
                    const double w_th = theta_weights[j];
                    
                    const double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                    if (Rval == 0.0) continue;
                    
                    const dcomplex fpt = sample_f(Rval);
                    const double ang_kernel = (1.0 / Rval) * (p - pt * cos_t);
                    sum_corr += Vval * ang_kernel * (f - fpt) * w_p * w_th;
                }
            }

            // Region 2: [split, pmax]
            for (int i = 0; i < n_radial; ++i) {
                const double pt = pt2_nodes[i];
                const double pt_sq = pt * pt;
                const double Vval = V_func(pt, mu);
                const double w_p = pt2_weights[i];
                
                for (int j = 0; j < n_angular; ++j) {
                    const double cos_t = cos_theta[j];
                    const double w_th = theta_weights[j];
                    
                    const double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                    if (Rval == 0.0) continue;
                    
                    const dcomplex fpt = sample_f(Rval);
                    const double ang_kernel = (1.0 / Rval) * (p - pt * cos_t);
                    sum_corr += Vval * ang_kernel * (f - fpt) * w_p * w_th;
                }
            }

            HF[ix] += -I * qtilde * CF * sum_corr;
            
        } else { // q_qg or g_gg
            
            // Region 1: [pmin, split]
            for (int i = 0; i < n_radial; ++i) {
                const double pt = pt1_nodes[i];
                const double pt_sq = pt * pt;
                const double Vval = V_func(pt, mu);
                const double w_p = pt1_weights[i];
                
                for (int j = 0; j < n_angular; ++j) {
                    const double cos_t = cos_theta[j];
                    const double w_th = theta_weights[j];
                    const double w_total = w_p * w_th;
                    
                    // g(z) = 1
                    double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * cos_t);
                        sum_corr += Vval * ang_kernel * (f - fpt) * w_total;
                    }

                    // g(z) = (1-z)
                    Rval = std::sqrt(p_sq + pt_sq * z_comp_sq - 2.0 * p * pt * z_complement * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * z_complement * cos_t);
                        // what changes between qqg and ggg is the asym_fac
                        sum_corr += asym_fac * Vval * ang_kernel * (f - fpt) * w_total;
                    }

                    // g(z) = z
                    Rval = std::sqrt(p_sq + pt_sq * z_sq - 2.0 * p * pt * z * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * z * cos_t);
                        sum_corr += Vval * ang_kernel * (f - fpt) * w_total;
                    }
                }
            }

            // Region 2: [split, pmax]
            for (int i = 0; i < n_radial; ++i) {
                const double pt = pt2_nodes[i];
                const double pt_sq = pt * pt;
                const double Vval = V_func(pt, mu);
                const double w_p = pt2_weights[i];
                
                for (int j = 0; j < n_angular; ++j) {
                    const double cos_t = cos_theta[j];
                    const double w_th = theta_weights[j];
                    const double w_total = w_p * w_th;
                    
                    // g(z) = 1
                    double Rval = std::sqrt(p_sq + pt_sq - 2.0 * p * pt * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * cos_t);
                        sum_corr += Vval * ang_kernel * (f - fpt) * w_total;
                    }

                    // g(z) = (1-z)
                    Rval = std::sqrt(p_sq + pt_sq * z_comp_sq - 2.0 * p * pt * z_complement * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * z_complement * cos_t);
                        sum_corr += asym_fac * Vval * ang_kernel * (f - fpt) * w_total;
                    }

                    // g(z) = z
                    Rval = std::sqrt(p_sq + pt_sq * z_sq - 2.0 * p * pt * z * cos_t);
                    if (Rval > 0.0) {
                        const dcomplex fpt = sample_f(Rval);
                        const double ang_kernel = (1.0 / Rval) * (p - pt * z * cos_t);
                        sum_corr += Vval * ang_kernel * (f - fpt) * w_total;
                    }
                }
            }

            HF[ix] += -I * qtilde * 0.5 * CA * sum_corr;
        }
    }
    
    return HF;
}
