#include "hamiltonian.hpp"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <cmath>
#include <stdexcept>

using namespace std;
using dcomplex_d = thrust::complex<double>;

__device__ constexpr double PI = 3.141592653589793;

// --------------------- Device potential functions ---------------------
__device__ double VYUK_eff(double q, double mu) {
    return 1.0 / (PI) * q / pow((4.0*q*q + mu*mu), 2);
}

__device__ double VHTL_eff(double q, double mu) {
    const double e = 2.718281828459045;
    return 1.0 / (PI) * 1.0 / (4.0*q*(4.0*q*q + (e*mu*mu)));
}

__device__ double VHO_eff(double q, double mu) {
    double eps = mu;
    double exp_factor = exp(q*q / (2*eps*eps));
    double fac = 0.25 / (2.0 * PI * pow(eps, 4)) * (q*q*q/ (eps*eps) - 2.0*q);
    return exp_factor * fac;
}

__device__ double V_eff(int mode, double q, double mu) {
    if(mode == 0) return VYUK_eff(q, mu);
    if(mode == 1) return VHTL_eff(q, mu);
    return VHO_eff(q, mu);
}

__device__ std::size_t IDX(int m_sig, int ip, int il, int ik, int Npsi, int Nl, int Nk) {
    return (((m_sig * Npsi + ip) * Nl + il) * Nk + ik);
}

// --------------------- Device trilinear sampler ---------------------
__device__ dcomplex_d sample_trilinear_device(
    const dcomplex_d* fH0,
    int sig,
    double psi, double k, double l,
    int Npsi, int Nk, int Nl,
    double psi_min, double psi_max,
    double k_min, double k_max,
    double l_min, double l_max,
    double inv_dpsi, double inv_dk, double inv_dl,
    const double* psi_array,
    const double* K_array,
    const double* L_array
) {
    // Boundary checks - clamp to edges
    if (psi < psi_min) psi = psi_min;
    if (psi > psi_max) psi = psi_max;
    if (k < k_min) k = k_min;
    if (k > k_max) k = k_max;
    if (l < l_min) l = l_min;
    if (l > l_max) l = l_max;

    // Find cell indices
    int ip = int((psi - psi_min) * inv_dpsi);
    int ik = int((k   - k_min)   * inv_dk);
    int il = int((l   - l_min)   * inv_dl);

    // Clamp to valid range [0, N-2]
    if (ip < 0) ip = 0;
    if (ip > Npsi - 2) ip = Npsi - 2;
    if (ik < 0) ik = 0;
    if (ik > Nk - 2) ik = Nk - 2;
    if (il < 0) il = 0;
    if (il > Nl - 2) il = Nl - 2;

    // Compute normalized distances within cell [0, 1]
    const double t_psi = (psi - psi_array[ip]) / (psi_array[ip+1] - psi_array[ip]);
    const double t_k   = (k   - K_array[ik])   / (K_array[ik+1]   - K_array[ik]);
    const double t_l   = (l   - L_array[il])   / (L_array[il+1]   - L_array[il]);

    // Get 8 corner values
    const std::size_t base000 = IDX(sig, ip, il, ik, Npsi, Nl, Nk);
    const std::size_t base100 = IDX(sig, ip+1, il, ik, Npsi, Nl, Nk);
    const std::size_t base010 = IDX(sig, ip, il+1, ik, Npsi, Nl, Nk);
    const std::size_t base110 = IDX(sig, ip+1, il+1, ik, Npsi, Nl, Nk);
    const std::size_t base001 = IDX(sig, ip, il, ik+1, Npsi, Nl, Nk);
    const std::size_t base101 = IDX(sig, ip+1, il, ik+1, Npsi, Nl, Nk);
    const std::size_t base011 = IDX(sig, ip, il+1, ik+1, Npsi, Nl, Nk);
    const std::size_t base111 = IDX(sig, ip+1, il+1, ik+1, Npsi, Nl, Nk);

    const dcomplex_d f000 = fH0[base000];
    const dcomplex_d f100 = fH0[base100];
    const dcomplex_d f010 = fH0[base010];
    const dcomplex_d f110 = fH0[base110];
    const dcomplex_d f001 = fH0[base001];
    const dcomplex_d f101 = fH0[base101];
    const dcomplex_d f011 = fH0[base011];
    const dcomplex_d f111 = fH0[base111];

    // Trilinear interpolation
    const dcomplex_d f00 = f000 * (1.0 - t_psi) + f100 * t_psi;
    const dcomplex_d f10 = f010 * (1.0 - t_psi) + f110 * t_psi;
    const dcomplex_d f01 = f001 * (1.0 - t_psi) + f101 * t_psi;
    const dcomplex_d f11 = f011 * (1.0 - t_psi) + f111 * t_psi;

    const dcomplex_d f0 = f00 * (1.0 - t_k) + f10 * t_k;
    const dcomplex_d f1 = f01 * (1.0 - t_k) + f11 * t_k;

    return f0 * (1.0 - t_l) + f1 * t_l;
}

// --------------------- GPU Kernel ---------------------
__global__ void hamiltonian_kernel(
    const dcomplex_d* fH0,
    dcomplex_d* HF,
    int Npsi, int Nk, int Nl,
    const double* psi_array,
    const double* cos_psi_array,
    const double* K_array,
    const double* L_array,
    double delta_psi, double delta_k, double delta_l,
    double omega, double qtilde, double mu,
    int mode,
    bool is_large_Nc,
    double pmin, double pmax, double z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total = Npsi*Nk*Nl*2; // 2 sig

    if(idx >= total) return;

    int sig = idx / (Npsi*Nk*Nl);
    int rem = idx % (Npsi*Nk*Nl);
    int ip = rem / (Nk*Nl);
    int il = (rem / Nk) % Nl;
    int ik = rem % Nk;

    const std::size_t base = IDX(sig, ip, il, ik, Npsi, Nl, Nk);

    double psi = psi_array[ip];
    double cos_psi = cos_psi_array[ip];
    double sin_psi = sqrt(fmax(0.0, 1.0 - cos_psi*cos_psi));
    double k = K_array[ik], l = L_array[il];
    double k2 = k*k, l2 = l*l, kl = k*l;

    dcomplex_d f = fH0[base];
    double inv_dpsi = 1.0/delta_psi;
    double inv_dk = 1.0/delta_k;
    double inv_dl = 1.0/delta_l;

    // Kinetic term
    if (il == 0) {
        HF[base] = dcomplex_d(0.0, 0.0);
    } else {
        HF[base] = dcomplex_d(k*l*cos_psi/omega, 0.0) * f;
    }

    if(!is_large_Nc) return;

    // prefactor
    dcomplex_d prefac = dcomplex_d(0.0, -2.0 * qtilde / PI);

    // --- precompute Gauss-Legendre nodes and weights (n=4 radial, n=10 angular) ---
    const int n_radial = 4;
    const int n_angular = 10;

    const double gl4_nodes[4] = { -0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526 };
    const double gl4_weights[4] = { 0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538 };

    const double gl10_nodes[10] = {
        -0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163122,
         0.14887433898163122,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717
    };
    const double gl10_weights[10] = {
        0.06667134430868814, 0.1494513491505806, 0.21908636251598204, 0.26926671930999635, 0.2955242247147529,
        0.2955242247147529,  0.26926671930999635, 0.21908636251598204, 0.1494513491505806, 0.06667134430868814
    };

    // map Gauss-Legendre nodes from [-1,1] to integration domains
    double split = 1.0 * mu;

    double a1 = pmin;
    double b1 = fmin(split, pmax);
    double a2 = fmax(split, pmin);
    double b2 = pmax;

    double mid1 = 0.5 * (b1 + a1);
    double half1 = 0.5 * (b1 - a1);
    double mid2 = 0.5 * (b2 + a2);
    double half2 = 0.5 * (b2 - a2);

    double p1_nodes[4];
    double p1_weights[4];
    double p2_nodes[4];
    double p2_weights[4];

    for (int i = 0; i < n_radial; ++i) {
        p1_nodes[i] = mid1 + half1 * gl4_nodes[i];
        p1_weights[i] = half1 * gl4_weights[i];
        p2_nodes[i] = mid2 + half2 * gl4_nodes[i];
        p2_weights[i] = half2 * gl4_weights[i];
    }

    double theta_nodes[10];
    double theta_weights[10];
    double cos_theta[10];
    double sin_theta[10];

    for (int j = 0; j < n_angular; ++j) {
        theta_nodes[j] = M_PI + M_PI * (gl10_nodes[j]);
        theta_weights[j] = M_PI * gl10_weights[j];
        cos_theta[j] = cos(theta_nodes[j]);
        sin_theta[j] = sin(theta_nodes[j]);
    }

    // some other precomputed constants
    double z2 = z * z;
    double one_z = 1.0 - z;
    double one_z_2 = one_z * one_z;
    double two_z_minus_1 = 2.0 * z - 1.0;
    double two_z_minus_1_2 = two_z_minus_1 * two_z_minus_1;

    double CF = is_large_Nc ? 1.5 : 4.0/3.0;
    double CA = 3.0;

    // Integration accumulator
    dcomplex_d sum_integral(0.0, 0.0);

    if (sig == 0) {
        // sig = 0 integration
        // region 1
        if (half1 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                double p = p1_nodes[i];
                double w_p = p1_weights[i];
                double p2 = p*p;
                for (int j = 0; j < n_angular; ++j) {
                    double cos_th = cos_theta[j];
                    double sin_th = sin_theta[j];
                    double w_th = theta_weights[j];

                    double kp = k * p;
                    double lp = l * p;
                    double Vp = V_eff(mode, p, mu);
                    double w_total = w_p * w_th;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    double Rp_m_k = sqrt(k2 + p2 - 2.0*kp*cos_th);
                    double Rp_p_k = sqrt(k2 + p2 + 2.0*kp*cos_th);
                    double Rp_m_l = (il == 0) ? p : sqrt(l2 + p2 - 2.0*lp*cos_theta_minus_psi);

                    double numer_pmk_lmp = (p2 - kp * cos_th - lp * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_lmp = fmin(fmax(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-7), -1.0), 1.0);
                    double angle_pmk_lmp = acos(cos_val_pmk_lmp);

                    double numer_ppk_lmp = (-(p2 + kp * cos_th - lp * cos_theta_minus_psi) + kl * cos_psi);
                    double cos_val_ppk_lmp = fmin(fmax(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-7), -1.0), 1.0);
                    double angle_ppk_lmp = acos(cos_val_ppk_lmp);

                    bool in_domain_1 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool in_domain_2 = (Rp_p_k >= K_array[0] && Rp_p_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);

                    if (in_domain_1 && in_domain_2) {
                        dcomplex_d f0_kmp_lmp = sample_trilinear_device(fH0, 0, angle_pmk_lmp, Rp_m_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_kpp_lmp = sample_trilinear_device(fH0, 0, angle_ppk_lmp, Rp_p_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);
                        
                        dcomplex_d S0_contrib = dcomplex_d(2.0 * CF, 0.0) * (dcomplex_d(2.0, 0.0) * f - f0_kpp_lmp - f0_kmp_lmp);
                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * S0_contrib;
                    }
                }
            }
        }

        // region 2
        if (half2 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                double p = p2_nodes[i];
                double w_p = p2_weights[i];
                double p2 = p*p;
                for (int j = 0; j < n_angular; ++j) {
                    double cos_th = cos_theta[j];
                    double sin_th = sin_theta[j];
                    double w_th = theta_weights[j];

                    double kp = k * p;
                    double lp = l * p;
                    double Vp = V_eff(mode, p, mu);
                    double w_total = w_p * w_th;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    double Rp_m_k = sqrt(k2 + p2 - 2.0*kp*cos_th);
                    double Rp_p_k = sqrt(k2 + p2 + 2.0*kp*cos_th);
                    double Rp_m_l = (il == 0) ? p : sqrt(l2 + p2 - 2.0*lp*cos_theta_minus_psi);

                    double numer_pmk_lmp = (p2 - kp * cos_th - lp * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_lmp = fmin(fmax(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-7), -1.0), 1.0);
                    double angle_pmk_lmp = acos(cos_val_pmk_lmp);

                    double numer_ppk_lmp = (-(p2 + kp * cos_th - lp * cos_theta_minus_psi) + kl * cos_psi);
                    double cos_val_ppk_lmp = fmin(fmax(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-7), -1.0), 1.0);
                    double angle_ppk_lmp = acos(cos_val_ppk_lmp);

                    bool in_domain_1 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool in_domain_2 = (Rp_p_k >= K_array[0] && Rp_p_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);

                    if (in_domain_1 && in_domain_2) {
                        dcomplex_d f0_kmp_lmp = sample_trilinear_device(fH0, 0, angle_pmk_lmp, Rp_m_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_kpp_lmp = sample_trilinear_device(fH0, 0, angle_ppk_lmp, Rp_p_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d S0_contrib = dcomplex_d(2.0 * CF, 0.0) * (dcomplex_d(2.0, 0.0) * f - f0_kpp_lmp - f0_kmp_lmp);

                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * S0_contrib;
                    }
                }
            }
        }

        // add result
        HF[base] += prefac * sum_integral;

    } else if (sig == 1) {
        // sig = 1 integration (M10 + M11)
        dcomplex_d f_ = f;

        if (half1 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                double p = p1_nodes[i];
                double w_p = p1_weights[i];
                double p2 = p*p;
                for (int j = 0; j < n_angular; ++j) {
                    double cos_th = cos_theta[j];
                    double sin_th = sin_theta[j];
                    double w_th = theta_weights[j];

                    double pl = p * l;
                    double pk = p * k;
                    double Vp = V_eff(mode, p, mu);
                    double w_total = w_p * w_th;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    double Rp_m_k = sqrt(k2 + p2 - 2.0*pk*cos_th);
                    double Rp_p_k = sqrt(k2 + p2 + 2.0*pk*cos_th);
                    double Rp_m_l = (il == 0) ? p : sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);
                    double R_l_m_2z_p = (il == 0) ? fabs(two_z_minus_1) * p : sqrt(l2 + two_z_minus_1_2*p2 - 2.0*pl*two_z_minus_1*cos_theta_minus_psi);
                    double R_l_m_2_1z_p = (il == 0) ? one_z * p : sqrt(l2 + one_z_2*p2 - 2.0*pl*one_z*cos_theta_minus_psi);

                    double numer_pmk_lmp = (p2 - pk * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_lmp = fmin(fmax(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-12), -1.0), 1.0);
                    double angle_pmk_lmp = acos(cos_val_pmk_lmp);

                    double numer_ppk_lmp = (-(p2 + pk * cos_th - pl * cos_theta_minus_psi) + kl * cos_psi);
                    double cos_val_ppk_lmp = fmin(fmax(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-12), -1.0), 1.0);
                    double angle_ppk_lmp = acos(cos_val_ppk_lmp);

                    double numer_pmk_l2zp = (p2 - pk * cos_th - pl * two_z_minus_1 * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_l2zp = fmin(fmax(numer_pmk_l2zp / (Rp_m_k * R_l_m_2z_p + 1e-12), -1.0), 1.0);
                    double angle_pmk_l2zp = acos(cos_val_pmk_l2zp);

                    double numer_pmk_l2_1zp = (p2 - pk * cos_th - pl * one_z * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_l2_1zp = fmin(fmax(numer_pmk_l2_1zp / (Rp_m_k * R_l_m_2_1z_p + 1e-12), -1.0), 1.0);
                    double angle_pmk_l2_1zp = acos(cos_val_pmk_l2_1zp);

                    bool d1 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool d2 = (Rp_p_k >= K_array[0] && Rp_p_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool d3 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && R_l_m_2z_p >= L_array[0] && R_l_m_2z_p <= L_array[Nl-1]);
                    bool d4 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && R_l_m_2_1z_p >= L_array[0] && R_l_m_2_1z_p <= L_array[Nl-1]);

                    if (d1 && d2 && d3 && d4) {
                        dcomplex_d f0_kmp_lmp = sample_trilinear_device(fH0, 0, angle_pmk_lmp, Rp_m_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_kpp_lmp = sample_trilinear_device(fH0, 0, angle_ppk_lmp, Rp_p_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_pmk_l2zp = sample_trilinear_device(fH0, 0, angle_pmk_l2zp, Rp_m_k, R_l_m_2z_p, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_pmk_l2_1zp = sample_trilinear_device(fH0, 0, angle_pmk_l2_1zp, Rp_m_k, R_l_m_2_1z_p, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d M10_contrib = dcomplex_d(CA, 0.0) * (f0_pmk_l2zp + f0_pmk_l2_1zp - f0_kmp_lmp - f0_kpp_lmp);

                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * M10_contrib;
                    }

                    // M11 part: check and add
                    double R_k_zp = sqrt(k2 + 4.0*z2*p2 - 4.0*pk*z*cos_th);
                    double R_k_1zp = sqrt(k2 + 4.0*one_z_2*p2 - 4.0*pk*one_z*cos_th);

                    double numer_k_zp_l = (kl * cos_psi - 2.0*z*pl*cos_theta_minus_psi);
                    double cos_val_k_zp_l = fmin(fmax(numer_k_zp_l / (R_k_zp * l + 1e-12), -1.0), 1.0);
                    double angle_k_zp_l = (il > 0) ? acos(cos_val_k_zp_l) : 0.0;

                    double numer_k_1zp_l = (kl * cos_psi - 2.0*one_z*pl*cos_theta_minus_psi);
                    double cos_val_k_1zp_l = fmin(fmax(numer_k_1zp_l / (R_k_1zp * l + 1e-12), -1.0), 1.0);
                    double angle_k_1zp_l = (il > 0) ? acos(cos_val_k_1zp_l) : 0.0;

                    bool d5 = (R_k_zp >= K_array[0] && R_k_zp <= K_array[Nk-1]);
                    bool d6 = (R_k_1zp >= K_array[0] && R_k_1zp <= K_array[Nk-1]);

                    if (d5 && d6) {

                        dcomplex_d f1_k_zp_l = sample_trilinear_device(fH0, 1, angle_k_zp_l, R_k_zp, l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f1_k_1zp_l = sample_trilinear_device(fH0, 1, angle_k_1zp_l, R_k_1zp, l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d M11_contrib = dcomplex_d(2.0 * CF, 0.0) * (dcomplex_d(2.0, 0.0) * f_ - f1_k_zp_l - f1_k_1zp_l);

                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * M11_contrib;
                    }
                }
            }
        }

        if (half2 > 0.0) {
            for (int i = 0; i < n_radial; ++i) {
                double p = p2_nodes[i];
                double w_p = p2_weights[i];
                double p2 = p*p;
                for (int j = 0; j < n_angular; ++j) {
                    double cos_th = cos_theta[j];
                    double sin_th = sin_theta[j];
                    double w_th = theta_weights[j];

                    double pl = p * l;
                    double pk = p * k;
                    double Vp = V_eff(mode, p, mu);
                    double w_total = w_p * w_th;
                    double cos_theta_minus_psi = cos_th * cos_psi + sin_th * sin_psi;

                    double Rp_m_k = sqrt(k2 + p2 - 2.0*pk*cos_th);
                    double Rp_p_k = sqrt(k2 + p2 + 2.0*pk*cos_th);
                    double Rp_m_l = (il == 0) ? p : sqrt(l2 + p2 - 2.0*pl*cos_theta_minus_psi);
                    double R_l_m_2z_p = (il == 0) ? fabs(two_z_minus_1) * p : sqrt(l2 + two_z_minus_1_2*p2 - 2.0*pl*two_z_minus_1*cos_th);
                    double R_l_m_2_1z_p = (il == 0) ? one_z * p : sqrt(l2 + one_z_2*p2 - 2.0*pl*one_z*cos_th);

                    double numer_pmk_lmp = (p2 - pk * cos_th - pl * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_lmp = fmin(fmax(numer_pmk_lmp / (Rp_m_k * Rp_m_l + 1e-12), -1.0), 1.0);
                    double angle_pmk_lmp = acos(cos_val_pmk_lmp);

                    double numer_ppk_lmp = (-(p2 + pk * cos_th - pl * cos_theta_minus_psi) + kl * cos_psi);
                    double cos_val_ppk_lmp = fmin(fmax(numer_ppk_lmp / (Rp_p_k * Rp_m_l + 1e-12), -1.0), 1.0);
                    double angle_ppk_lmp = acos(cos_val_ppk_lmp);

                    double numer_pmk_l2zp = (p2 - pk * cos_th - pl * two_z_minus_1 * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_l2zp = fmin(fmax(numer_pmk_l2zp / (Rp_m_k * R_l_m_2z_p + 1e-12), -1.0), 1.0);
                    double angle_pmk_l2zp = acos(cos_val_pmk_l2zp);

                    double numer_pmk_l2_1zp = (p2 - pk * cos_th - pl * one_z * cos_theta_minus_psi + kl * cos_psi);
                    double cos_val_pmk_l2_1zp = fmin(fmax(numer_pmk_l2_1zp / (Rp_m_k * R_l_m_2_1z_p + 1e-12), -1.0), 1.0);
                    double angle_pmk_l2_1zp = acos(cos_val_pmk_l2_1zp);

                    bool d1 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool d2 = (Rp_p_k >= K_array[0] && Rp_p_k <= K_array[Nk-1] && Rp_m_l >= L_array[0] && Rp_m_l <= L_array[Nl-1]);
                    bool d3 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && R_l_m_2z_p >= L_array[0] && R_l_m_2z_p <= L_array[Nl-1]);
                    bool d4 = (Rp_m_k >= K_array[0] && Rp_m_k <= K_array[Nk-1] && R_l_m_2_1z_p >= L_array[0] && R_l_m_2_1z_p <= L_array[Nl-1]);

                    if (d1 && d2 && d3 && d4) {
                        dcomplex_d f0_kmp_lmp = sample_trilinear_device(fH0, 0, angle_pmk_lmp, Rp_m_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_kpp_lmp = sample_trilinear_device(fH0, 0, angle_ppk_lmp, Rp_p_k, Rp_m_l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_pmk_l2zp = sample_trilinear_device(fH0, 0, angle_pmk_l2zp, Rp_m_k, R_l_m_2z_p, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d f0_pmk_l2_1zp = sample_trilinear_device(fH0, 0, angle_pmk_l2_1zp, Rp_m_k, R_l_m_2_1z_p, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);

                        dcomplex_d M10_contrib = dcomplex_d(CA, 0.0) * (f0_pmk_l2zp + f0_pmk_l2_1zp - f0_kmp_lmp - f0_kpp_lmp);

                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * M10_contrib;
                    }

                    // M11
                    double R_k_zp = sqrt(k2 + 4.0*z2*p2 - 4.0*pk*z*cos_th);
                    double R_k_1zp = sqrt(k2 + 4.0*one_z_2*p2 - 4.0*pk*one_z*cos_th);

                    double numer_k_zp_l = (kl * cos_psi - 2.0*z*pl*cos_theta_minus_psi);
                    double cos_val_k_zp_l = fmin(fmax(numer_k_zp_l / (R_k_zp * l + 1e-12), -1.0), 1.0);
                    double angle_k_zp_l = (il > 0) ? acos(cos_val_k_zp_l) : 0.0;

                    double numer_k_1zp_l = (kl * cos_psi - 2.0*one_z*pl*cos_theta_minus_psi);
                    double cos_val_k_1zp_l = fmin(fmax(numer_k_1zp_l / (R_k_1zp * l + 1e-12), -1.0), 1.0);
                    double angle_k_1zp_l = (il > 0) ? acos(cos_val_k_1zp_l) : 0.0;

                    bool d5 = (R_k_zp >= K_array[0] && R_k_zp <= K_array[Nk-1]);
                    bool d6 = (R_k_1zp >= K_array[0] && R_k_1zp <= K_array[Nk-1]);

                    if (d5 && d6) {
                        dcomplex_d f1_k_zp_l = sample_trilinear_device(fH0, 1, angle_k_zp_l, R_k_zp, l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);
                        dcomplex_d f1_k_1zp_l = sample_trilinear_device(fH0, 1, angle_k_1zp_l, R_k_1zp, l, Npsi, Nk, Nl, psi_array[0], psi_array[Npsi-1], K_array[0], K_array[Nk-1], L_array[0], L_array[Nl-1], inv_dpsi, inv_dk, inv_dl, psi_array, K_array, L_array);
                        dcomplex_d M11_contrib = dcomplex_d(2.0 * CF, 0.0) * (dcomplex_d(2.0, 0.0) * f_ - f1_k_zp_l - f1_k_1zp_l);
                        sum_integral += dcomplex_d(Vp * w_total, 0.0) * M11_contrib;
                    }
                }
            }
        }

        HF[base] += prefac * sum_integral;
    }
}

// --------------------- GPU Dispatcher ---------------------
std::vector<std::complex<double>> Hamiltonian_qqbar_GPU(const Physis& sys, const std::vector<std::complex<double>>& fH0) {
    int Nk = sys.Nk();
    int Nl = sys.Nl();
    int Npsi = sys.NPsi();
    int Nsig = 2;
    size_t total_size = Nsig*Npsi*Nk*Nl;

    // Convert std::complex to thrust::complex for GPU
    std::vector<dcomplex> fH0_thrust(fH0.size());
    for (size_t i = 0; i < fH0.size(); ++i) {
        fH0_thrust[i] = dcomplex_d(fH0[i].real(), fH0[i].imag());
    }

    // Allocate device memory
    thrust::device_vector<dcomplex_d> d_fH0(fH0_thrust.begin(), fH0_thrust.end());
    thrust::device_vector<dcomplex_d> d_HF(total_size);

    const auto& psi_host = sys.Psi();
    const auto& cos_psi_host = sys.Cos_Psi();
    const auto& K_host = sys.K();
    const auto& L_host = sys.L();

    thrust::device_vector<double> d_psi(psi_host.begin(), psi_host.end());
    thrust::device_vector<double> d_cos_psi(cos_psi_host.begin(), cos_psi_host.end());
    thrust::device_vector<double> d_K(K_host.begin(), K_host.end());
    thrust::device_vector<double> d_L(L_host.begin(), L_host.end());

    double delta_k = sys.Lk()/(Nk-1);
    double delta_l = sys.Ll()/(Nl-1);
    double delta_psi = PI/(Npsi-1);

    int threads = 256;
    int blocks = (total_size + threads - 1)/threads;

    // pass pmin, pmax, z
    double pmin = sys.pmin();
    double pmax = sys.pmax();
    double z = sys.z();

    hamiltonian_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_fH0.data()),
        thrust::raw_pointer_cast(d_HF.data()),
        Npsi, Nk, Nl,
        thrust::raw_pointer_cast(d_psi.data()),
        thrust::raw_pointer_cast(d_cos_psi.data()),
        thrust::raw_pointer_cast(d_K.data()),
        thrust::raw_pointer_cast(d_L.data()),
        delta_psi, delta_k, delta_l,
        sys.omega(), sys.qtilde(), sys.mu(),
        sys.mode(), sys.Ncmode() == "LNc",
        pmin, pmax, z
    );

    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel error: " + std::string(cudaGetErrorString(err)));
    }

    std::vector<dcomplex_d> HF_thrust(total_size);
    thrust::copy(d_HF.begin(), d_HF.end(), HF_thrust.begin());

    // Convert back to std::complex
    std::vector<std::complex<double>> HF(total_size);
    for (size_t i = 0; i < total_size; ++i) {
        HF[i] = std::complex<double>(HF_thrust[i].real(), HF_thrust[i].imag());
    }

    // Enforce psi-independence at l = 0 by averaging (for both sig=0 and sig=1)
    for (int sig = 0; sig < Nsig; ++sig) {
        for (int ik = 0; ik < Nk; ++ik) {
            std::complex<double> avg(0.0, 0.0);
            for (int ip = 0; ip < Npsi; ++ip) {
                avg += HF[sys.idx(sig, ip, 0, ik)];
            }
            avg /= double(Npsi);
            for (int ip = 0; ip < Npsi; ++ip) {
                HF[sys.idx(sig, ip, 0, ik)] = avg;
            }
        }
    }

    return HF;
}