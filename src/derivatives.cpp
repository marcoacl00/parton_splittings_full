#include "derivatives.hpp"
#include "spline.h"

using namespace std;
using dcomplex = complex<double>;
const double PI = 3.141592653589793;


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
){
    int Nsig = sys.Nsig();
    int Nk = sys.Nk();
    int Nl = sys.Nl();
    int Npsi = sys.NPsi();

    double delta_k = sys.Lk() / (Nk - 1);
    double delta_l = sys.Ll() / (Nl - 1);
    double delta_psi = PI / (Npsi - 1);

    double inv_delta_k = 1.0 / delta_k;
    double inv_delta_l = 1.0 / delta_l;
    double inv_delta_psi = 1.0 / delta_psi;

    double inv_2_delta_k2 = 1.0 / (2.0 * delta_k * delta_k);
    double inv_2_delta_l2 = 1.0 / (2.0 * delta_l * delta_l);
    double inv_2_delta_psi2 = 1.0 / (2.0 * delta_psi * delta_psi);

    fk.assign(fH0.size(), dcomplex(0.0, 0.0));
    fkk.assign(fH0.size(), dcomplex(0.0, 0.0));
    fl.assign(fH0.size(), dcomplex(0.0, 0.0));
    fll.assign(fH0.size(), dcomplex(0.0, 0.0));
    fp.assign(fH0.size(), dcomplex(0.0, 0.0));
    fpp.assign(fH0.size(), dcomplex(0.0, 0.0));
    flk.assign(fH0.size(), dcomplex(0.0, 0.0));
    fpl.assign(fH0.size(), dcomplex(0.0, 0.0));
    fpk.assign(fH0.size(), dcomplex(0.0, 0.0));

    //#pragma omp parallel for schedule(dynamic, 8) if(fH0.size() > 64)
    for(int s = 0; s < Nsig; s++){
        for (int idx = 0; idx < Npsi * Nk * Nl; ++idx) {
            int ip = idx / (Nl * Nk);
            int il = (idx / Nk) % Nl;
            int ik = idx % Nk;
            std::size_t flat_idx = sys.idx(s, ip, il, ik);
            
            // ========== K-derivatives ==========
            // First derivative fk
            if (ik == 0) {
                // Forward difference (2nd order): f'(0) ≈ (-3f0 + 4f1 - f2)/(2Δk)
                if (Nk > 2) {
                    fk[flat_idx] = (-3.0 * fH0[sys.idx(s, ip, il, 0)] 
                                + 4.0 * fH0[sys.idx(s, ip, il, 1)] 
                                - fH0[sys.idx(s, ip, il, 2)]) * (0.5 * inv_delta_k);
                } else if (Nk > 1) {
                    fk[flat_idx] = (fH0[sys.idx(s, ip, il, 1)] - fH0[sys.idx(s, ip, il, 0)]) * inv_delta_k;
                }
            } else if (ik == Nk - 1) {
                // Backward difference (2nd order): f'(N) ≈ (3fN - 4f(N-1) + f(N-2))/(2Δk)
                if (Nk > 2) {
                    fk[flat_idx] = (3.0 * fH0[sys.idx(s, ip, il, Nk-1)] 
                                - 4.0 * fH0[sys.idx(s, ip, il, Nk-2)] 
                                + fH0[sys.idx(s, ip, il, Nk-3)]) * (0.5 * inv_delta_k);
                } else {
                    fk[flat_idx] = (fH0[sys.idx(s, ip, il, Nk-1)] - fH0[sys.idx(s, ip, il, Nk-2)]) * inv_delta_k;
                }
            } else {
                // Central difference (2nd order): f'(i) ≈ (f(i+1) - f(i-1))/(2Δk)
                fk[flat_idx] = (fH0[sys.idx(s, ip, il, ik + 1)] - fH0[sys.idx(s, ip, il, ik - 1)]) * (0.5 * inv_delta_k);
            }
            
            // Second derivative fkk
            if (ik == 0) {
                // Forward difference (2nd order): f''(0) ≈ (2f0 - 5f1 + 4f2 - f3)/Δk²
                if (Nk > 3) {
                    fkk[flat_idx] = (2.0 * fH0[sys.idx(s, ip, il, 0)] 
                                - 5.0 * fH0[sys.idx(s, ip, il, 1)] 
                                + 4.0 * fH0[sys.idx(s, ip, il, 2)] 
                                - fH0[sys.idx(s, ip, il, 3)]) * inv_delta_k * inv_delta_k;
                } else if (Nk > 2) {
                    // Fallback to lower order if not enough points
                    fkk[flat_idx] = (fH0[sys.idx(s, ip, il, 0)] 
                                - 2.0 * fH0[sys.idx(s, ip, il, 1)] 
                                + fH0[sys.idx(s, ip, il, 2)]) * inv_delta_k * inv_delta_k;
                } else {
                    fkk[flat_idx] = 0.0;
                }
            } else if (ik == Nk - 1) {
                // Backward difference (2nd order): f''(N) ≈ (2fN - 5f(N-1) + 4f(N-2) - f(N-3))/Δk²
                if (Nk > 3) {
                    fkk[flat_idx] = (2.0 * fH0[sys.idx(s, ip, il, Nk-1)] 
                                - 5.0 * fH0[sys.idx(s, ip, il, Nk-2)] 
                                + 4.0 * fH0[sys.idx(s, ip, il, Nk-3)] 
                                - fH0[sys.idx(s, ip, il, Nk-4)]) * inv_delta_k * inv_delta_k;
                } else if (Nk > 2) {
                    fkk[flat_idx] = (fH0[sys.idx(s, ip, il, Nk-1)] 
                                - 2.0 * fH0[sys.idx(s, ip, il, Nk-2)] 
                                + fH0[sys.idx(s, ip, il, Nk-3)]) * inv_delta_k * inv_delta_k;
                } else {
                    fkk[flat_idx] = 0.0;
                }
            } else {
                // Central difference (2nd order)
                fkk[flat_idx] = (fH0[sys.idx(s, ip, il, ik + 1)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip, il, ik - 1)]) * inv_delta_k * inv_delta_k;
            }
            
            // ========== L-derivatives ==========
            // First derivative fl
            if (il == 0) {
                // Forward difference (2nd order): f'(0) ≈ (-3f0 + 4f1 - f2)/(2Δl)
                if (Nl > 2) {
                    fl[flat_idx] = (-3.0 * fH0[sys.idx(s, ip, 0, ik)] 
                                + 4.0 * fH0[sys.idx(s, ip, 1, ik)] 
                                - fH0[sys.idx(s, ip, 2, ik)]) * (0.5 * inv_delta_l);
                } else if (Nl > 1) {
                    fl[flat_idx] = (fH0[sys.idx(s, ip, 1, ik)] - fH0[sys.idx(s, ip, 0, ik)]) * inv_delta_l;
                }
            } else if (il == Nl - 1) {
                // Backward difference (2nd order): f'(N) ≈ (3fN - 4f(N-1) + f(N-2))/(2Δl)
                if (Nl > 2) {
                    fl[flat_idx] = (3.0 * fH0[sys.idx(s, ip, Nl-1, ik)] 
                                - 4.0 * fH0[sys.idx(s, ip, Nl-2, ik)] 
                                + fH0[sys.idx(s, ip, Nl-3, ik)]) * (0.5 * inv_delta_l);
                } else {
                    fl[flat_idx] = (fH0[sys.idx(s, ip, Nl-1, ik)] - fH0[sys.idx(s, ip, Nl-2, ik)]) * inv_delta_l;
                }
            } else {
                // Central difference (2nd order): f'(i) ≈ (f(i+1) - f(i-1))/(2Δl)
                fl[flat_idx] = (fH0[sys.idx(s, ip, il + 1, ik)] - fH0[sys.idx(s, ip, il - 1, ik)]) * (0.5 * inv_delta_l);
            }
            
            // Second derivative fll
            if (il == 0) {
                // Forward difference (2nd order): f''(0) ≈ (2f0 - 5f1 + 4f2 - f3)/Δl²
                if (Nl > 3) {
                    fll[flat_idx] = (2.0 * fH0[sys.idx(s, ip, 0, ik)] 
                                - 5.0 * fH0[sys.idx(s, ip, 1, ik)] 
                                + 4.0 * fH0[sys.idx(s, ip, 2, ik)] 
                                - fH0[sys.idx(s, ip, 3, ik)]) * inv_delta_l * inv_delta_l;
                } else if (Nl > 2) {
                    fll[flat_idx] = (fH0[sys.idx(s, ip, 0, ik)] 
                                - 2.0 * fH0[sys.idx(s, ip, 1, ik)] 
                                + fH0[sys.idx(s, ip, 2, ik)]) * inv_delta_l * inv_delta_l;
                } else {
                    fll[flat_idx] = 0.0;
                }
            } else if (il == Nl - 1) {
                // Backward difference (2nd order): f''(N) ≈ (2fN - 5f(N-1) + 4f(N-2) - f(N-3))/Δl²
                if (Nl > 3) {
                    fll[flat_idx] = (2.0 * fH0[sys.idx(s, ip, Nl-1, ik)] 
                                - 5.0 * fH0[sys.idx(s, ip, Nl-2, ik)] 
                                + 4.0 * fH0[sys.idx(s, ip, Nl-3, ik)] 
                                - fH0[sys.idx(s, ip, Nl-4, ik)]) * inv_delta_l * inv_delta_l;
                } else if (Nl > 2) {
                    fll[flat_idx] = (fH0[sys.idx(s, ip, Nl-1, ik)] 
                                - 2.0 * fH0[sys.idx(s, ip, Nl-2, ik)] 
                                + fH0[sys.idx(s, ip, Nl-3, ik)]) * inv_delta_l * inv_delta_l;
                } else {
                    fll[flat_idx] = 0.0;
                }
            } else {
                // Central difference (2nd order)
                fll[flat_idx] = (fH0[sys.idx(s, ip, il + 1, ik)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip, il - 1, ik)]) * inv_delta_l * inv_delta_l;
            }
            
            // ========== PSI-derivatives ==========
            // First derivative fp
            if (ip == 0) {
                // Forward difference (2nd order)
                if (Npsi > 2) {
                    fp[flat_idx] = (-3.0 * fH0[sys.idx(s, 0, il, ik)] 
                                + 4.0 * fH0[sys.idx(s, 1, il, ik)] 
                                - fH0[sys.idx(s, 2, il, ik)]) * (0.5 * inv_delta_psi);
                } else if (Npsi > 1) {
                    fp[flat_idx] = (fH0[sys.idx(s, 1, il, ik)] - fH0[sys.idx(s, 0, il, ik)]) * inv_delta_psi;
                }
            } else if (ip == Npsi - 1) {
                // Backward difference (2nd order)
                if (Npsi > 2) {
                    fp[flat_idx] = (3.0 * fH0[sys.idx(s, Npsi-1, il, ik)] 
                                - 4.0 * fH0[sys.idx(s, Npsi-2, il, ik)] 
                                + fH0[sys.idx(s, Npsi-3, il, ik)]) * (0.5 * inv_delta_psi);
                } else {
                    fp[flat_idx] = (fH0[sys.idx(s, Npsi-1, il, ik)] - fH0[sys.idx(s, Npsi-2, il, ik)]) * inv_delta_psi;
                }
            } else {
                // Central difference (2nd order)
                fp[flat_idx] = (fH0[sys.idx(s, ip + 1, il, ik)] - fH0[sys.idx(s, ip - 1, il, ik)]) * (0.5 * inv_delta_psi);
            }
            
            // Second derivative fpp
            if (ip == 0) {
                // Forward difference (2nd order): f''(0) ≈ (2f0 - 5f1 + 4f2 - f3)/Δpsi²
                if (Npsi > 3) {
                    fpp[flat_idx] = (2.0 * fH0[sys.idx(s, 0, il, ik)] 
                                - 5.0 * fH0[sys.idx(s, 1, il, ik)] 
                                + 4.0 * fH0[sys.idx(s, 2, il, ik)] 
                                - fH0[sys.idx(s, 3, il, ik)]) * inv_delta_psi * inv_delta_psi;
                } else if (Npsi > 2) {
                    fpp[flat_idx] = (fH0[sys.idx(s, 0, il, ik)] 
                                - 2.0 * fH0[sys.idx(s, 1, il, ik)] 
                                + fH0[sys.idx(s, 2, il, ik)]) * inv_delta_psi * inv_delta_psi;
                } else {
                    fpp[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1) {
                // Backward difference (2nd order): f''(N) ≈ (2fN - 5f(N-1) + 4f(N-2) - f(N-3))/Δpsi²
                if (Npsi > 3) {
                    fpp[flat_idx] = (2.0 * fH0[sys.idx(s, Npsi-1, il, ik)] 
                                - 5.0 * fH0[sys.idx(s, Npsi-2, il, ik)] 
                                + 4.0 * fH0[sys.idx(s, Npsi-3, il, ik)] 
                                - fH0[sys.idx(s, Npsi-4, il, ik)]) * inv_delta_psi * inv_delta_psi;
                } else if (Npsi > 2) {
                    fpp[flat_idx] = (fH0[sys.idx(s, Npsi-1, il, ik)] 
                                - 2.0 * fH0[sys.idx(s, Npsi-2, il, ik)] 
                                + fH0[sys.idx(s, Npsi-3, il, ik)]) * inv_delta_psi * inv_delta_psi;
                } else {
                    fpp[flat_idx] = 0.0;
                }
            } else {
                // Central difference (2nd order)
                fpp[flat_idx] = (fH0[sys.idx(s, ip + 1, il, ik)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip - 1, il, ik)]) * inv_delta_psi * inv_delta_psi;
            }
            
            // ========== Mixed derivatives ==========
            // flk: use central differences when possible, one-sided at boundaries
                        // fpl
            if (il == 0 && ik == 0) {
                // Both at left boundary: forward-forward
                if (Nl > 2 && Nk > 2) {
                    // ∂²f/∂l∂k ≈ [∂f/∂k(l=1) - ∂f/∂k(l=0)] / Δl
                    // where ∂f/∂k uses forward difference
                    dcomplex dfdk_l0 = (-3.0 * fH0[sys.idx(s, ip, 0, 0)] + 4.0 * fH0[sys.idx(s, ip, 0, 1)] - fH0[sys.idx(s, ip, 0, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_l1 = (-3.0 * fH0[sys.idx(s, ip, 1, 0)] + 4.0 * fH0[sys.idx(s, ip, 1, 1)] - fH0[sys.idx(s, ip, 1, 2)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_l1 - dfdk_l0) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (il == 0 && ik == Nk - 1) {
                // l at left, k at right: forward-backward
                if (Nl > 2 && Nk > 2) {
                    dcomplex dfdk_l0 = (3.0 * fH0[sys.idx(s, ip, 0, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, 0, Nk-2)] + fH0[sys.idx(s, ip, 0, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_l1 = (3.0 * fH0[sys.idx(s, ip, 1, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, 1, Nk-2)] + fH0[sys.idx(s, ip, 1, Nk-3)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_l1 - dfdk_l0) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (il == Nl - 1 && ik == 0) {
                // l at right, k at left: backward-forward
                if (Nl > 2 && Nk > 2) {
                    dcomplex dfdk_lN = (-3.0 * fH0[sys.idx(s, ip, Nl-1, 0)] + 4.0 * fH0[sys.idx(s, ip, Nl-1, 1)] - fH0[sys.idx(s, ip, Nl-1, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_lNm1 = (-3.0 * fH0[sys.idx(s, ip, Nl-2, 0)] + 4.0 * fH0[sys.idx(s, ip, Nl-2, 1)] - fH0[sys.idx(s, ip, Nl-2, 2)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_lN - dfdk_lNm1) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (il == Nl - 1 && ik == Nk - 1) {
                // Both at right boundary: backward-backward
                if (Nl > 2 && Nk > 2) {
                    dcomplex dfdk_lN = (3.0 * fH0[sys.idx(s, ip, Nl-1, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, Nl-1, Nk-2)] + fH0[sys.idx(s, ip, Nl-1, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_lNm1 = (3.0 * fH0[sys.idx(s, ip, Nl-2, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, Nl-2, Nk-2)] + fH0[sys.idx(s, ip, Nl-2, Nk-3)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_lN - dfdk_lNm1) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (il == 0) {
                // l at left boundary, k in interior: forward in l, central in k
                if (Nl > 1 && ik > 0 && ik < Nk - 1) {
                    dcomplex dfdk_l0 = (fH0[sys.idx(s, ip, 0, ik+1)] - fH0[sys.idx(s, ip, 0, ik-1)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_l1 = (fH0[sys.idx(s, ip, 1, ik+1)] - fH0[sys.idx(s, ip, 1, ik-1)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_l1 - dfdk_l0) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (il == Nl - 1) {
                // l at right boundary, k in interior: backward in l, central in k
                if (Nl > 1 && ik > 0 && ik < Nk - 1) {
                    dcomplex dfdk_lN = (fH0[sys.idx(s, ip, Nl-1, ik+1)] - fH0[sys.idx(s, ip, Nl-1, ik-1)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_lNm1 = (fH0[sys.idx(s, ip, Nl-2, ik+1)] - fH0[sys.idx(s, ip, Nl-2, ik-1)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_lN - dfdk_lNm1) * inv_delta_l;
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (ik == 0) {
                // k at left boundary, l in interior: central in l, forward in k
                if (Nk > 2 && il > 0 && il < Nl - 1) {
                    dcomplex dfdk_lp = (-3.0 * fH0[sys.idx(s, ip, il+1, 0)] + 4.0 * fH0[sys.idx(s, ip, il+1, 1)] - fH0[sys.idx(s, ip, il+1, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_lm = (-3.0 * fH0[sys.idx(s, ip, il-1, 0)] + 4.0 * fH0[sys.idx(s, ip, il-1, 1)] - fH0[sys.idx(s, ip, il-1, 2)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_lp - dfdk_lm) * (0.5 * inv_delta_l);
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else if (ik == Nk - 1) {
                // k at right boundary, l in interior: central in l, backward in k
                if (Nk > 2 && il > 0 && il < Nl - 1) {
                    dcomplex dfdk_lp = (3.0 * fH0[sys.idx(s, ip, il+1, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, il+1, Nk-2)] + fH0[sys.idx(s, ip, il+1, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_lm = (3.0 * fH0[sys.idx(s, ip, il-1, Nk-1)] - 4.0 * fH0[sys.idx(s, ip, il-1, Nk-2)] + fH0[sys.idx(s, ip, il-1, Nk-3)]) * (0.5 * inv_delta_k);
                    flk[flat_idx] = (dfdk_lp - dfdk_lm) * (0.5 * inv_delta_l);
                } else {
                    flk[flat_idx] = 0.0;
                }
            } else {
                // Both in interior: standard central difference
                flk[flat_idx] = (fH0[sys.idx(s, ip, il+1, ik+1)] - fH0[sys.idx(s, ip, il-1, ik+1)]
                            - fH0[sys.idx(s, ip, il+1, ik-1)] + fH0[sys.idx(s, ip, il-1, ik-1)]) 
                            * (0.25 * inv_delta_l * inv_delta_k);
            }
            
            // ========== Mixed derivatives: ∂²f/∂p∂l ==========
            if (ip == 0 && il == 0) {
                // Both at left boundary: forward-forward
                if (Npsi > 2 && Nl > 2) {
                    dcomplex dfdl_p0 = (-3.0 * fH0[sys.idx(s, 0, 0, ik)] + 4.0 * fH0[sys.idx(s, 0, 1, ik)] - fH0[sys.idx(s, 0, 2, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_p1 = (-3.0 * fH0[sys.idx(s, 1, 0, ik)] + 4.0 * fH0[sys.idx(s, 1, 1, ik)] - fH0[sys.idx(s, 1, 2, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_p1 - dfdl_p0) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (ip == 0 && il == Nl - 1) {
                // p at left, l at right: forward-backward
                if (Npsi > 2 && Nl > 2) {
                    dcomplex dfdl_p0 = (3.0 * fH0[sys.idx(s, 0, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, 0, Nl-2, ik)] + fH0[sys.idx(s, 0, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_p1 = (3.0 * fH0[sys.idx(s, 1, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, 1, Nl-2, ik)] + fH0[sys.idx(s, 1, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_p1 - dfdl_p0) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1 && il == 0) {
                // p at right, l at left: backward-forward
                if (Npsi > 2 && Nl > 2) {
                    dcomplex dfdl_pN = (-3.0 * fH0[sys.idx(s, Npsi-1, 0, ik)] + 4.0 * fH0[sys.idx(s, Npsi-1, 1, ik)] - fH0[sys.idx(s, Npsi-1, 2, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_pNm1 = (-3.0 * fH0[sys.idx(s, Npsi-2, 0, ik)] + 4.0 * fH0[sys.idx(s, Npsi-2, 1, ik)] - fH0[sys.idx(s, Npsi-2, 2, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_pN - dfdl_pNm1) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1 && il == Nl - 1) {
                // Both at right boundary: backward-backward
                if (Npsi > 2 && Nl > 2) {
                    dcomplex dfdl_pN = (3.0 * fH0[sys.idx(s, Npsi-1, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, Npsi-1, Nl-2, ik)] + fH0[sys.idx(s, Npsi-1, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_pNm1 = (3.0 * fH0[sys.idx(s, Npsi-2, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, Npsi-2, Nl-2, ik)] + fH0[sys.idx(s, Npsi-2, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_pN - dfdl_pNm1) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (ip == 0) {
                // p at left boundary, l in interior: forward in p, central in l
                if (Npsi > 1 && il > 0 && il < Nl - 1) {
                    dcomplex dfdl_p0 = (fH0[sys.idx(s, 0, il+1, ik)] - fH0[sys.idx(s, 0, il-1, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_p1 = (fH0[sys.idx(s, 1, il+1, ik)] - fH0[sys.idx(s, 1, il-1, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_p1 - dfdl_p0) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1) {
                // p at right boundary, l in interior: backward in p, central in l
                if (Npsi > 1 && il > 0 && il < Nl - 1) {
                    dcomplex dfdl_pN = (fH0[sys.idx(s, Npsi-1, il+1, ik)] - fH0[sys.idx(s, Npsi-1, il-1, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_pNm1 = (fH0[sys.idx(s, Npsi-2, il+1, ik)] - fH0[sys.idx(s, Npsi-2, il-1, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_pN - dfdl_pNm1) * inv_delta_psi;
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (il == 0) {
                // l at left boundary, p in interior: central in p, forward in l
                if (Nl > 2 && ip > 0 && ip < Npsi - 1) {
                    dcomplex dfdl_pp = (-3.0 * fH0[sys.idx(s, ip+1, 0, ik)] + 4.0 * fH0[sys.idx(s, ip+1, 1, ik)] - fH0[sys.idx(s, ip+1, 2, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_pm = (-3.0 * fH0[sys.idx(s, ip-1, 0, ik)] + 4.0 * fH0[sys.idx(s, ip-1, 1, ik)] - fH0[sys.idx(s, ip-1, 2, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_pp - dfdl_pm) * (0.5 * inv_delta_psi);
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else if (il == Nl - 1) {
                // l at right boundary, p in interior: central in p, backward in l
                if (Nl > 2 && ip > 0 && ip < Npsi - 1) {
                    dcomplex dfdl_pp = (3.0 * fH0[sys.idx(s, ip+1, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, ip+1, Nl-2, ik)] + fH0[sys.idx(s, ip+1, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    dcomplex dfdl_pm = (3.0 * fH0[sys.idx(s, ip-1, Nl-1, ik)] - 4.0 * fH0[sys.idx(s, ip-1, Nl-2, ik)] + fH0[sys.idx(s, ip-1, Nl-3, ik)]) * (0.5 * inv_delta_l);
                    fpl[flat_idx] = (dfdl_pp - dfdl_pm) * (0.5 * inv_delta_psi);
                } else {
                    fpl[flat_idx] = 0.0;
                }
            } else {
                // Both in interior: standard central difference
                fpl[flat_idx] = (fH0[sys.idx(s, ip+1, il+1, ik)] - fH0[sys.idx(s, ip-1, il+1, ik)]
                            - fH0[sys.idx(s, ip+1, il-1, ik)] + fH0[sys.idx(s, ip-1, il-1, ik)]) 
                            * (0.25 * inv_delta_psi * inv_delta_l);
            }
            
            // ========== Mixed derivatives: ∂²f/∂p∂k ==========
            if (ip == 0 && ik == 0) {
                // Both at left boundary: forward-forward
                if (Npsi > 2 && Nk > 2) {
                    dcomplex dfdk_p0 = (-3.0 * fH0[sys.idx(s, 0, il, 0)] + 4.0 * fH0[sys.idx(s, 0, il, 1)] - fH0[sys.idx(s, 0, il, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_p1 = (-3.0 * fH0[sys.idx(s, 1, il, 0)] + 4.0 * fH0[sys.idx(s, 1, il, 1)] - fH0[sys.idx(s, 1, il, 2)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_p1 - dfdk_p0) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ip == 0 && ik == Nk - 1) {
                // p at left, k at right: forward-backward
                if (Npsi > 2 && Nk > 2) {
                    dcomplex dfdk_p0 = (3.0 * fH0[sys.idx(s, 0, il, Nk-1)] - 4.0 * fH0[sys.idx(s, 0, il, Nk-2)] + fH0[sys.idx(s, 0, il, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_p1 = (3.0 * fH0[sys.idx(s, 1, il, Nk-1)] - 4.0 * fH0[sys.idx(s, 1, il, Nk-2)] + fH0[sys.idx(s, 1, il, Nk-3)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_p1 - dfdk_p0) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1 && ik == 0) {
                // p at right, k at left: backward-forward
                if (Npsi > 2 && Nk > 2) {
                    dcomplex dfdk_pN = (-3.0 * fH0[sys.idx(s, Npsi-1, il, 0)] + 4.0 * fH0[sys.idx(s, Npsi-1, il, 1)] - fH0[sys.idx(s, Npsi-1, il, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_pNm1 = (-3.0 * fH0[sys.idx(s, Npsi-2, il, 0)] + 4.0 * fH0[sys.idx(s, Npsi-2, il, 1)] - fH0[sys.idx(s, Npsi-2, il, 2)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_pN - dfdk_pNm1) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1 && ik == Nk - 1) {
                // Both at right boundary: backward-backward
                if (Npsi > 2 && Nk > 2) {
                    dcomplex dfdk_pN = (3.0 * fH0[sys.idx(s, Npsi-1, il, Nk-1)] - 4.0 * fH0[sys.idx(s, Npsi-1, il, Nk-2)] + fH0[sys.idx(s, Npsi-1, il, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_pNm1 = (3.0 * fH0[sys.idx(s, Npsi-2, il, Nk-1)] - 4.0 * fH0[sys.idx(s, Npsi-2, il, Nk-2)] + fH0[sys.idx(s, Npsi-2, il, Nk-3)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_pN - dfdk_pNm1) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ip == 0) {
                // p at left boundary, k in interior: forward in p, central in k
                if (Npsi > 1 && ik > 0 && ik < Nk - 1) {
                    dcomplex dfdk_p0 = (fH0[sys.idx(s, 0, il, ik+1)] - fH0[sys.idx(s, 0, il, ik-1)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_p1 = (fH0[sys.idx(s, 1, il, ik+1)] - fH0[sys.idx(s, 1, il, ik-1)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_p1 - dfdk_p0) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ip == Npsi - 1) {
                // p at right boundary, k in interior: backward in p, central in k
                if (Npsi > 1 && ik > 0 && ik < Nk - 1) {
                    dcomplex dfdk_pN = (fH0[sys.idx(s, Npsi-1, il, ik+1)] - fH0[sys.idx(s, Npsi-1, il, ik-1)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_pNm1 = (fH0[sys.idx(s, Npsi-2, il, ik+1)] - fH0[sys.idx(s, Npsi-2, il, ik-1)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_pN - dfdk_pNm1) * inv_delta_psi;
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ik == 0) {
                // k at left boundary, p in interior: central in p, forward in k
                if (Nk > 2 && ip > 0 && ip < Npsi - 1) {
                    dcomplex dfdk_pp = (-3.0 * fH0[sys.idx(s, ip+1, il, 0)] + 4.0 * fH0[sys.idx(s, ip+1, il, 1)] - fH0[sys.idx(s, ip+1, il, 2)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_pm = (-3.0 * fH0[sys.idx(s, ip-1, il, 0)] + 4.0 * fH0[sys.idx(s, ip-1, il, 1)] - fH0[sys.idx(s, ip-1, il, 2)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_pp - dfdk_pm) * (0.5 * inv_delta_psi);
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else if (ik == Nk - 1) {
                // k at right boundary, p in interior: central in p, backward in k
                if (Nk > 2 && ip > 0 && ip < Npsi - 1) {
                    dcomplex dfdk_pp = (3.0 * fH0[sys.idx(s, ip+1, il, Nk-1)] - 4.0 * fH0[sys.idx(s, ip+1, il, Nk-2)] + fH0[sys.idx(s, ip+1, il, Nk-3)]) * (0.5 * inv_delta_k);
                    dcomplex dfdk_pm = (3.0 * fH0[sys.idx(s, ip-1, il, Nk-1)] - 4.0 * fH0[sys.idx(s, ip-1, il, Nk-2)] + fH0[sys.idx(s, ip-1, il, Nk-3)]) * (0.5 * inv_delta_k);
                    fpk[flat_idx] = (dfdk_pp - dfdk_pm) * (0.5 * inv_delta_psi);
                } else {
                    fpk[flat_idx] = 0.0;
                }
            } else {
                // Both in interior: standard central difference
                fpk[flat_idx] = (fH0[sys.idx(s, ip+1, il, ik+1)] - fH0[sys.idx(s, ip-1, il, ik+1)]
                            - fH0[sys.idx(s, ip+1, il, ik-1)] + fH0[sys.idx(s, ip-1, il, ik-1)]) 
                            * (0.25 * inv_delta_psi * inv_delta_k);
            }
        }
    }
}