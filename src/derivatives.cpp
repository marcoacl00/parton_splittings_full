#include "derivatives.hpp"
#include "spline.h"

using namespace std;
using dcomplex = complex<double>;
const double PI = 3.141592653589793;


void precompute_derivatives_3d(
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

    // some precomputed factors for first derivative finite difference formulas
    double inv_delta_k = 1.0 / delta_k; 
    double inv_delta_l = 1.0 / delta_l;
    double inv_delta_psi = 1.0 / delta_psi;

    // and for second derivative formulas
    double inv_delta_k2 = inv_delta_k * inv_delta_k;
    double inv_delta_l2 = inv_delta_l * inv_delta_l;
    double inv_delta_psi2 = inv_delta_psi * inv_delta_psi;

    // allocate derivative arrays
    fk.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂f/∂k
    fkk.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂k²
    fl.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂f/∂l
    fll.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂l²
    fp.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂f/∂ψ
    fpp.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂ψ²
    flk.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂l∂k = ∂²f/∂k∂l
    fpl.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂ψ∂l = ∂²f/∂l∂ψ
    fpk.assign(fH0.size(), dcomplex(0.0, 0.0)); // ∂²f/∂ψ∂k = ∂²f/∂k∂ψ

    
    for(int s = 0; s < Nsig; s++){

        #pragma omp parallel for collapse(3)
        for (int ip = 0; ip < Npsi; ++ip) {
            for (int il = 0; il < Nl; ++il) {
                for (int ik = 0; ik < Nk; ++ik) {
            std::size_t flat_idx = sys.idx(s, ip, il, ik);

            // note that for the boundary corrections, the function has by construction more than 3 points in each direction,
            // (and more than 5 for l) so we can always apply the 3-point formulas without worrying about out-of-bounds access.

            
            // ------ PURE k-DERIVATIVES ------
            // --------------------------------
            
            // ----- ∂f/∂k -----
            if (ik == 0) {

                // left boundary; use forward difference formulas 
                // f'(k[0]) = (-3f0 + 4f1 - f2)/(2Δk) + O(Δk²)
                fk[flat_idx] = (-3.0 * fH0[sys.idx(s, ip, il, 0)] 
                                + 4.0 * fH0[sys.idx(s, ip, il, 1)] 
                                - fH0[sys.idx(s, ip, il, 2)]) * (0.5 * inv_delta_k);
  

            } else if (ik == Nk - 1) {

                // right boundary; use backward difference formulas
                // f'(k[N-1]) = (3fN - 4f(N-1) + f(N-2))/(2Δk) + O(Δk²)
                fk[flat_idx] = (3.0 * fH0[sys.idx(s, ip, il, Nk-1)] 
                                - 4.0 * fH0[sys.idx(s, ip, il, Nk-2)] 
                                + fH0[sys.idx(s, ip, il, Nk-3)]) * (0.5 * inv_delta_k);
  

            } else {
                // Bulk points
                //  f'(i) = (f(i+1) - f(i-1))/(2Δk) + O(Δk²)
                fk[flat_idx] = (fH0[sys.idx(s, ip, il, ik + 1)] - fH0[sys.idx(s, ip, il, ik - 1)]) * (0.5 * inv_delta_k);
            }
            


            // ----- ∂²f/∂k² -----
            if (ik == 0) {
               // right boundary; use forward difference (2nd order) formula:
                // f''(0) = (2f0 - 5f1 +  4f2 - f3)/Δk² + O(Δk²)

                fkk[flat_idx] = (2.0 * fH0[sys.idx(s, ip, il, 0)] 
                            - 5.0 * fH0[sys.idx(s, ip, il, 1)] 
                            + 4.0 * fH0[sys.idx(s, ip, il, 2)] 
                            - fH0[sys.idx(s, ip, il, 3)]) * inv_delta_k2;

            } else if (ik == Nk - 1) {
                // left boundary; use backward difference (2nd order) formula:
                // f''(N) = (2fN - 5f(N-1) + 4f(N-2) - f(N-3))/Δk² + O(Δk²)

                fkk[flat_idx] = (2.0 * fH0[sys.idx(s, ip, il, Nk-1)] 
                            - 5.0 * fH0[sys.idx(s, ip, il, Nk-2)] 
                            + 4.0 * fH0[sys.idx(s, ip, il, Nk-3)] 
                            - fH0[sys.idx(s, ip, il, Nk-4)]) * inv_delta_k2;
           
                
            } else {
                // bulk points
                // Central difference (2nd order)
                fkk[flat_idx] = (fH0[sys.idx(s, ip, il, ik + 1)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip, il, ik - 1)]) * inv_delta_k2;
            }
            

            // ------ PURE l-DERIVATIVES ------
            // --------------------------------

            // ----- ∂f/∂l -----
            if (il == 0) {
                // this one is very important; the relevant solution is at l = 0
                // left boundary; use forward difference (2nd order) formula:
                // f'(0) = (-11f0 + 18f1 - 9f2 + 2f3)/(6Δl) + O(Δl²)

              fl[flat_idx] = (-11.0 * fH0[sys.idx(s, ip, 0, ik)] 
                            + 18.0 * fH0[sys.idx(s, ip, 1, ik)] 
                            - 9.0 * fH0[sys.idx(s, ip, 2, ik)] 
                            + 2.0 * fH0[sys.idx(s, ip, 3, ik)]) * (1.0 / (6.0)) * inv_delta_l ;

            } else if (il == Nl - 1) {
                // right boundary; use backward difference (2nd order) formula:
                // f'(N) ≈ (3fN - 4f(N-1) + f(N-2))/(2Δl)
                fl[flat_idx] = (3.0 * fH0[sys.idx(s, ip, Nl-1, ik)] 
                            - 4.0 * fH0[sys.idx(s, ip, Nl-2, ik)] 
                            + fH0[sys.idx(s, ip, Nl-3, ik)]) * (0.5 * inv_delta_l);

            } else {
                // bulk points
                // Central difference (2nd order)
                fl[flat_idx] = (fH0[sys.idx(s, ip, il + 1, ik)] - fH0[sys.idx(s, ip, il - 1, ik)]) * (0.5 * inv_delta_l);
            }
            
            // ----- ∂²f/∂l² -----
            if (il == 0) {
                // this one is very important; the relevant solution is at l = 0
                // left boundary; we will use 3rd order forward difference formula to get better accuracy:
                // f''(0) = (-35f0 + 104f1 - 114f2 + 56f3 - 11f4)/(12Δl²) + O(Δl³)
                fll[flat_idx] = 1.0/12.0 * (-35.0 * fH0[sys.idx(s, ip, 0, ik)] 
                            + 104.0 * fH0[sys.idx(s, ip, 1, ik)] 
                            - 114.0 * fH0[sys.idx(s, ip, 2, ik)] 
                            + 56.0 * fH0[sys.idx(s, ip, 3, ik)] 
                            - 11.0 * fH0[sys.idx(s, ip, 4, ik)]) * inv_delta_l2;

            } else if (il == Nl - 1) {
                // right boundary; use backward difference (3nd order) formula:
                // f''(N) = (35fN - 104f(N-1) + 114f(N-2) - 56f(N-3) + 11f(N-4))/(12Δl²) + O(Δl³)
                fll[flat_idx] = 1.0/12.0 * (35.0 * fH0[sys.idx(s, ip, Nl-1, ik)] 
                            - 104.0 * fH0[sys.idx(s, ip, Nl-2, ik)] 
                            + 114.0 * fH0[sys.idx(s, ip, Nl-3, ik)] 
                            - 56.0 * fH0[sys.idx(s, ip, Nl-4, ik)] 
                            + 11.0 * fH0[sys.idx(s, ip, Nl-5, ik)]) * inv_delta_l2;
              

            } else {
                // bulk points
                // Central difference (2nd order)
                fll[flat_idx] = (fH0[sys.idx(s, ip, il + 1, ik)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip, il - 1, ik)]) * inv_delta_l2;
            }

            // ------ PURE psi-DERIVATIVES ------
            // ----------------------------------

            // ----- ∂f/∂psi -----
            if (ip == 0) {
                // given that f(psi) = f(-psi), use 
                // f'(0) = 0
                fp[flat_idx] = 0.0;
               
            } else if (ip == Npsi - 1) {
                // the same happens here, since f(pi - x) = f(-x) = f(x - pi)
                fp[flat_idx] = 0.0;

            } else {
                fp[flat_idx] = (fH0[sys.idx(s, ip + 1, il, ik)] - fH0[sys.idx(s, ip - 1, il, ik)]) * (0.5 * inv_delta_psi);
            }
            
            // ----- ∂²f/∂psi² -----
            if (ip == 0) {
                // Forward difference (2nd order): f''(0) ≈ (2f0 - 5f1 + 4f2 - f3)/Δpsi²
                fpp[flat_idx] = (2.0 * fH0[sys.idx(s, ip + 1, il, ik)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            ) * inv_delta_psi * inv_delta_psi;

            } else if (ip == Npsi - 1) {
                // Backward difference (2nd order): f''(N) ≈ (2fN - 5f(N-1) + 4f(N-2) - f(N-3))/Δpsi²
                fpp[flat_idx] = (- 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + 2.0 * fH0[sys.idx(s, ip - 1, il, ik)]) * inv_delta_psi * inv_delta_psi;

            } else {
                // Central difference (2nd order)
                fpp[flat_idx] = (fH0[sys.idx(s, ip + 1, il, ik)] 
                            - 2.0 * fH0[sys.idx(s, ip, il, ik)] 
                            + fH0[sys.idx(s, ip - 1, il, ik)]) * inv_delta_psi * inv_delta_psi;
            }
            

            // ------- MIXED DERIVATIVES -------
            // ---------------------------------
            
            // ----- ∂²f/∂l∂k -----
            if (il == 0 && ik == 0) {
                // Both at left boundary: forward-forward (2nd order)
                // ∂²f/∂l∂k ≈ (f(1,1) - f(1,0) - f(0,1) + f(0,0))/(Δl·Δk)
                flk[flat_idx] = (fH0[sys.idx(s, ip, 1, 1)] 
                            - fH0[sys.idx(s, ip, 1, 0)] 
                            - fH0[sys.idx(s, ip, 0, 1)] 
                            + fH0[sys.idx(s, ip, 0, 0)]) * inv_delta_l * inv_delta_k;

            } else if (il == 0 && ik == Nk - 1) {
                // l at left, k at right: forward-backward (2nd order)
                flk[flat_idx] = (fH0[sys.idx(s, ip, 1, Nk-1)] 
                            - fH0[sys.idx(s, ip, 1, Nk-2)] 
                            - fH0[sys.idx(s, ip, 0, Nk-1)] 
                            + fH0[sys.idx(s, ip, 0, Nk-2)]) * inv_delta_l * inv_delta_k;

            } else if (il == Nl - 1 && ik == 0) {
                // l at right, k at left: backward-forward (2nd order)
                flk[flat_idx] = (fH0[sys.idx(s, ip, Nl-1, 1)] 
                            - fH0[sys.idx(s, ip, Nl-2, 1)] 
                            - fH0[sys.idx(s, ip, Nl-1, 0)] 
                            + fH0[sys.idx(s, ip, Nl-2, 0)]) * inv_delta_l * inv_delta_k;

            } else if (il == Nl - 1 && ik == Nk - 1) {
                // Both at right boundary: backward-backward (2nd order)
                flk[flat_idx] = (fH0[sys.idx(s, ip, Nl-1, Nk-1)] 
                            - fH0[sys.idx(s, ip, Nl-2, Nk-1)] 
                            - fH0[sys.idx(s, ip, Nl-1, Nk-2)] 
                            + fH0[sys.idx(s, ip, Nl-2, Nk-2)]) * inv_delta_l * inv_delta_k;

            } else if (il == 0) {
                // l at left boundary, k in interior: forward in l, central in k
                flk[flat_idx] = (fH0[sys.idx(s, ip, 1, ik+1)] 
                            - fH0[sys.idx(s, ip, 1, ik-1)] 
                            - fH0[sys.idx(s, ip, 0, ik+1)] 
                            + fH0[sys.idx(s, ip, 0, ik-1)]) * (0.5 * inv_delta_l * inv_delta_k);

            } else if (il == Nl - 1) {
                // l at right boundary, k in interior: backward in l, central in k
                flk[flat_idx] = (fH0[sys.idx(s, ip, Nl-1, ik+1)] 
                            - fH0[sys.idx(s, ip, Nl-1, ik-1)] 
                            - fH0[sys.idx(s, ip, Nl-2, ik+1)] 
                            + fH0[sys.idx(s, ip, Nl-2, ik-1)]) * (0.5 * inv_delta_l * inv_delta_k);

            } else if (ik == 0) {
                // k at left boundary, l in interior: central in l, forward in k
                flk[flat_idx] = (fH0[sys.idx(s, ip, il+1, 1)] 
                            - fH0[sys.idx(s, ip, il-1, 1)] 
                            - fH0[sys.idx(s, ip, il+1, 0)] 
                            + fH0[sys.idx(s, ip, il-1, 0)]) * (0.5 * inv_delta_l * inv_delta_k);

            } else if (ik == Nk - 1) {
                // k at right boundary, l in interior: central in l, backward in k
                flk[flat_idx] = (fH0[sys.idx(s, ip, il+1, Nk-1)] 
                            - fH0[sys.idx(s, ip, il-1, Nk-1)] 
                            - fH0[sys.idx(s, ip, il+1, Nk-2)] 
                            + fH0[sys.idx(s, ip, il-1, Nk-2)]) * (0.5 * inv_delta_l * inv_delta_k);

            } else {
                // Both in interior: standard central difference (2nd order)
                flk[flat_idx] = (fH0[sys.idx(s, ip, il+1, ik+1)] 
                            - fH0[sys.idx(s, ip, il-1, ik+1)]
                            - fH0[sys.idx(s, ip, il+1, ik-1)] 
                            + fH0[sys.idx(s, ip, il-1, ik-1)]) * (0.25 * inv_delta_l * inv_delta_k);
            }
            
            // ----- ∂²f/∂ψ∂l -----
            if (ip == 0 && il == 0) {
                // Both at left boundary: forward-forward (2nd order)
                fpl[flat_idx] = (fH0[sys.idx(s, 1, 1, ik)] 
                            - fH0[sys.idx(s, 1, 0, ik)] 
                            - fH0[sys.idx(s, 0, 1, ik)] 
                            + fH0[sys.idx(s, 0, 0, ik)]) * inv_delta_psi * inv_delta_l;

            } else if (ip == 0 && il == Nl - 1) {
                // p at left, l at right: forward-backward (2nd order)
                fpl[flat_idx] = (fH0[sys.idx(s, 1, Nl-1, ik)] 
                            - fH0[sys.idx(s, 1, Nl-2, ik)] 
                            - fH0[sys.idx(s, 0, Nl-1, ik)] 
                            + fH0[sys.idx(s, 0, Nl-2, ik)]) * inv_delta_psi * inv_delta_l;

            } else if (ip == Npsi - 1 && il == 0) {
                // p at right, l at left: backward-forward (2nd order)
                fpl[flat_idx] = (fH0[sys.idx(s, Npsi-1, 1, ik)] 
                            - fH0[sys.idx(s, Npsi-2, 1, ik)] 
                            - fH0[sys.idx(s, Npsi-1, 0, ik)] 
                            + fH0[sys.idx(s, Npsi-2, 0, ik)]) * inv_delta_psi * inv_delta_l;

            } else if (ip == Npsi - 1 && il == Nl - 1) {
                // Both at right boundary: backward-backward (2nd order)
                fpl[flat_idx] = (fH0[sys.idx(s, Npsi-1, Nl-1, ik)] 
                            - fH0[sys.idx(s, Npsi-2, Nl-1, ik)] 
                            - fH0[sys.idx(s, Npsi-1, Nl-2, ik)] 
                            + fH0[sys.idx(s, Npsi-2, Nl-2, ik)]) * inv_delta_psi * inv_delta_l;

            } else if (ip == 0) {
                // p at left boundary, l in interior: forward in p, central in l
                fpl[flat_idx] = (fH0[sys.idx(s, 1, il+1, ik)] 
                            - fH0[sys.idx(s, 1, il-1, ik)] 
                            - fH0[sys.idx(s, 0, il+1, ik)] 
                            + fH0[sys.idx(s, 0, il-1, ik)]) * (0.5 * inv_delta_psi * inv_delta_l);

            } else if (ip == Npsi - 1) {
                // p at right boundary, l in interior: backward in p, central in l
                fpl[flat_idx] = (fH0[sys.idx(s, Npsi-1, il+1, ik)] 
                            - fH0[sys.idx(s, Npsi-1, il-1, ik)] 
                            - fH0[sys.idx(s, Npsi-2, il+1, ik)] 
                            + fH0[sys.idx(s, Npsi-2, il-1, ik)]) * (0.5 * inv_delta_psi * inv_delta_l);

            } else if (il == 0) {
                // l at left boundary, p in interior: central in p, forward in l
                fpl[flat_idx] = (fH0[sys.idx(s, ip+1, 1, ik)] 
                            - fH0[sys.idx(s, ip-1, 1, ik)] 
                            - fH0[sys.idx(s, ip+1, 0, ik)] 
                            + fH0[sys.idx(s, ip-1, 0, ik)]) * (0.5 * inv_delta_psi * inv_delta_l);

            } else if (il == Nl - 1) {
                // l at right boundary, p in interior: central in p, backward in l
                fpl[flat_idx] = (fH0[sys.idx(s, ip+1, Nl-1, ik)] 
                            - fH0[sys.idx(s, ip-1, Nl-1, ik)] 
                            - fH0[sys.idx(s, ip+1, Nl-2, ik)] 
                            + fH0[sys.idx(s, ip-1, Nl-2, ik)]) * (0.5 * inv_delta_psi * inv_delta_l);

            } else {
                // Both in interior: standard central difference (2nd order)
                fpl[flat_idx] = (fH0[sys.idx(s, ip+1, il+1, ik)] 
                            - fH0[sys.idx(s, ip-1, il+1, ik)]
                            - fH0[sys.idx(s, ip+1, il-1, ik)] 
                            + fH0[sys.idx(s, ip-1, il-1, ik)]) * (0.25 * inv_delta_psi * inv_delta_l);
            }
            
            // ----- ∂²f/∂ψ∂k -----
            if (ip == 0 && ik == 0) {
                // Both at left boundary: forward-forward (2nd order)
                fpk[flat_idx] = (fH0[sys.idx(s, 1, il, 1)] 
                            - fH0[sys.idx(s, 1, il, 0)] 
                            - fH0[sys.idx(s, 0, il, 1)] 
                            + fH0[sys.idx(s, 0, il, 0)]) * inv_delta_psi * inv_delta_k;

            } else if (ip == 0 && ik == Nk - 1) {
                // p at left, k at right: forward-backward (2nd order)
                fpk[flat_idx] = (fH0[sys.idx(s, 1, il, Nk-1)] 
                            - fH0[sys.idx(s, 1, il, Nk-2)] 
                            - fH0[sys.idx(s, 0, il, Nk-1)] 
                            + fH0[sys.idx(s, 0, il, Nk-2)]) * inv_delta_psi * inv_delta_k;

            } else if (ip == Npsi - 1 && ik == 0) {
                // p at right, k at left: backward-forward (2nd order)
                fpk[flat_idx] = (fH0[sys.idx(s, Npsi-1, il, 1)] 
                            - fH0[sys.idx(s, Npsi-2, il, 1)] 
                            - fH0[sys.idx(s, Npsi-1, il, 0)] 
                            + fH0[sys.idx(s, Npsi-2, il, 0)]) * inv_delta_psi * inv_delta_k;

            } else if (ip == Npsi - 1 && ik == Nk - 1) {
                // Both at right boundary: backward-backward (2nd order)
                fpk[flat_idx] = (fH0[sys.idx(s, Npsi-1, il, Nk-1)] 
                            - fH0[sys.idx(s, Npsi-2, il, Nk-1)] 
                            - fH0[sys.idx(s, Npsi-1, il, Nk-2)] 
                            + fH0[sys.idx(s, Npsi-2, il, Nk-2)]) * inv_delta_psi * inv_delta_k;

            } else if (ip == 0) {
                // p at left boundary, k in interior: forward in p, central in k
                fpk[flat_idx] = (fH0[sys.idx(s, 1, il, ik+1)] 
                            - fH0[sys.idx(s, 1, il, ik-1)] 
                            - fH0[sys.idx(s, 0, il, ik+1)] 
                            + fH0[sys.idx(s, 0, il, ik-1)]) * (0.5 * inv_delta_psi * inv_delta_k);

            } else if (ip == Npsi - 1) {
                // p at right boundary, k in interior: backward in p, central in k
                fpk[flat_idx] = (fH0[sys.idx(s, Npsi-1, il, ik+1)] 
                            - fH0[sys.idx(s, Npsi-1, il, ik-1)] 
                            - fH0[sys.idx(s, Npsi-2, il, ik+1)] 
                            + fH0[sys.idx(s, Npsi-2, il, ik-1)]) * (0.5 * inv_delta_psi * inv_delta_k);

            } else if (ik == 0) {
                // k at left boundary, p in interior: central in p, forward in k
                fpk[flat_idx] = (fH0[sys.idx(s, ip+1, il, 1)] 
                            - fH0[sys.idx(s, ip-1, il, 1)] 
                            - fH0[sys.idx(s, ip+1, il, 0)] 
                            + fH0[sys.idx(s, ip-1, il, 0)]) * (0.5 * inv_delta_psi * inv_delta_k);

            } else if (ik == Nk - 1) {
                // k at right boundary, p in interior: central in p, backward in k
                fpk[flat_idx] = (fH0[sys.idx(s, ip+1, il, Nk-1)] 
                            - fH0[sys.idx(s, ip-1, il, Nk-1)] 
                            - fH0[sys.idx(s, ip+1, il, Nk-2)] 
                            + fH0[sys.idx(s, ip-1, il, Nk-2)]) * (0.5 * inv_delta_psi * inv_delta_k);

            } else {
                // Both in interior: standard central difference (2nd order)
                fpk[flat_idx] = (fH0[sys.idx(s, ip+1, il, ik+1)] 
                            - fH0[sys.idx(s, ip-1, il, ik+1)]
                            - fH0[sys.idx(s, ip+1, il, ik-1)] 
                            + fH0[sys.idx(s, ip-1, il, ik-1)]) * (0.25 * inv_delta_psi * inv_delta_k);
            }
        }
    }}}
}