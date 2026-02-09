#include <bits/stdc++.h>
#include <complex>
#include <chrono>
#include "Physis_J.hpp"
#include "Physis.hpp"
#include "hamiltonian_j.hpp"
#include "hamiltonian.hpp"
#include "faber.hpp"

//#ifdef _OPENMP
#include <omp.h>
//#endif

#include "spline.h"

using namespace std;
using dcomplex = complex<double>;
const dcomplex I(0.0, 1.0);

int main(int argc, char* argv[]) {

    // physics parameters
    double E  = 100.0;
    double z = 0.5;
    double qtilde = 1.5;
    double Lp =  4.0* E * z * (1 - z);
    double Lk =  0.325 * Lp;
    double Ll = 0.3 *  Lk;
    double mu = 0.6;
    string vertex = "gamma_qq";
    string Nc_mode = "LNc";

    int mode = 0;
    if (argc > 1) {
        try {
            mode = std::stoi(argv[1]);
        } catch (const std::exception &e) {
            std::cerr << "Invalid mode argument '" << argv[1] << "'. Using default mode = " << mode << "\n";
        }
    }
    if (mode < 0 || mode > 2) {
        std::cerr << "Mode out of expected range {0, 1, 2}. Using default mode = 0\n";
        mode = 1;
    }

    string file_name_2D;
    string file_name_3D;
    double p_min_coeff;
    double p_max_coeff;


    if (mode ==  0){
        file_name_2D = "./data/fsol_final_yuk.dat";
        file_name_3D = "./data/fsol3D_final_yuk.dat";
        p_min_coeff = 0.8;
        p_max_coeff = 30.0;
    }
    else if (mode == 1){
        file_name_2D = "./data/fsol_final_htl.dat";
        file_name_3D = "./data/fsol3D_final_htl.dat";
        p_min_coeff = 0.8;
        p_max_coeff = 30.0;
    }
    else{
         file_name_2D = "./data/fsol_final_ho.dat";
        file_name_3D = "./data/fsol3D_final_ho.dat";
        p_min_coeff = 4.0;
        p_max_coeff = 10.0;
    }

    // initialize simulation objects 
    Physis_J sis(E, z, qtilde, Lp, mu, mode, vertex);
    Physis sis_f(E, z, qtilde, Lk, Ll, mu, mode, vertex, Nc_mode);
    // set dimensions
    sis.set_dim(512);
    sis_f.set_dim(180, 60, 4);


    // initialize the solution
    sis.init_fsol();
    sis_f.init_fsol();
    

    // maximum time (medium length)
    double t_L = 1.0;

    // time step
    
    double ht = 0.01;
    double ht_j = 0.5 * ht; //has to be half because we need the source term at t+ht/2 for faber expansion

    //set convolution integration limits
    sis.set_pmin(p_min_coeff * sis.mu());
    sis.set_pmax(p_max_coeff * sis.mu());

    sis_f.set_pmin(0);
    sis_f.set_pmax(0.1 * p_max_coeff * sis.mu());

    // define vector with times
    vector<double> time_list;
    for (double tt = 1e-8; tt < t_L; tt += ht) time_list.push_back(tt);

    // faber params
    double lambF, lambF_1;
    dcomplex gamma0, gamma1, gamma0_f, gamma1_f;
    faber_params3D_ker(sis, lambF, gamma0, gamma1); // note that faber parameters are independent of chosen delta t
    faber_params3D_full(sis_f, lambF_1, gamma0_f, gamma1_f);
    double one_lamb = 1.0 / lambF;
    double one_lamb_1 = 1.0 / lambF_1;


    cerr << "Faber params: lamb = " << lambF << " gamma0 = " << gamma0 << " gamma1 = " << gamma1 << "\n";
    cerr << "Faber params (full): lamb = " << lambF_1 << " gamma0 = " << gamma0_f << " gamma1 = " << gamma1_f << "\n";

    // pre-compute coeff_array for full step
    double threshold = 1e-7;
    vector<dcomplex> coeff_array;
    int m = 0;
    do {
        coeff_array.push_back(coeff(m, ht_j, gamma0, gamma1, lambF));
        ++m;
    } while ( (abs(coeff_array.back()) > threshold) || m < 3 );

    // pre-compute coeff_array_2 for half step
    vector<dcomplex> coeff_array_2;
    int o = 0;
    do {
        coeff_array_2.push_back(coeff(o, ht_j * 0.5, gamma0, gamma1, lambF));
        ++o;
    } while ( (abs(coeff_array_2.back()) > threshold) || o < 3 );


    vector<dcomplex> coeff_array_f;
    int n = 0;
    do {
        coeff_array_f.push_back(coeff(n, ht, gamma0_f, gamma1_f, lambF_1));
        ++n;
    } while ( (abs(coeff_array_f.back()) > threshold) || n < 3 );

    vector<dcomplex> coeff_array_2_f;
    int p = 0;
    do {
        coeff_array_2_f.push_back(coeff(p, ht * 0.5, gamma0_f, gamma1_f, lambF_1));
        ++p;
    } while ( (abs(coeff_array_2_f.back()) > threshold) || p < 3 );

    cerr << "Number of Faber coeffs for ht: " << coeff_array.size() << "\n";
    cerr << "Number of Faber coeffs for ht/2: " << coeff_array_2.size() << "\n";
    cerr << "Number of Faber coeffs for ht (full): " << coeff_array_f.size() << "\n";
    cerr << "Number of Faber coeffs for ht/2 (full): " << coeff_array_2_f.size() << "\n";

    // building potential moments to approximate the integral in the first (small, importat)
    // interval [0, pmin] 
    vector<double> taylor_coeffs_0(sis.Np()), taylor_coeffs_1(sis.Np()), taylor_coeffs_2(sis.Np());

    for (int ix = 0; ix < sis.Np(); ++ix) {
        double p = sis.P()[ix];
        taylor_coeffs_0[ix] = compute_f_r_pc(0, p, sis.z(), 0.01,  sis.pmin(), 128, 64, sis.mu(), sis.mode(), sis.vertex());
        taylor_coeffs_1[ix] = compute_f_r_pc(1, p, sis.z(), 0.01,  sis.pmin(), 128, 64, sis.mu(), sis.mode(), sis.vertex());
        taylor_coeffs_2[ix] = compute_f_r_pc(2, p, sis.z(), 0.01,  sis.pmin(), 128, 64, sis.mu(), sis.mode(), sis.vertex());
    }

    /*for (int ix = 0; ix < sis.Np(); ++ix) {
        cout << "ix=" << ix 
             << " taylor_coeffs_0=" << taylor_coeffs_0[ix]
             << " taylor_coeffs_1=" << taylor_coeffs_1[ix]
             << " taylor_coeffs_2=" << taylor_coeffs_2[ix]
             << endl;
    }*/



    size_t cont = 0;
    auto tstart = chrono::high_resolution_clock::now();

    //-----------------------//
    //--- SIMULATION LOOP ---//
    //-----------------------//

    for (double tt : time_list) {
        auto nHom = sis.source_term();

        // -------------------------------//
        // In-out contribution first


        vector<dcomplex> fsol_t = sis.Fsol();
        vector <dcomplex> fsol_t_dt_2 = fsol_t; // fsol at t + dt/2
        vector<dcomplex> fsol_t_dt = fsol_t; // fsol at time

        for(int _ = 0; _ < 2; ++_){ // two half steps to reach t + ht
            // Simpson first term

            vector<dcomplex> nFsol = sis.Fsol();
            for (size_t k = 0; k < nFsol.size(); ++k) nFsol[k] += nHom[k] * (ht_j / 6.0);
            sis.set_fsol(nFsol);

            // Full-step Faber
            auto f_sol_n = faber_expandJ(sis, ht_j, gamma0, gamma1, one_lamb, coeff_array, 
                                            taylor_coeffs_0, taylor_coeffs_1, taylor_coeffs_2);

            // midpoint term
            auto nHom_dt_2 = sis.source_term();
            Physis_J sis_aux = sis;
            sis_aux.set_fsol(nHom_dt_2);
            auto f_term_2 = faber_expandJ(sis_aux, ht_j/2.0, gamma0, gamma1, one_lamb, coeff_array_2, 
                                            taylor_coeffs_0, taylor_coeffs_1, taylor_coeffs_2);

            // final non-homogeneous
            auto nHom_end = sis.source_term();

            // complete Simpson update
            vector<dcomplex> newFsol(f_sol_n.size(), dcomplex(0.0, 0.0));
            for (size_t k = 0; k < newFsol.size(); ++k) {
                newFsol[k] = f_sol_n[k] + (4.0/6.0) * f_term_2[k] * ht_j + nHom_end[k] * (ht_j / 6.0);
            }

            sis.set_fsol(newFsol);

            // NaN checks
            for (auto &v : nHom) {
                if (isnan(v.real()) || isnan(v.imag())) {
                    cerr << "Warning: NaN detected in nHom at t = " << sis.t() << "\n";
                    break;
                }
            }
            for (auto &v : newFsol) {
                if (isnan(v.real()) || isnan(v.imag())) {
                    throw runtime_error("NaN detected in nFsol at t = " + to_string(sis.t()));
                }
            }

            if(_ == 0){
                fsol_t_dt_2 = sis.Fsol();
            }
            else{
                fsol_t_dt = sis.Fsol();
            }

            sis.increase_t(ht_j);



        } // end of half-step loop

        // fsol_mid = (sis.Fsol() + fsol_save)



        // -------------------------------//
        // FULL 3D EVOLUTION STEP
        // after completing the in-out contribution, we move to the full 3D evolution
        // in-out will be ued as source term for the full evolution
        // -------------------------------//

        auto nHom_f = sis_f.source_term(sis.P(), fsol_t_dt, sis_f.t());  // source term at time t + ht
        //repeat logic for full 3D evolution

        // Simpson first term
        vector<dcomplex> nFsol_f = sis_f.Fsol();
        
        for(int sig = 0; sig < sis_f.Nsig(); ++sig){
            for(int ip = 0; ip < sis_f.NPsi(); ++ip){
                for(int il = 0; il < sis_f.Nl(); ++il){
                    for(int ik = 0; ik < sis_f.Nk(); ++ik){
                        int idx = sis_f.idx(sig, ip, il, ik);
                        int s_idx = sis_f.idx(0, ip, il, ik);
                        nFsol_f[idx] += nHom_f[s_idx] * (ht / 6.0);
                    }
                }
            }
        }

        sis_f.set_fsol(nFsol_f);
        
        auto f_sol_n_f = faber_expand_full(sis_f, ht, gamma0_f, gamma1_f, one_lamb_1, coeff_array_f);

        // midpoint term
        auto nHom_dt_2_f = sis_f.source_term(sis.P(), fsol_t_dt_2, sis_f.t() + ht/2.0); // source term at time t + ht/2

        Physis sis_aux_f = sis_f; 
        // set fsol as the source term copied sig times
        vector<dcomplex> nHom_dt_2_f_copied(sis_f.Fsol().size(), dcomplex(0.0, 0.0));
        
        for(int sig = 0; sig < sis_f.Nsig(); ++sig){
            for(int ip = 0; ip < sis_f.NPsi(); ++ip){
                for(int il = 0; il < sis_f.Nl(); ++il){
                    for(int ik = 0; ik < sis_f.Nk(); ++ik){
                        int idx = sis_f.idx(sig, ip, il, ik);
                        int s_idx = sis_f.idx(0, ip, il, ik);
                        nHom_dt_2_f_copied[idx] = nHom_dt_2_f[s_idx];
                    }
                }
            }
        }
        sis_aux_f.set_fsol(nHom_dt_2_f_copied);

        auto f_term_2_f = faber_expand_full(sis_aux_f, ht/2.0, gamma0_f, gamma1_f, one_lamb_1, coeff_array_2_f);

        // final non-homogeneous
        auto nHom_end_f = sis_f.source_term(sis.P(), fsol_t, sis_f.t() + ht);   

        // complete Simpson update
        vector<dcomplex> newFsol_f(f_sol_n_f.size(), dcomplex(0.0, 0.0));

        for (int sig = 0; sig < sis_f.Nsig(); ++sig){
            for(int ip = 0; ip < sis_f.NPsi(); ++ip){
                for(int il = 0; il < sis_f.Nl(); ++il){
                    for(int ik = 0; ik < sis_f.Nk(); ++ik){
                        int idx = sis_f.idx(sig, ip, il, ik);
                        int s_idx = sis_f.idx(0, ip, il, ik);
                        newFsol_f[idx] = f_sol_n_f[idx] + (4.0/6.0) * f_term_2_f[idx] * ht + nHom_end_f[s_idx] * (ht / 6.0);
                    }
                }
            }
        }

        sis_f.set_fsol(newFsol_f);

        // NaN checks
        for (auto &v : nHom_f) {
            if (isnan(v.real()) || isnan(v.imag())) {
                cerr << "Warning: NaN detected in nHom_f at t = " << sis_f.t() << "\n";
                break;  

            }
        }

        sis_f.increase_t(ht);

        ++cont;
        if (cont % 1 == 0) {
            cerr << "Completed step " << cont << ", t = " << sis_f.t() << "\n";
        }

    }


    auto tend = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = tend - tstart;
    cerr << "Simulation completed in " << elapsed.count() << " s, steps: " << cont << "\n";

    //-----------------------//
    //---------END OF--------//
    //--- SIMULATION LOOP ---//
    //-----------------------//


    // ---- Save results of in-out in .dat files...-----//
    {
        // append parameter values to filename before the extension
        string base = file_name_2D;
        string ext;
        auto p = base.rfind('.');
        if (p != string::npos) {
            ext = base.substr(p);
            base = base.substr(0, p);
        }
        std::ostringstream ss;
        ss << base << "_" <<vertex << "_" "_E_" << E << "_z_" << z << "_q_" << qtilde << "_mu_" << mu << ext;
        file_name_2D = ss.str();
    }

    
    ofstream ofs(file_name_2D);
    for (int ix = 0; ix < sis.Np(); ++ix) {
                auto v = sis.Fsol()[ix];
                ofs << " "  << v.real() << " " << v.imag() << "\n";
        }
    ofs.close();

    cout << "Result saved in " << file_name_2D << endl;

    // ---- Save results of full 3D in .dat files...-----//
    {
        // append parameter values to filename before the extension
        string base = file_name_3D;
        string ext;
        auto p = base.rfind('.');
        if (p != string::npos) {
            ext = base.substr(p);
            base = base.substr(0, p);
        }
        std::ostringstream ss;
        ss << base << "_" <<vertex << "_" "_E_" << E << "_z_" << z << "_q_" << qtilde << "_mu_" << mu << ext;
        file_name_3D = ss.str();
    }

    //only sig = 1, il = 0 integrated in Psi is necessary, as a function of k
    ofstream ofs_f(file_name_3D);

    for (int ik = 0; ik < sis_f.Nk(); ++ik) {
        ofs_f << " " << sis_f.Fsol()[sis_f.idx(1, 0, 0, ik)].real() << "\n";
    }
    ofs_f.close();

    return 0;
}