#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstddef>
#include <bits/stdc++.h>
#include <complex>
#include <chrono>
#include "spline.h"


using dcomplex = std::complex<double>;

class Physis {
private:
    static constexpr double fm = 5.067;
    static inline constexpr dcomplex Iunit{0.0, 1.0};

    // these will be set at initializing
    const double E_;
    const double z_;
    const double omega_;
    const double qtilde_;
    const double Lk_;
    const double Ll_;
    const double mu_;
    const int mode_;
    const std::string Nc_mode_;
    const std::string vertex_;
    
    // depends on vertex + Nc mode
    int n_sig;

    double t_ = 1e-8;

    int Nk_ = 0;
    int Nl_ = 0;
    int Npsi_ = 0;
    double pmin_ = 0;
    double pmax_ = 0;

    std::vector<double> K_;
    std::vector<double> L_;
    std::vector<double> Psi_;
    std::vector<double> Cos_Psi_;

    std::vector<dcomplex> Fsol_;

public:
    Physis(double E,
           double z,
           double qtilde,
           double Lk,
           double Ll,
           double mu,
           int mode,
           std::string vertex,
           std::string Nc_mode
        );

    // getters 
    double E() const noexcept { return E_; }
    double z() const noexcept { return z_; }
    double omega() const noexcept { return omega_; }
    double qtilde() const noexcept { return qtilde_; }
    double Lk() const noexcept { return Lk_; }
    double Ll() const noexcept { return Ll_; }
    double mu() const noexcept { return mu_; }
    int mode() const noexcept { return mode_; }
    const std::string& vertex() const noexcept { return vertex_; }
    double t() const noexcept { return t_; }
    int Nk() const noexcept { return Nk_; }
    int Nl() const noexcept { return Nl_; }
    int NPsi() const noexcept { return Npsi_; }
    int Nsig() const noexcept { return n_sig; }
    double pmin() const noexcept { return pmin_; }
    double pmax() const noexcept { return pmax_; }
    const std::string& Ncmode() const noexcept { return Nc_mode_; }

    const std::vector<double>& K() const noexcept { return K_; }
    const std::vector<double>& L() const noexcept { return L_; }
    const std::vector<double>& Psi() const noexcept { return Psi_; }
    const std::vector<double>& Cos_Psi() const noexcept { return Cos_Psi_; }
    const std::vector<dcomplex>& Fsol() const noexcept { return Fsol_; }

    // mutation 
    inline std::size_t idx(int m_sig, int ip, int il, int ik) const {
        if (m_sig < 0 || m_sig >= n_sig) {
            throw std::out_of_range("Physis::idx: m_sig out of range");
        }
        if (ip < 0 || ip >= Npsi_) {
            throw std::out_of_range("Physis::idx: psi index out of range");
        }
        if (ik < 0 || ik >= Nk_) {
            throw std::out_of_range("Physis::idx: ik out of range");
        }
        if (il < 0 || il >= Nl_) {
            throw std::out_of_range("Physis::idx: il out of range");
        }

        // Layout chosen: ((m_sig * Npsi_ + psi) * Nl_ + il) * Nk_ + ik
        // = m_sig * Npsi_ * Nk_ * Nl_ + psi * Nk_ * Nl_ + il *  Nk_ + ik
        return static_cast<std::size_t>((((m_sig * Npsi_) + ip) * Nl_ + il) * Nk_ + ik);
    }

    void increase_t(double dt) {
        t_ += dt;
    }

    void set_t(double t0) {
        t_ = t0;
    }

    void set_dim(int Nk, int Nl, int Npsi);

    void init_fsol();
    void set_fsol(const std::vector<dcomplex>& arr);

    void set_pmin(double p){
        pmin_ = p;
    }

    void set_pmax(double p){
        pmax_ = p;
    }

    std::vector<dcomplex> source_term(const std::vector<double>& p, const std::vector<dcomplex>& j_p);

};


