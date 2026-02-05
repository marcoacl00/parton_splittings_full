#pragma once
#include <iostream>
#include <vector>
#include <complex>
#include <stdexcept>
#include <cstddef>
#include <bits/stdc++.h>
#include <complex>
#include <chrono>

using dcomplex = std::complex<double>;

class Physis_J {
private:
    static constexpr double fm = 5.067;
    static inline constexpr dcomplex Iunit{0.0, 1.0};

    const double E_;
    const double qtilde_;
    const double z_;
    const double omega_;
    const double Lp_;
    const double mu_;
    double t_ = 1e-8;
    //double t_ = 0.01;
    double pmin_;
    double pmax_;
    const int mode_;
    int Np_ = 0;
    const int refineFactor_ = 1;
    const std::string vertex_;

    std::vector<double> P_;
    std::vector<dcomplex> Fsol_;


public:
    
    Physis_J(double E,
             double z,
             double qtilde,
             double Lp,
             double mu,
             int mode,
             std::string vertex);

    double E() const noexcept { return E_; }
    double z() const noexcept { return z_; }
    double omega() const noexcept { return omega_; }
    double qtilde() const noexcept { return qtilde_; }
    double Lp() const noexcept { return Lp_; }
    double mu() const noexcept { return mu_; }
    int mode() const noexcept { return mode_; }
    const std::string& vertex() const noexcept { return vertex_; }
    double t() const noexcept { return t_; }
    int Np() const noexcept { return Np_; }
    double pmin() const noexcept { return pmin_; }
    double pmax() const noexcept { return pmax_; }


    void set_dim(int Np_);
    void init_fsol();

    
    const std::vector<double>& P() const noexcept { return P_; }
    const std::vector<dcomplex>& Fsol() const noexcept { return Fsol_; }


    std::vector<dcomplex> source_term() const;

    void set_fsol(const std::vector<dcomplex>& arr);

    void increase_t(double dt) {
        t_ += dt;
    }

    void set_t(double t0) {
        t_ = t0;
    }

    void set_pmin(double p){
        pmin_ = p;
    }

    void set_pmax(double p){
        pmax_ = p;
    }

};
