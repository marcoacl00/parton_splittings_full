#include "Physis_J.hpp"
#include <iostream>


Physis_J::Physis_J(double E,
                   double z,
                   double qtilde,
                   double Lp,
                   double mu,
                   int mode,
                std::string vertex)
    : E_(E * fm),
      qtilde_(qtilde * fm * fm),
      z_(z),
      omega_(E_ * z_ * (1.0 - z_)),
      Lp_(Lp * fm),
      mu_(mu * fm),
      mode_(mode),
      vertex_(vertex){}

void Physis_J::set_dim(int Np) {
    if (Np < 2) {
        throw std::runtime_error("Np must be >= 2");
    }

    Np_ = Np;

    P_.resize(Np_);

    for (int i = 0; i < Np_; ++i) {
        P_[i] = (Lp_ * i) / double(Np_ - 1);
    }
}

void Physis_J::init_fsol() {
    if (Np_ == 0) {
        throw std::runtime_error("set_dim() first");
    }

    Fsol_.assign(Np_, dcomplex(0.0, 0.0));
    std::cerr << "Initialized solution grid with shape(" << Np_ << ")\n";
}


std::vector<dcomplex> Physis_J::source_term() const {
    std::vector<dcomplex> S(Np_, dcomplex(0.0, 0.0));

    for (int ix = 0; ix < Np_; ++ix) {
        S[ix] = -Iunit * P_[ix];
    }

    return S;
}

void Physis_J::set_fsol(const std::vector<dcomplex>& arr) {
    if (arr.size() != Fsol_.size()) {
        throw std::runtime_error("Fsol size mismatch");
    }

    Fsol_ = arr;
}

