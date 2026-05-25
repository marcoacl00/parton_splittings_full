#include "Physis.hpp"
#include <iostream>


Physis::Physis(double E,
           double z,
           double qtilde,
           double Lk,
           double Ll,
           double mu,
           int mode,
           std::string vertex,
        std::string Nc_mode)
    : E_(E * fm),
      qtilde_(qtilde * fm * fm),
      z_(z),
      omega_(E_ * z_ * (1.0 - z_)),
      Lk_(Lk * fm),
      Ll_(Ll * fm),
      mu_(mu * fm),
      mode_(mode),
      vertex_(vertex),
      Nc_mode_{Nc_mode}{}


void Physis::set_dim(int Nk, int Nl, int Npsi) {
    if (Nk < 3 || Nl < 3 || Npsi < 3) {
        throw std::runtime_error("N_(k/l/psi) must be >= 3");
    }
    
    Nk_ = Nk;
    Nl_ = Nl;
    Npsi_ = Npsi;

    K_.resize(Nk_);
    L_.resize(Nl_);
    Psi_.resize(Npsi_);
    Cos_Psi_.resize(Npsi_);

    // we are using ik for the index in the k axis
    for (int ik = 0; ik < Nk_; ++ik) {
        K_[ik] = (Lk_ * (ik) / double(Nk_ - 1));
    }

    // il for the l axis
    for (int il = 0; il < Nl_; ++il) {
        L_[il] = (Ll_ * il) / double(Nl_ - 1);
    }

    // ... and ip for the psi axis ∈ [0, pi]
    for (int ip = 0; ip < Npsi_; ++ip) {
        Psi_[ip] = (M_PI * ip) / double(Npsi_ - 1);
        Cos_Psi_[ip] = std::cos(Psi_[ip]); // initialize the cos array to save computational time
    }

    std::cout << "Initialized grid with Nk=" << Nk_ << ", Nl=" << Nl_ << ", Npsi=" << Npsi_ << "\n";
    std::cout << "K range: [" << K_.front() << ", " << K_.back() << "]\n";
    std::cout << "L range: [" << L_.front() << ", " << L_.back() << "]\n";
    std::cout << "Psi range: [" << Psi_.front() << ", " << Psi_.back() << "]\n";


}

void Physis::init_fsol() {
    
    if (vertex_ == "gamma_qq"){
        n_sig = 2;
    }

    else if (vertex_ == "q_qg"){
        if (Nc_mode_ == "FNc"){
            n_sig = 3;
        }
        else{
            n_sig = 2;
        }
    }

    else if (vertex_ == "g_gg"){
        if (Nc_mode_ == "FNc"){
            n_sig = 6;
        }
        else{
            n_sig = 2;
        }
    }

    else {throw std::runtime_error("Insert a valid vertex");}

    Fsol_.assign(n_sig * Nk_ * Nl_ * Npsi_, dcomplex(0.0, 0.0));
    std::cerr << "Initialized solution grid with shape(" << n_sig<<"x"<< Nk_ << "x" << Nl_ << "x" << Npsi_ << ")\n";

}


void Physis::set_fsol(const std::vector<dcomplex>& arr) {
    if (arr.size() != Fsol_.size()) {
        throw std::runtime_error("Fsol size mismatch");
    }

    Fsol_ = arr;
}

std::vector<dcomplex> Physis::source_term(const std::vector<double>& p, const std::vector<dcomplex>& j_p, const double time){

    // split j_p into real and imaginary parts
    std::vector<double> j_real(j_p.size()), j_imag(j_p.size());
    for (std::size_t i = 0; i < j_p.size(); ++i) {
        j_real[i] = std::real(j_p[i]);
        j_imag[i] = std::imag(j_p[i]);
    }

    if (p.size() < 2 || j_real.size() != p.size() || j_imag.size() != p.size()) {
        throw std::runtime_error("Interpolation arrays must have same size >= 2");
    }

    // Linear interpolation lambdas (clamped outside the provided p range)
    auto j_r = [&](double x) -> double {
        if (x <= p.front()) return j_real.front();
        if (x >= p.back())  return j_real.back();
        for (std::size_t i = 1; i < p.size(); ++i) {
            if (x <= p[i]) {
                double x0 = p[i-1], x1 = p[i];
                double y0 = j_real[i-1], y1 = j_real[i];
                double t = (x - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }
        return j_real.back(); // fallback (shouldn't reach)
    };

    auto j_i = [&](double x) -> double {
        if (x <= p.front()) return j_imag.front();
        if (x >= p.back())  return j_imag.back();
        for (std::size_t i = 1; i < p.size(); ++i) {
            if (x <= p[i]) {
                double x0 = p[i-1], x1 = p[i];
                double y0 = j_imag[i-1], y1 = j_imag[i];
                double t = (x - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }
        return j_imag.back();
    };

    double delta_l = Ll_ / (Nl_ - 1);

    double k;
    double l;
    double psi;
    double cos_psi;
    double R;
    dcomplex pref;

    double omega = omega_;
    double qtilde = qtilde_;
    double z = z_;
    double CF = (Nc_mode_ == "LNc") ? 1.5 : 4.0/3.0;
    double CA = 3.0;

    dcomplex Omega = (1.0 - Iunit) * 0.5 * std::sqrt(((1-z) * CA + z*z * CF) * qtilde / omega);
    
    std::vector<dcomplex> s_term;
    s_term.resize(Npsi_*Nk_*Nl_);

    for(int ip = 0; ip < Npsi_; ip++){
        psi = Psi_[ip];
        cos_psi = Cos_Psi_[ip];
        for(int il = 0; il < Nl_; il++){
            l = L_[il];
            double l2 = l * l;
            for(int ik = 0; ik < Nk_; ik++){
                k = K_[ik];
                double k2 = k * k;


                R = std::sqrt(k2 + l2 + 2.0*k*l*cos_psi); // add small number to avoid singularity at R=0

                pref = (k2 - l2); 

                double jr = j_r(R);
                double ji = j_i(R);

                // note that idx(0, ip, ik, ip) = idx3(ip, ik, il).
                if (R == 0) continue;
                s_term[idx(0, ip, il, ik)] =  pref * dcomplex(-ji, jr); // source multiplies by -i*/
                // double R2 = k2 + l2 + 2.0 * k * l * cos_psi; // add small number to avoid singularity at R=0
                // double r_ratio = (k2 - l2)/R2;

                // pref = -Iunit * 2.0 * omega;
                // int idxc = idx(0, ip, il, ik);

                // ////s_term[idxc] =  pref * dcomplex(-ji, jr); // source multiplies by -i*/



                // if (R2 < delta_l*delta_l) {
                //      s_term[idxc] = -Iunit / (2 * omega * Omega) * std::tan(Omega * time) * (2 * k2 + 2 * k * l * cos_psi) ;
                // }

                // else{
                //      s_term[idxc] = r_ratio * pref * (1.0 - std::exp(-Iunit / (2 * omega * Omega) * std::tan(Omega * time) * R2));
                // }


                /*if(il == 0){std::cout << "Source term" << s_term[idx(0, ip, il, ik)] << " at (ip, ik, il)=(" << ip << ", " << ik << ", " << il << "): "
                    << "R2=" << R2 << ", pref=" << pref 
                    << "\n";
                }*/
                
            }
        }
    }

    return s_term;
}
