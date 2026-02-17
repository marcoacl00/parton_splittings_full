#include <iostream>
#include "spline.h"
#include <vector>

using namespace std;

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


int main(){

    int N = 12;
    static const GaussLegendre GL_ANGULAR(N); 

    // integrating cos(theta) from 0 to 2*pi
    double integral = 0.0;
    for (int j = 0; j < N; ++j) {
        double theta = M_PI + M_PI * GL_ANGULAR.nodes[j];
        cout << "theta: " << theta << " cos(theta): " << cos(theta) << endl;
        double w = M_PI * GL_ANGULAR.weights[j];
        integral += cos(theta) * w;
    }

    cout << "Integral of cos(theta) from 0 to 2*pi: " << integral << endl;

    return 0;
}