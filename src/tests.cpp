#include <iostream>
#include "spline.h"

int main(){

    std::vector<double> vec = {0.1 , 0.3, 0.7, 0.9, 1.0, 1.3};
    std::vector<double> dom = {0.6, 0.5, 0.4, 0.3, 0.2, 0.1};
    // invert vector dom
    std::reverse(dom.begin(), dom.end());

    tk::spline v(dom, vec);

    double val = v(0.25);
    std::cout << val << std::endl;

    return 0;
}