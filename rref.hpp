#pragma once

namespace arma {
    template<typename eT> class Mat;
}

arma::Mat<double> mat_to_rref(const arma::Mat<double>& b, const double EPSILON = 1e-8);
