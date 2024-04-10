#include "rref.hpp"

#include <armadillo>

using namespace arma;

arma::Mat<double> mat_to_rref(const arma::Mat<double>& b, const double EPSILON)
{
  // based on https://github.com/t3nsor/codebook

  mat a = b;

  uword n_rows = a.n_rows;
  uword n_cols = a.n_cols;
  uword target_row = 0;

  for (uword col = 0; col < n_cols; ++col)
  {
    uword source_row = target_row;

    for (uword row = target_row + 1; row < n_rows; ++row)
    {
      if (fabs(a(row, col)) > fabs(a(source_row, col)))
      {
        source_row = row;
      }
    }

    if (fabs(a(source_row, col)) < EPSILON) continue;

    a.swap_rows(source_row, target_row);

    double s = 1.0 / a(target_row, col);

    for (uword col = 0; col < n_cols; ++col)
    {
      a(target_row, col) *= s;
    }

    for (uword row = 0; row < n_rows; ++row)
    {
      if (row != target_row)
      {
        double s = a(row, col);

        for (int col = 0; col < n_cols; ++col)
        {
          a(row, col) -= s * a(target_row, col);
        }
      }
    }

    if (++target_row == n_rows) break;
  }

  return a;
}
