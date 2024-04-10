/******************************************************************************
        BOOK: Linear Algebra: Theory, Intuition, Code
      AUTHOR: Mike X Cohen
     WEBSITE: sincxpress.com
******************************************************************************/

#include <iostream>
#include <ranges>
#include <format>
#include <algorithm>

#define ARMA_PRINT_EXCEPTIONS
#include <armadillo>

#include <matplot/matplot.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "rref.hpp"

using namespace std::complex_literals;
using namespace arma;
namespace plt = matplot;

std::vector<std::vector<double>> mat_to_vector_2d(const mat& A)
{
  auto a_row_to_vector = [&](s64 row_index) {return conv_to<std::vector<double>>::from(A.row(row_index));};
  auto a_rows_view = regspace<ivec>(0, 1, A.n_rows - 1) | std::views::transform(a_row_to_vector);
  return std::vector(a_rows_view.begin(), a_rows_view.end());
}

// https://stackoverflow.com/a/4609795
template <typename T> int signum(T val) { return (T(0) < val) - (val < T(0)); }

void chapter_02()
{
  cout << "CHAPTER: Vectors (chapter 2)" << endl << endl;

  // Section 2.1, code block 2.2

  // creating scalars (numeric variables)
  auto aScalar = 4.l;

  // Section 2.2, code block 2.4

  // create a vector
  vec v = { 2, -1 };

  // plot it
  auto f = plt::figure(true);
  f->width(f->width() * 2);
  f->height(f->height() * 2);
  plt::plot({ 0.,v[0] }, { 0.,v[1] });
  plt::axis(plt::square);
  plt::axis({ -3.,3.,-3.,3 });
  plt::grid(plt::on);
  plt::show();

  // Section 2.2, code block 2.6

  // row vector
  rowvec v1 = { 2, 5, 4, 7 };

  // column vector
  colvec v2 = { 2, 5, 4, 7 };

  // Section 2.3, code block 2.8
  {
    // start with a row vector
    rowvec v1 = { 2, 5, 4, 7 };

    // transpose to a column vector
    auto v2 = v1.t();
  }

  // Section 2.5, code block 2.10
  {
    // two vectors
    vec v1 = { 2, 5, 4, 7, };
    vec v2 = { 4, 1, 0, 2, };

    // scalar-multiply and add
    vec v3 = 4 * v1 - 2 * v2;
  }

  // Section 2.9, code block 2.12
  {
    // the "base" vector
    vec v = { 1, 2 };

    auto f = plt::figure(true); //clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);

    plt::hold(plt::on);

    plt::plot({ 0., v[0] }, { 0., v[1] })->line_width(2);

    for (int i = 0; i < 10; ++i)
    {
      // random scalar
      auto s = plt::randn(0, 1);
      auto sv = s * v;

      // plot that one on top
      plt::plot({ 0., sv[0] }, { 0., sv[1] })->line_width(2);
    }

    plt::grid(plt::on);
    plt::axis(plt::square);
    plt::axis({ -4., 4., -4., 4. });
    plt::show();
  }

  cout << endl;

  // done.
}

void chapter_03()
{
  cout << "CHAPTER: Vector Multiplications (chapter 3)" << endl << endl;

  // Section 3.1, code block 3.2

  // create two vectors
  vec v1 = { 2., 5., 4., 7. };
  vec v2 = { 4., 1., 0., 2. };

  // dot product between them
  double dp = dot(v1, v2);

  // Section 3.5, code block 3.4

  // some scalars
  double l1 = 1;
  double l2 = 2;
  double l3 = -3;

  {
    // some vectors
    rowvec v1 = { 4, 5, 1 };
    rowvec v2 = { -4, 0, -4 };
    rowvec v3 = { 1, 3, 2 };

    // a linear weighted combination
    cout << "l1*v1 + l2*v2 + l3*v3 = " << endl << (l1 * v1 + l2 * v2 + l3 * v3) << endl;
  }

  // Section 3.6, code block 3.6
  {
    // two column vectors
    colvec v1 = { 2, 5, 4, 7 };
    rowvec v2 = { 4, 1, 0, 2 };

    // outer product
    auto op = (v1.t() * v2.t()).eval();
  }

  // Section 3.7, code block 3.8
  {
    // two vectors
    vec v1 = { 2, 5, 4, 7 };
    vec v2 = { 4, 1, 0, 2 };

    // Hadamard multiplication
    vec v3 = v1 % v2;
  }
  // Section 3.9, code block 3.10

  // a vector
  vec v = { 2, 5, 4, 7 };

  // its norm
  auto vMag = norm(v);

  // the unit vector
  auto v_unit = v / vMag;

  // Section 3.13, code block 3.12
  {
    // three vectors
    vec v1 = { 1, 2, 3, 4, 5 };
    vec v2 = { 2, 3, 4, 5, 6 };
    vec v3 = { 3, 4, 5, 6, 7 };

    // linear weighted combo
    vec w = { -1, 3, -2 };
    cout << "v1*w(0) + v2*w(1) + v3*w(2) = " << endl << (v1 * w(0) + v2 * w(1) + v3 * w(2)) << endl;
  }

  // Section 3.13, code block 3.14
  {
    rowvec v = { 7, 4, -5, 8, 3 };
    auto o = ones(size(v));

    // average via dot product
    cout << "dot(v, o) / v.n_elem = " << (dot(v, o) / v.n_elem) << endl;
  }

  // Section 3.13, code block 3.16
  {
    // vector
    vec v = { 7, 4, -5, 8, 3 };

    // random weighting vector
    vec w = randu(size(v));

    // weighted dp
    auto wAve = dot(v, w / sum(w));
  }

  cout << endl;

  // done.
}

void chapter_05()
{
  cout << "CHAPTER: Matrices (chapter 5)" << endl << endl;

  // Section 5.4, code block 5.2

  // create a matrix of random numbers
  mat A = randn(2, 5);

  // two ways to transpose
  mat At1 = A.t();
  mat At2 = trans(A);

  // Section 5.5, code block 5.4

  // identity matrix
  mat I = eye(4, 4);

  // ones matrix
  mat O = ones(4);

  // zeros matrix
  mat Z = zeros(4);

  // Section 5.5, code block 5.6

  // diagonal matrix from a vecot
  mat D = diagmat(vec({ 1, 2, 3, 5 }));

  // diagonal of a full matrix
  mat R = randn(3, 4);
  mat d = diagmat(R);

  // Section 5.5, code block 5.8
  {
    // random numbers matrix
    mat A = randn(3, 5);

    // another random matrix
    mat B = randn(3, 4);

    // augmented matrix
    mat AB = join_rows(A, B);
  }

  // Section 5.5, code block 5.10
  {
    // Create a matrix
    mat A = randn(5, 5);

    // extract the lower triangle
    mat L = trimatl(A);

    // extract the upper triangle
    mat U = trimatu(A);
  }

  // Section 5.5, code block 5.12

  // start from this vector
  vec t = regspace(1, 3);

  // toeplitz
  mat T = toeplitz(t);

  // hankel
  // H = hankel(t,t([end 1:end-1]));

  // Section 5.8, code block 5.14
  {
    // scalar to shift by
    double l = .01;

    // identity matrix
    mat I = eye(4, 4);

    // just some matrix
    mat A = randn(4, 4);

    // shifted version
    mat As = A + l * I;
  }

  // Section 5.9, code block 5.16
  {
    // a matrix
    mat A = randn(4, 4);

    // its trace
    auto tr = trace(A);
  }

  // Section 5.13, code block 5.18
  {
    // create two matrices
    mat A = randn(4, 2);
    mat B = randn(4, 2);

    // initialize the result
    mat C = zeros(2, 2);

    // the multiplications
    for (auto coli = 0; coli < 2; ++coli) // columns in A
      for (auto colj = 0; colj < 2; ++colj) // columns in B
        C(coli, colj) = dot(A.col(coli), B.col(colj));
  }

  // Section 5.13, code block 5.20
  {
    // a full matrix
    mat A = randn(4, 4);

    // get the upper-triangle
    mat Al = trimatl(A);

    // sum it with its transpose
    mat S = Al + Al.t();

    // Section 5.13, code block 5.22

    // empty rectangular matrix
    mat D = zeros(4, 8);

    // populate its diagonals
    for (auto d = 0; d < min(size(D)); ++d)
      D(d, d) = d + 1;
  }

  cout << endl;

  // done.
}

void chapter_06()
{
  cout << "CHAPTER: Matrix multiplications (chapter 6)" << endl << endl;

  // Section 6.1, code block 6.2

  // two matrices
  mat M1 = randn(4, 3);
  mat M2 = randn(3, 5);

  // and their product
  mat C = M1 * M2;

  // Section 6.2, code block 6.4

  mat A = randn(2, 2);
  mat B = randn(2, 2);

  // notice that C1 != C2
  mat C1 = A * B;
  mat C2 = B * A;

  // Section 6.8, code block 6.6
  {
    // a pair of matrices
    mat M1 = randn(4, 3);
    mat M2 = randn(4, 3);

    // their Hadamard multiplication
    cout << "M1 % M2 = " << endl << (M1 % M2) << endl;
  }

  // Section 6.9, code block 6.8
  {
    // a small matrix
    mat A = { { 1, 2, 3}, {4, 5, 6 } };

    // vectorized
    cout << "vectorise(A) = " << endl << vectorise(A) << endl;
  }

  // Section 6.9, code block 6.10
  {
    mat A = randn(4, 3);
    mat B = randn(4, 3);

    // the transpose-trace trick for the frobenius dot product
    auto f = trace(A.t() * B);
  }

  // Section 6.10, code block 6.12
  {
    mat A = randn(4, 3);
    cout << "norm(A,\"fro\") = " << norm(A, "fro") << endl;
  }

  // Section 6.15, code block 6.14
  {
    // the matrices
    mat A = randn(2, 4);
    mat B = randn(4, 3);

    // initialize
    mat C1 = zeros(2, 3);

    // loop over (N) columns in A
    for (int i = 0; i < size(A, 1); ++i)
      C1 = C1 + A.col(i) * B.row(i);

    // show equality by subtraction (expect zeros)
    cout << "C1 - A*B = " << endl << (C1 - A * B) << endl;
  }

  // Section 6.15, code block 6.16
  {
    // create the matrices
    mat D = diagmat(regspace(1, 4));
    mat A = randn(4, 4);

    // two kinds of multiplication
    mat C1 = D % A;
    mat C2 = D * A;

    // they're the same
    cout << "diagmat(C1) = " << endl << diagmat(C1) << endl;
    cout << "diagmat(C2) = " << endl << diagmat(C2) << endl;
  }

  // Section 6.15, code block 6.18
  {
    // the matrix
    mat A = diagmat(randu(3, 1));

    // the two symmetric matrices
    mat C1 = (A.t() + A) / 2;
    mat C2 = A.t() * A;

    // their equivalence
    cout << "C1-sqrt(C2) = " << endl << (C1 - sqrt(C2)) << endl;
  }

  // Section 6.15, code block 6.20
  {
    // matrix and vector
    auto m = 5;
    mat A = randn(m, m);
    vec v = randn(m, 1);

    // the two sides of the equation
    auto LHS = norm(A * v);
    auto RHS = norm(A, "fro") * norm(v);

    // their difference
    cout << "RHS-LHS = " << (RHS - LHS) << endl; // should always be positive
  }

  cout << endl;

  // done.
}

void chapter_07()
{
  cout << "CHAPTER: Rank (chapter 7)" << endl << endl;

  // Section 7.3, code block 7.2

  // a matrix
  mat A = randn(3, 6);

  // and its rank
  cout << "rank(A) = " << rank(A) << endl;

  // Section 7.4, code block 7.4

  // scalar
  auto s = randn();

  // matrix
  mat M = randn(3, 5);

  // their ranks
  auto r1 = rank(M);
  auto r2 = rank(s * M);

  // are the same
  cout << "r1 = " << r1 << ", r2 = " << r2 << endl;

  // Section 7.10, code block 7.6

  // inspect the source code for rank
  // edit rank

  // Section 7.15, code block 7.8
  {
    // two random matrices
    mat A = randn(9, 2);
    mat B = randn(2, 16);

    // the rank of their product (assume max possible)
    mat C = A * B;
  }

  // Section 7.15, code block 7.10

  // zeros matrix
  mat Z = zeros(5, 5);

  // tiny noise matrix
  mat N = randn(5, 5) * std::numeric_limits<double>::epsilon() * 1e-307;

  // add them together
  mat ZN = Z + N;

  // check their ranks
  cout << "rank(Z) = " << rank(Z) << endl; // r=0
  cout << "rank(ZN) = " << rank(ZN) << endl; // r=5

  // and the matrix norm
  cout << "rank(ZN, \"fro\") = " << norm(ZN, "fro") << endl;

  cout << endl;

  // done.
}

void chapter_08()
{
  cout << "CHAPTER: Matrix spaces (chapter 8)" << endl << endl;

  // Section 8.7, code block 8.2

  mat A = randn(3, 4);
  cout << "null(A) = " << endl << null(A) << endl;

  // Section 8.15, code block 8.4
  {
    // create reduced-rank matrices
    mat A = randn(4, 3) * randn(3, 4);
    mat B = randn(4, 3) * randn(3, 4);

    // find a vector in A's nullspace
    vec n = null(A);

    // zeros vector
    cout << "B*A*n = " << endl << (B * A * n) << endl;

    // not zeros vector
    cout << "A*B*n = " << endl << (A * B * n) << endl;
  }

  // Section 8.15, code block 8.6
  {
    // create a rank-9 matrix
    mat A = randn(16, 9) * randn(9, 11);

    // "right" null space
    mat rn = null(A);

    // left-null space
    mat ln = null(A.t());

    // rank of the matrix
    auto r = rank(A);

    // check the dimensionalities!
    cout << "size(rn,2) + r = " << (size(rn, 1) + r) << endl;
    cout << "size(ln,2) + r = " << (size(ln, 1) + r) << endl;
  }

  cout << endl;

  // done.
}

void chapter_09()
{
  cout << "CHAPTER: Complex numbers (chapter 9)" << endl << endl;

  // Section 9.2, code block 9.2

  // one way to create a complex number
  cx_double z = { 3,4 };

  // initialize zeros
  cx_mat Z = zeros<cx_mat>(2, 1);

  // can simply replace one element with a complex number
  Z(0) = 3.0 + 4i;

  // Section 9.3, code block 9.4
  {
    // some random real and imaginary parts
    cx_mat r = randi<cx_mat>(1, 3, distr_param(-3, 3));
    cx_mat i = randi<cx_mat>(1, 3, distr_param(-3, 3));

    // combine into a matrix
    cx_mat Z = r + i * 1i;

    // its conjugate
    cout << "conj(Z) = " << endl << conj(Z) << endl;
  }

  // Section 9.5, code block 9.6

  // a complex vector
  cx_vec v = { 0, 1i };

  // Hermitian dot product
  cout << "dot(v,v) = " << dot(v, v) << endl;

  // Section 9.10, code block 9.8

  cx_mat U = { {1. + 1i, 1. - 1i}, {1. - 1i, 1. + 1i} };
  U *= .5;

  // Hermitian
  cout << "U'*U = " << endl << U.ht() * U << endl;

  // not Hermitian
  cout << "transpose(U)*U = " << endl << U.t() * U << endl;

  // Section 9.10, code block 9.10

  // create a complex matrix
  cx_mat A = cx_mat(randn(3, 3), randn(3, 3));

  // new matrices by adding and multiplying
  cx_mat A1 = A + A.ht();
  cx_mat A2 = A * A.ht();

  cout << "ishermitian(A1) = " << A1.is_hermitian() << endl; // issymmetric(A1) is false!
  cout << "ishermitian(A2) = " << A2.is_hermitian() << endl;

  cout << endl;

  // done.
}

void chapter_10()
{
  cout << "CHAPTER: Systems of equations (chapter 10)" << endl << endl;

  // Section 10.3, code block 10.2

  // create a matrix
  mat A = randn(4, 3);

  // take its LU decomposition
  mat L, U, P;
  lu(L, U, P, A);

  // Section 10.5, code block 10.4
  {
    mat A = randn(2, 4);

    // its RREF
    cout << "rref(A) = " << endl << mat_to_rref(A) << endl;
  }

  // Section 10.12, code block 10.6
  {
    // the matrix
    mat A = { { 2, 0, -3}, {3, 1, 4}, { 1, 0, -1} };

    // note: column vector!
    vec x = { 2, 3, 4 };

    // the constants vector
    vec b = A * x;
  }

  // Section 10.12, code block 10.8

  // one example
  cout << "rref(randn(3,6)) = " << endl << mat_to_rref(randn(3, 6)) << endl;

  cout << endl;

  // done.
}

void chapter_11()
{
  cout << "CHAPTER: Determinant (chapter 11)" << endl << endl;

  // Section 11.6, code block 11.2

  mat A = randn(3, 3);
  cout << "det(A) = " << det(A) << endl;

  // Section 11.6, code block 11.4
  {
    // random matrix and vector
    mat A = randi<mat>(4, 4, distr_param(0, 10));
    sword b = randi(distr_param(-10, -1));

    // show equivalence
    cout << "det(b*A) = " << det(b * A) << ", b^4*det(A) = " << (pow(b, 4) * det(A)) << endl;
  }

  // Section 11.6, code block 11.6

  // matrix sizes
  ivec ns = regspace<ivec>(3, 1, 30);

  // iterations
  auto iters = 100;

  // initialize results matrix
  mat dets = zeros(ns.size(), iters);

  // loop over matrix sizes
  for (int ni = 0; ni < ns.size(); ++ni)
  {
    for (int i = 0; i < iters; ++i)
    {
      // step 1
      mat A = randn(ns(ni), ns(ni));

      // step 2
      A.col(0) = A.col(1);

      // step 3
      dets(ni, i) = fabs(det(A));
    }
  }

  auto get_log_mean = [&](s64 row_index) {return log(mean(dets.row(row_index)));};
  auto log_means_view = regspace<ivec>(0, 1, ns.size() - 1) | std::views::transform(get_log_mean);
  auto log_means = std::vector(log_means_view.begin(), log_means_view.end());

  // show in a plot
  auto f = plt::figure(true); //clf
  f->width(f->width() * 2);
  f->height(f->height() * 2);
  plt::plot(ns, log_means, "s-");
  plt::xlabel("Matrix size");
  plt::ylabel("Log determinant");
  plt::show();

  cout << endl;

  // done.
}

void chapter_12()
{
  cout << "CHAPTER: Matrix inverse (chapter 12)" << endl << endl;

  // Section 12.4, code block 12.2

  // a square matrix (full-rank!)
  mat A = randn(3, 3);

  // inverse
  mat Ai = inv(A);

  // should equal identity
  cout << "A*Ai = " << endl << (A * Ai) << endl;

  // Section 12.5, code block 12.4
  {
    // invertible matrix
    mat A = randn(3, 3);

    // RREF with identity
    mat Ar = mat_to_rref(join_rows(A, eye(3, 3))); // RREF

    // extract the inverse part
    Ar = Ar.cols(3, 5);

    // inverse via inv function
    mat Ai = inv(A);

    // check for equality
    cout << "Ar-Ai = " << endl << (Ar - Ai) << endl;
  }

  // Section 12.7, code block 12.6
  {
    // tall matrix
    mat A = randn(5, 3);

    // left inverse
    mat Al = inv(A.t() * A) * A.t();

    // check for I
    cout << "Al*A = " << endl << (Al * A) << endl;
  }

  // Section 12.8, code block 12.8
  {
    // make a reduced-rank matrix
    mat A = randn(3, 3);
    A.row(1) = A.row(0);

    // MP pseudoinverse
    mat Api = pinv(A);

    cout << "Api*A = " << endl << (Api * A) << endl;
  }

  // Section 12.12, code block 12.10
  {
    // create matrix
    auto m = 4;
    mat A = randn(m, m);
    mat M = zeros(m, m);
    mat G = zeros(m, m);

    // compute matrices
    for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < m; ++j)
      {
        // select rows/cols
        uvec rows(m);
        rows.fill(1);
        rows[i] = 0;

        uvec cols(m);
        cols.fill(1);
        cols[j] = 0;

        // compute M
        M(i, j) = det(A.submat(find(rows == 1), find(cols == 1)));

        // compute G
        G(i, j) = pow(-1, i + j);
      }
    }

    // compute C
    mat C = M % G;

    // compute A
    mat Ainv = C.t() / det(A);
    mat AinvI = inv(A);
    cout << "AinvI-Ainv = " << endl << (AinvI - Ainv) << endl; //compare against inv()
  }

  // Section 12.12, code block 12.12
  {
    // square matrix
    mat A = randn(5, 5);
    mat Ai = inv(A);
    mat Api = pinv(A);
    cout << "Ai-Api = " << endl << (Ai - Api) << endl; // test equivalence

    // tall matrix
    mat T = randn(5, 3);
    mat Tl = inv(T.t() * T) * T.t(); // left inv
    mat Tpi = pinv(T); // pinv
    cout << "Tl-Tpi = " << endl << (Tl - Tpi) << endl; // test equivalance
  }

  cout << endl;

  // done.
}

void chapter_13()
{
  cout << "CHAPTER: Projections and orthogonalization (chapter 13)" << endl << endl;

  // Section 13.2, code block 13.2

  // matrix and vector
  mat A = { {1, 2}, {3, 1}, {1, 1} };
  colvec b = { 5.5, -3.5, 1.5 };

  // short-cut for least-squares solver
  cout << "solve(A,b) = " << endl << solve(A, b) << endl;

  // Section 13.6, code block 13.4
  {
    // the matrix
    mat A = randn(4, 3);

    // its QR decomposition
    mat Q, R;
    qr(Q, R, A); // add ,"econ" to get economy decomposition
  }

  // Section 13.11, code block 13.6
  {
    // sizes
    u64 m = 4;
    u64 n = 4;

    // matrix
    mat A = randn(m, n);

    // initialize
    mat Q = zeros(m, n);

    for (u64 i = 0; i < n; ++i) // loop through columns (n)
    {
      Q.col(i) = A.col(i);

      // orthogonalize
      colvec a = A.col(i); // convenience
      for (u64 j = 0; j < i; ++j)
      {
        colvec q = Q.col(j); // convenience
        Q.col(i) = Q.col(i) - as_scalar(a.t() * q / (q.t() * q)) * q;
      }

      // normalize
      Q.col(i) = Q.col(i) / norm(Q.col(i));
    }

    cout << "Q' * Q = " << endl << (Q.t() * Q) << endl;

    // test against "real" Q matrix
    mat Q2, R;
    qr(Q2, R, A);

    // note the possible sign differences.
    cout << "Q - Q2 = " << endl << (Q - Q2) << endl;
    // seemingly non-zero columns will be 0 when adding
    cout << "Q + Q2 = " << endl << (Q + Q2) << endl;
  }

  cout << endl;

  // done.
}

void chapter_14()
{
  cout << "CHAPTER: Least squares (chapter 14)" << endl << endl;

  // Section 14.10, code block 14.2

  // load the data
  mat data;
  data.load("widget_data.txt", csv_ascii);

  // design matrix
  mat X = join_rows(ones(1000, 1), data.cols(0, 1));

  // outcome variable
  colvec y = data.col(2);

  // beta coefficients
  colvec beta = solve(X, y);

  // scaled coefficients (intercept not scaled)
  rowvec betaScaled = beta.t() / stddev(X);

  // Section 14.10, code block 14.4

  auto f = plt::figure(true);
  f->width(f->width() * 3);
  f->height(f->height() * 1.5);

  plt::tiledlayout(1, 2);

  auto ax1 = plt::nexttile();

  plt::axis(ax1, plt::square);
  plt::title(ax1, "Time variable");
  plt::xlabel(ax1, "Time of day");
  plt::ylabel(ax1, "Widgets purchased");

  auto l1 = plt::scatter(
    ax1,
    conv_to<std::vector<double>>::from(X.col(1)),
    conv_to<std::vector<double>>::from(y)
  );
  l1->marker_color("k");
  l1->marker_style(plt::line_spec::marker_style::circle);

  auto ax2 = plt::nexttile();

  plt::axis(ax2, plt::square);
  plt::title(ax2, "Age variable");
  plt::xlabel(ax2, "Age");
  plt::ylabel(ax2, "Widgets purchased");

  auto l2 = plt::scatter(
    ax2,
    conv_to<std::vector<double>>::from(X.col(2)),
    conv_to<std::vector<double>>::from(y)
  );
  l2->marker_color("k");
  l2->marker_style(plt::line_spec::marker_style::circle);

  plt::show();

  // Section 14.10, code block 14.6

  // predicted data values
  mat yHat = X * beta;

  // R-squared
  double r2 = 1 - sum(pow(yHat - y, 2)) / sum(pow(y - mean(y), 2));

  cout << endl;

  // done.
}

void chapter_15()
{
  cout << "CHAPTER: Eigendecomposition (chapter 15)" << endl << endl;

  // Section 15.4, code block 15.2

  // friendly little matrix
  mat A = { { 2, 3}, {3, 2 } };

  // vector of eigenvalues
  cx_vec L;
  eig_gen(L, A);

  // diagonalization
  cx_mat V;
  eig_gen(L, V, A);

  // Section 15.12, code block 15.4
  {
    // create two matrices
    u64 n = 3;
    mat A = randn(n, n);
    mat B = randn(n, n);

    // note the order of inputs
    cx_vec L;
    cx_mat V;
    eig_pair(L, V, A, B);
  }

  // Section 15.16, code block 15.6

  // largest matrix size
  u64 maxN = 100;

  // initialize
  vec avediffs = zeros(maxN, 1);

  for (u64 n = 1; n <= maxN; ++n)
  {
    // create matrices
    mat A = randn(n, n);
    mat B = randn(n, n);

    // GED two ways
    cx_vec l1, l2;
    eig_pair(l1, A, B);
    eig_gen(l2, inv(B) * A);

    // sort the eigenvalues
    l1 = sort(l1);
    l2 = sort(l2);

    // store the differences
    avediffs(n - 1) = mean(abs(l1 - l2));
  }

  auto f = plt::figure(true); // clf
  f->width(f->width() * 3);
  f->height(f->height() * 1.5);
  plt::plot(conv_to<std::vector<double>>::from(avediffs), "s-");
  plt::xlabel("Matrix size");
  plt::ylabel("Δλ");
  plt::show();

  // Section 15.16, code block 15.8
  {
    // create a diagonal matrix
    mat D = diagmat(regspace(1, 5));

    // check out its eigendecomposition
    cx_vec L;
    cx_mat V;
    eig_gen(L, V, D);
  }

  // Section 15.16, code block 15.10
  {
    // create the Hankel matrix
    u64 N = 50;
    vec v = regspace(1, N);
    mat T = toeplitz(v);
    mat H = zeros(N, N);

    for (u64 r = 0; r < N; ++r)
    {
      for (u64 c = 0; c < N; ++c)
      {
        H(r, c) = v((r + c) % N);
      }
    }

    // diagonalize
    cx_vec D;
    cx_mat V;
    eig_gen(D, V, H);
    vec rD = conv_to<vec>::from(D);
    uvec indices = sort_index(rD, "descend");
    V = V.cols(indices);

    auto v_row_to_vector = [&](s64 row_index) {return conv_to<std::vector<double>>::from(V.row(row_index));};
    auto v_row_view = regspace<ivec>(0, 1, N - 1) | std::views::transform(v_row_to_vector);
    auto vvV = std::vector(v_row_view.begin(), v_row_view.end());

    // visualize
    auto f = plt::figure(true);
    f->width(f->width() * 2);
    f->height(f->height() * 2);

    // the matrix
    plt::subplot(2, 2, 0);
    plt::imagesc(mat_to_vector_2d(H));
    plt::axis(plt::square);

    // all eigenvectors
    plt::subplot(2, 2, 1);
    plt::imagesc(vvV);
    plt::axis(plt::square);

    // a few evecs
    plt::subplot(2, 1, 1);
    plt::plot(
      plt::vector_2d{
        conv_to<std::vector<double>>::from(V.col(0)),
        conv_to<std::vector<double>>::from(V.col(1)),
        conv_to<std::vector<double>>::from(V.col(2)),
        conv_to<std::vector<double>>::from(V.col(3)),
        conv_to<std::vector<double>>::from(V.col(4)),
      },
      "o-"
      );

    plt::show();
  }

  cout << endl;

  // done.
}

void chapter_16()
{
  cout << "CHAPTER: Singular value decomposition (chapter 16)" << endl << endl;

  // Section 16.3, code block 16.2

  // a fun matrix
  mat A = { {1, 1, 0},{ 0, 1, 1 } };

  // and its glorious SVD
  mat U;
  vec S;
  mat V;

  svd(U, S, V, A);

  // Section 16.10, code block 16.4
  {
    // matrix
    mat A = randn(5, 5);

    // "my" condition number
    vec s = svd(A);
    double condnum = max(s) / min(s);

    // MATLAB's condition number
    double condnumM = cond(A); // Armadillo's

    // comparison
    cout << "condnum = " << condnum << ", condnumM = " << condnumM << endl;
  }

  // Section 16.14, code block 16.6
  {
    // the matrix
    u64 m = 6;
    u64 n = 3;
    mat A = randn(m, n);

    // the SVD's
    mat Uf;
    vec Sf;
    mat Vf;
    svd(Uf, Sf, Vf, A); // f for full
    mat Ue;
    vec Se;
    mat Ve;
    svd_econ(Ue, Se, Ve, A); // e for econ

    // check out their sizes
    // whos A U* S* V*
  }

  // Section 16.14, code block 16.8
  {
    // matrix
    mat A = randn(4, 5);

    // get V
    cx_vec L2;
    cx_mat V;
    eig_gen(L2, V, A.t() * A);

    // sort by descending eigenvalues
    uvec idx = sort_index(L2, "descend");
    V = V.cols(idx);

    // same for U
    cx_mat U;
    eig_gen(L2, U, A * A.t());
    idx = sort_index(L2, "descend");
    U = U.cols(idx); // sort by descending L

    // create Sigma
    mat S = zeros(size(A));
    L2 = sort(L2, "descend");
    for (u64 i = 0; i < L2.n_rows; ++i)
    {
      S(i, i) = sqrt(L2(i).real());
    }

    // check against MATLAB's SVD function
    mat U2;
    vec S2;
    mat V2;
    svd(U2, S2, V2, A); // Armadillo's

    cout << "S = " << endl << S << endl << endl << "S2 = " << endl << diagmat(S2) << endl;
  }

  // Section 16.14, code block 16.10
  {
    // the matrix and its decomp
    mat A = randn(5, 3);
    mat U;
    vec s;
    mat V;
    svd(U, s, V, A);
    mat S = diagmat(s);

    // loop over layers
    auto f = plt::figure(true); // clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);

    for (u64 i = 0; i < 3; ++i)
    {
      plt::subplot(2, 4, i);

      // create a layer
      mat onelayer = U.col(i) * S(i, i) * V.col(i).t();
      plt::imagesc(mat_to_vector_2d(onelayer));
      plt::title(std::format("Layer {}", i));

      // low-rank approx up to this layer
      plt::subplot(2, 4, i + 4);
      mat lowrank = U.cols(0, i) * S(span(0, i), span(0, i)) * V.cols(0, i).t();
      plt::imagesc(mat_to_vector_2d(lowrank));
      plt::title(std::format("Layers 0:{}", i));
    }

    // the original (full-rank) image
    plt::subplot(2, 4, 7);
    plt::imagesc(mat_to_vector_2d(A));
    plt::title("Original A");

    plt::show();
  }

  // Section 16.14, code block 16.12
  {
    // matrix sizes
    u64 m = 6;
    u64 n = 16;

    // desired condition number
    u64 condnum = 42;

    // create U and V from random numbers, orthogonalized
    mat U, V, r;
    qr(U, r, randn(m, m));
    qr(V, r, randn(n, n));

    // create singular values vector
    vec s = linspace(condnum, 1, std::min(m, n));
    mat S = zeros(m, n);
    for (u64 i = 0; i < std::min(m, n); ++i)
    {
      S(i, i) = s(i);
    }

    // construct matrix
    mat A = U * S * V.t();

    // confirm!
    cout << "cond(A) = " << cond(A) << endl;
  }

  // Section 16.14, code block 16.14
  {
    // get pic and convert to double
    int width, height, channels;
    unsigned char* img = stbi_load("Einstein_tongue.jpg", &width, &height, &channels, 0);
    mat pic = zeros(height, width);
    for (u64 row = 0; row < height; ++row)
    {
      for (u64 column = 0; column < width; ++column)
      {
        pic(row, column) = img[row * width + column];
      }
    }

    mat U;
    vec s;
    mat V;
    svd(U, s, V, pic);
    mat S = diagmat(s);

    // components to keep
    span comps = span(0, 19);

    // low-rank approximation
    mat lowrank = U.cols(comps) * S(comps, comps) * V.cols(comps).t();

    // show the original and low-rank
    auto f = plt::figure(true); // clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);

    plt::subplot(1, 2, 0);
    plt::imagesc(mat_to_vector_2d(pic));
    plt::title("Original");
    plt::colormap(plt::palette::gray());

    plt::subplot(1, 2, 1);
    plt::imagesc(mat_to_vector_2d(lowrank));
    plt::title(std::format("Comps. {}-{}", comps.a, comps.b));
    plt::colormap(plt::palette::gray());

    plt::show();

    // Section 16.14, code block 16.16

    // convert to percent explained
    vec s_pct = 100.0 * s / sum(s);

    f = plt::figure(true); // clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    plt::plot(conv_to<std::vector<double>>::from(s_pct), "s-");
    plt::xlim({ 0, 100 });
    plt::xlabel("Component number");
    plt::ylabel("Pct variance explains");
    plt::show();

    // threshold in percent
    double thresh = 4;

    // comps greater than X%
    uvec s_above_thresh = s_pct > thresh;
    lowrank = U.cols(comps) * S(comps, comps) * V.cols(comps).t();

    // show the original and low-rank
    f = plt::figure(true);
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    plt::subplot(1, 2, 0);
    plt::imagesc(mat_to_vector_2d(pic));
    plt::title("Original");
    plt::colormap(plt::palette::gray());

    plt::subplot(1, 2, 1);
    plt::imagesc(mat_to_vector_2d(lowrank));
    plt::title(std::format(
      "{} of {} comps with > {}%",
      as_scalar(sum(s_above_thresh)),
      s.n_rows,
      thresh
    ));
    plt::colormap(plt::palette::gray());

    plt::show();

    // Section 16.14, code block 16.18

    // initialize
    mat RMS = zeros(s.n_rows, 1);

    for (u64 si = 0; si < (s.n_rows - 1); ++si)
    {
      // compute low-rank approx
      u64 i = si + 1;
      lowrank = U.cols(0, i) * S(span(0, i), span(0, i)) * V.cols(0, i).t();

      // difference image
      mat diffimg = lowrank - pic;

      // RMS
      RMS(si) = sqrt(as_scalar(mean(pow(vectorise(diffimg), 2))));
    }

    f = plt::figure(true); //clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    plt::plot(conv_to<std::vector<double>>::from(RMS), "s-");
    plt::xlabel("Rank approximation");
    plt::ylabel("Error (a.u.)");

    plt::show();
  }

  // Section 16.14, code block 16.20
  {
    // some tall matrix
    mat X = randi<mat>(4, 2, distr_param(1, 6));

    // eq. 29
    mat U;
    vec s;
    mat V;
    svd(U, s, V, X);
    mat S = zeros(4, 2);
    for (u64 i = 0; i < 2; ++i)
    {
      S(i, i) = s(i);
    }

    // eq. 30
    mat longV1 = inv((U * S * V.t()).t() * U * S * V.t()) * (U * S * V.t()).t();

    // eq. 31
    mat longV2 = inv(V * S.t() * U.t() * U * S * V.t()) * (U * S * V.t()).t();

    // eq. 32
    mat longV3 = inv(V * S.t() * S * V.t()) * (U * S * V.t()).t();

    // eq. 33
    mat longV4 = V * pow(S.t() * S, -1) * V.t() * V * S.t() * U.t();

    // eq. 34
    mat MPpinv = pinv(X);

    // compare any of them to the pinv, e.g.,
    cout << "MPpinv - longV3 = " << endl << (MPpinv - longV3) << endl;
  }

  // Section 16.14, code block 16.22

  u64 k = 5;
  u64 n = 13;
  mat a = pinv(ones(n, 1) * k);
  cout << "a - 1/(k*n) = " << endl << (a - 1 / (k * n)) << endl; // check for zeros

  // Section 16.14, code block 16.24
  {
    // parameters
    u64 M = 10; // matrix size
    u64 nIters = 100; // number of iterations
    vec condnums = linspace(10, 1e10, 30);

    // initialize the average eigval differences
    mat avediffs = zeros(nIters, condnums.n_rows);

    // loop over experiment iterations
    for (u64 iteri = 0; iteri < nIters; ++iteri)
    {
      // condition numbers
      for (u64 condi = 0; condi < condnums.n_rows; ++condi)
      {
        // create A
        mat U, V, _;
        qr(U, _, randn(M, M));
        qr(V, _, randn(M, M));
        mat S = diagmat(linspace(condnums(condi), 1, M));
        mat A = U * S * V.t(); // construct matrix

        // create B
        qr(U, _, randn(M, M));
        qr(V, _, randn(M, M));
        S = diagmat(linspace(condnums(condi), 1, M));
        mat B = U * S * V.t(); // construct matrix

        // eigenvalues
        cx_vec l1 = eig_pair(A, B);
        cx_vec l2 = eig_gen(inv(B) * A);

        // and sort
        l1 = sort(l1);
        l2 = sort(l2);

        // store the differences
        avediffs(iteri, condi) = mean(abs(l1 - l2));
      }
    }

    // plot
    auto f = plt::figure(true); //clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    plt::plot(
      conv_to<std::vector<double>>::from(condnums),
      conv_to<std::vector<double>>::from(mean(avediffs)),
      "s-"
    );
    plt::xlabel("Condition number");
    plt::ylabel("Δλ");
    plt::show();
  }

  cout << endl;

  // done.
}

void chapter_17()
{
  cout << "CHAPTER: Quadratic form and definiteness (chapter 17)" << endl << endl;

  // Section 17.1, code block 17.2

  // create matrix and vector
  u64 m = 4;
  mat A = randn(m, m);
  rowvec v = randn(1, m);

  // the quadratic form
  cout << "v*A*v' = " << (v * A * v.t()) << endl;

  // Section 17.9, code block 17.4
  {
    mat A = { {1, 2}, {2, 3} }; // matrix
    vec vi = linspace(-2, 2, 30); // vector elements
    mat quadform = zeros(vi.n_rows, vi.n_rows);

    for (u64 i = 0; i < vi.n_rows; ++i)
    {
      for (u64 j = 0; j < vi.n_rows; ++j)
      {
        vec v = { vi(i), vi(j) }; // vector
        quadform(i, j) = as_scalar(v.t() * A * v / (v.t() * v));
      }
    }

    plt::vector_1d vvi = conv_to<std::vector<double>>::from(vi);
    auto [x, y] = plt::meshgrid(vvi, vvi);
    auto z = mat_to_vector_2d(quadform.t());

    // auto z = plt::transform(
    //   x,
    //   y,
    //   [&](double x, double y) {
    //     vec v = { x, y }; // vector
    //     return as_scalar(v.t() * A * v / (v.t() * v));
    //   }
    // );

    auto f = plt::figure(true); //clf
    f->width(f->width() * 2);
    f->height(f->height() * 2);
    plt::surf(x, y, z);
    plt::xlabel("v_1");
    plt::ylabel("v_2");
    plt::zlabel("ζ");
    plt::show();
  }

  // Section 17.9, code block 17.6
  {
    u64 n = 4;
    u64 nIterations = 500;
    vec defcat = zeros(nIterations, 1);

    auto is_real_cx_vec = [](const cx_vec& v) {
      return std::all_of(
        v.begin(),
        v.end(),
        [](std::complex<double> c) {return c.imag() == 0.;}
      );
      };

    for (u64 iteri = 0; iteri < nIterations; ++iteri)
    {
      // create the matrix
      mat A = randi<mat>(n, n, distr_param(-10, 10));
      cx_vec ev = eig_gen(A); // ev = EigenValues
      while (!is_real_cx_vec(ev))
      {
        A = randi<mat>(n, n, distr_param(-10, 10));
        ev = eig_gen(A); // ev = EigenValues
      }

      auto sign_of_ev = conv_to<ivec>::from(ev).transform(signum<s64>);;

      // "zero" threshold (from rank)
      double t = n * eps(max(svd(A)));

      // test definiteness
      if (std::all_of(sign_of_ev.begin(), sign_of_ev.end(), [](s64 s) {return s == 1;}))
        defcat(iteri) = 1; // pos. def
      else if (std::all_of(sign_of_ev.begin(), sign_of_ev.end(), [](s64 s) {return s > -1;}) && sum(abs(ev) < t) > 0)
        defcat(iteri) = 2; // pos. semidef
      else if (std::all_of(sign_of_ev.begin(), sign_of_ev.end(), [](s64 s) {return s < 1;}) && sum(abs(ev) < t) > 0)
        defcat(iteri) = 4; // neg. semidef
      else if (std::all_of(sign_of_ev.begin(), sign_of_ev.end(), [](s64 s) {return s == -1;}))
        defcat(iteri) = 5; // neg. def
      else
        defcat(iteri) = 3; // indefinite
    }

    // print out summary
    for (u64 i = 1; i <= 5; ++i)
      cout << std::format("cat {}: {}", i, sum(defcat == i)) << endl;
  }

  cout << endl;

  // done.
}

void chapter_18()
{
  cout << "CHAPTER: Covariance matrices (chapter 18)" << endl << endl;

  // Section 18.8, code block 18.2

  // create the "data"
  u64 n = 200;
  mat X = randn(n, 4);

  // mean-center
  rowvec means = mean(X);

  for (u64 i = 0; i < 4; ++i)
  {
    X.col(i) -= as_scalar(means(i));
  }

  // covariance
  mat covM = X.t() * X / (n - 1);

  // stdevs
  mat stdM = inv(diagmat(stddev(X)));

  // correlation matrix
  mat corM = stdM * X.t() * X * stdM / (n - 1);

  // compare covariances
  cout << "covM-cov(X) = " << endl << (covM - cov(X)) << endl;

  // compare corrs
  cout << "corM-corrcoef(X) = " << endl << (corM - cor(X)) << endl;

  cout << endl;

  // done.
}

void chapter_19()
{
  cout << "CHAPTER: Principal components analysis (chapter 19)" << endl << endl;

  // Section 19.7, code block 19.2

  // create data
  u64 N = 1000;
  rowvec h = linspace(150, 190, N).t() + randn(1, N) * 5;
  rowvec w = h * .7 - 50 + randn(1, N) * 10;

  // covariance
  mat X = join_rows(h.t(), w.t());
  rowvec means = mean(X);
  for (u64 i = 0; i < 2; ++i)
  {
    X.col(i) -= as_scalar(means(i));
  }
  mat C = X.t() * X / (h.n_cols - 1);

  // PCA and sort results
  cx_vec D;
  cx_mat V;
  eig_gen(D, V, C);
  vec rD = conv_to<vec>::from(D);
  mat rV = conv_to<mat>::from(V);
  uvec i = sort_index(rD, "descend");
  rV = rV.cols(i);
  mat eigvals = diagmat(rD) * 100 / sum(rD);
  mat scores = X * rV; // not used but useful code

  // plot data with PCs
  auto f = plt::figure(true); //clf
  f->width(f->width() * 2);
  f->height(f->height() * 2);
  plt::hold(plt::on);
  plt::plot(
    conv_to<std::vector<double>>::from(X.col(0)),
    conv_to<std::vector<double>>::from(X.col(1)),
    "ro"
  );
  plt::plot(
    { 0.,as_scalar(rV(0,0) * 45) },
    { 0.,as_scalar(rV(1,0) * 45) },
    "k"
  )->line_width(2);
  plt::plot(
    { 0.,as_scalar(rV(0,1) * 25) },
    { 0.,as_scalar(rV(1,1) * 25) },
    "k"
  )->line_width(2);
  plt::xlabel("Height (cm)");
  plt::ylabel("Weight (kg)");
  plt::axis(plt::square);
  plt::axis({ -50, 50, -50, 50 });
  plt::show();

  // Section 19.7, code block 19.4

  // mean-center
  means = mean(X);
  for (u64 i = 0; i < 2; ++i)
  {
    X.col(i) -= as_scalar(means(i));
  }

  // SVD
  mat U;
  vec s;
  mat Vv;
  svd(U, s, Vv, X); // Vv == V

  // scores
  scores = X * Vv;

  // normalized svals
  s = pow(s, 2) / (X.n_elem - 1);
  s = 100 * s / sum(s); // s == eigvals

  cout << endl;

  // done.
}

int main()
{
  arma_rng::set_seed_random();

  chapter_02();
  chapter_03();
  chapter_05();
  chapter_06();
  chapter_07();
  chapter_08();
  chapter_09();
  chapter_10();
  chapter_11();
  chapter_12();
  chapter_13();
  chapter_14();
  chapter_15();
  chapter_16();
  chapter_17();
  chapter_18();
  chapter_19();

  return 0;
}
