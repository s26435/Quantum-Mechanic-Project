#include <iostream>
#include <cstdlib>
#include <vector>
#include <complex>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <cmath>

#define N 5
#define start_state 0
#define target_state 1

struct ga_args
{
    float jbulk = 1.0f;
    float upper_bound = 100.0f;
    float lower_bound = -100.0f;
    unsigned population_size = 80;
    unsigned generations = 120;
    unsigned elite = 4;
    unsigned turnament_k = 3;
    float mut_p = 0.25f;
    float lambda_time = 0.05f;

    ga_args() = default;
};

template <typename T>
class matrix
{
    std::vector<std::vector<T>> data = std::vector<std::vector<T>>();      // dane w wektorach
    std::pair<unsigned, unsigned> shape = std::pair<unsigned, unsigned>(); // wymiary (pair bo zawsze 2d)

public:
    matrix() = default; // pusta macierz

    // inicjalizacja macierzy wektorem, wektorów
    matrix(std::vector<std::vector<T>> data) : data(data)
    {
        unsigned int rows = data.size();
        unsigned int cols = data.empty() ? 0 : data[0].size();
        for (const std::vector<T> &row : data)
        {
            if (row.size() != cols)
                throw std::runtime_error("niejednorodna macierz");
        }
        shape = {rows, cols};
    }

    // inicjalizacja macierzy wymiarami i wypełnieniem
    matrix(unsigned rows, unsigned cols, T val)
    {
        this->shape = std::pair<unsigned, unsigned>(rows, cols);
        this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols, val));
    }

    // destruktor
    ~matrix() = default;

    // zwraca obiekt na [r][c]
    const T &at(unsigned r, unsigned c) const { return this->data[r][c]; }

    // iość kolumn (shape.second)
    unsigned cols() const { return this->shape.second; }

    // ilosc wierszy (shape.first)
    unsigned rows() const { return this->shape.first; }

    // wymiary jako string do łatrwego debugu
    std::string get_shape_as_str() const
    {
        return "(" + std::to_string(shape.first) + "," + std::to_string(shape.second) + ")";
    }

    float mat1norm()
    {
        float best = 0.f;
        for (unsigned j = 0; j < this->cols(); ++j)
        {
            float col = 0.f;
            for (unsigned i = 0; i < this->rows(); ++i)
                col += std::abs(this->at(i, j));
            best = std::max(best, col);
        }
        return best;
    }

    // macierz + macierz
    matrix operator+(const matrix &o) const
    {
        if ((this->shape.first != o.shape.first) || (this->shape.second != o.shape.second))
        {
            throw std::runtime_error("shape missmatch");
        }

        std::vector<std::vector<T>> r = std::vector<std::vector<T>>(static_cast<size_t>(this->shape.first));

        for (unsigned int i = 0; i < this->shape.first; i++)
        {
            std::vector<T> res = std::vector<T>(static_cast<size_t>(this->shape.second));
            for (unsigned int j = 0; j < this->shape.second; j++)
            {
                res[j] = this->data[i][j] + o.data[i][j];
            }
            r[i] = res;
        }

        return matrix<T>(r);
    }

    // macierz - macierz
    matrix operator-(const matrix &o) const
    {
        if ((this->shape.first != o.shape.first) || (this->shape.second != o.shape.second))
        {
            throw std::runtime_error("shape missmatch");
        }
        std::vector<std::vector<T>> r = std::vector<std::vector<T>>(static_cast<size_t>(this->shape.first));

        for (unsigned int i = 0; i < this->shape.first; i++)
        {
            std::vector<T> res = std::vector<T>(static_cast<size_t>(this->shape.second));
            for (unsigned int j = 0; j < this->shape.second; j++)
            {
                res[j] = this->data[i][j] - o.data[i][j];
            }
            r[i] = res;
        }

        return matrix<T>(r);
    }

    // macierz * skalar
    template <typename Ti>
    auto operator*(const Ti &scalar) const -> matrix<std::common_type_t<T, Ti>>
    {
        using R = std::common_type_t<T, Ti>;
        std::vector<std::vector<R>> out = std::vector<std::vector<R>>(this->data.size());

        for (size_t i = 0; i < data.size(); i++)
        {
            out[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); j++)
            {
                out[i][j] = static_cast<R>(data[i][j]) * static_cast<R>(scalar);
            }
        }
        return matrix<R>(out);
    }

    // macierz + (macierz jedynek * skalar)
    template <typename Ti>
    auto operator+(const Ti &scalar) const -> matrix<std::common_type_t<T, Ti>>
    {
        using R = std::common_type_t<T, Ti>;
        std::vector<std::vector<R>> out = std::vector<std::vector<R>>(this->data.size());

        for (size_t i = 0; i < data.size(); i++)
        {
            out[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); j++)
            {
                out[i][j] = static_cast<R>(data[i][j]) + static_cast<R>(scalar);
            }
        }
        return matrix<R>(out);
    }

    // macierz - (macierz jedynek * skalar)
    template <typename Ti>
    auto operator-(const Ti &scalar) const -> matrix<std::common_type_t<T, Ti>>
    {
        using R = std::common_type_t<T, Ti>;
        std::vector<std::vector<R>> out = std::vector<std::vector<R>>(this->data.size());

        for (size_t i = 0; i < data.size(); i++)
        {
            out[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); j++)
            {
                out[i][j] = static_cast<R>(data[i][j]) - static_cast<R>(scalar);
            }
        }
        return matrix<R>(out);
    }

    // macierz / skalar
    template <typename Ti>
    auto operator/(const Ti &scalar) const -> matrix<std::common_type_t<T, Ti>>
    {
        using R = std::common_type_t<T, Ti>;
        std::vector<std::vector<R>> out = std::vector<std::vector<R>>(this->data.size());

        for (size_t i = 0; i < data.size(); i++)
        {
            out[i].resize(data[i].size());
            for (size_t j = 0; j < data[i].size(); j++)
            {
                out[i][j] = static_cast<R>(static_cast<R>(data[i][j]) / static_cast<R>(scalar));
            }
        }
        return matrix<R>(out);
    }

    // mnożenie macierzowe
    matrix operator*(const matrix &o) const
    {
        if (this->cols() != o.rows())
        {
            std::cerr << "shapes: " << this->get_shape_as_str() << " and " << o.get_shape_as_str() << std::endl;
            throw std::runtime_error("shape missmatch during matmul");
        }

        std::vector<std::vector<T>> out = std::vector<std::vector<T>>(this->rows(), std::vector<T>(o.cols(), T{}));
        for (unsigned i = 0; i < this->rows(); i++)
        {
            for (unsigned k = 0; k < this->cols(); ++k)
            {
                const T aik = this->at(i, k);
                for (unsigned j = 0; j < o.cols(); j++)
                {
                    out[i][j] += aik * o.at(k, j);
                }
            }
        }
        return matrix(out);
    }
};

// wypisywanie macierzy
template <typename T>
std::ostream &operator<<(std::ostream &os, const matrix<T> &m)
{
    os << "[";
    for (unsigned i = 0; i < m.rows(); i++)
    {
        os << "[";
        for (unsigned j = 0; j < m.cols(); j++)
        {
            os << m.at(i, j);
            if (j + 1 < m.cols())
                os << ", ";
        }
        os << "]";
    }
    os << "]";
    return os;
}

template <typename T>
matrix<std::complex<T>> eye(unsigned k)
{
    std::vector<std::vector<std::complex<T>>> d(k, std::vector<std::complex<T>>(k, std::complex<T>{0, 0}));
    for (unsigned i = 0; i < k; ++i)
        d[i][i] = std::complex<T>{1, 0};
    return matrix<std::complex<T>>(d);
}

matrix<std::complex<float>> expm(matrix<std::complex<float>> &A, const int max_terms = 50)
{
    const unsigned k = A.rows();
    if (k != A.cols())
        throw std::runtime_error("matrix not squared");

    float n1 = A.mat1norm();
    unsigned s = 0;

    while (n1 > 0.5f)
    {
        n1 *= 0.5f;
        s++;
    }

    float inv = 1.0f / std::ldexp(1.0f, s);
    A = A * inv;

    matrix<std::complex<float>> E = eye<float>(k);
    matrix<std::complex<float>> term = eye<float>(k);

    for (int i = 1; i <= max_terms; i++)
    {
        term = term * A;
        float invn = 1.0f / static_cast<float>(i);
        E = E + (term * invn);
        if (term.mat1norm() * invn < 1e-6f)
            break;
    }

    for (unsigned i = 0; i < s; i++)
        E = E * E;

    return E;
}

// zwraca rzeczywistą częśc z macierzy liczb zespolonych
template <typename T>
matrix<T> real(const matrix<std::complex<T>> &o)
{
    std::vector<std::vector<T>> out = std::vector<std::vector<T>>(o.rows(), std::vector<T>(o.cols(), T{}));
    for (unsigned i = 0; i < o.rows(); i++)
    {
        for (unsigned j = 0; j < o.cols(); j++)
        {
            out[i][j] = std::real(o.at(i, j));
        }
    }
    return matrix<T>(out);
}

// zwraca urojoną częśc z macierzy liczb zespolonych
template <typename T>
matrix<T> imag(const matrix<std::complex<T>> &o)
{
    std::vector<std::vector<T>> out = std::vector<std::vector<T>>(o.rows(), std::vector<T>(o.cols(), T{}));
    for (unsigned i = 0; i < o.rows(); i++)
    {
        for (unsigned j = 0; j < o.cols(); j++)
        {
            out[i][j] = std::imag(o.at(i, j));
        }
    }
    return matrix<T>(out);
}

// kroneker produkt 2 macierzy
template <typename TA, typename TB>
auto kron(const matrix<TA> &A, const matrix<TB> &B) -> matrix<std::common_type_t<TA, TB>>
{
    using R = std::common_type_t<TA, TB>;

    const unsigned Ar = A.rows(), Ac = A.cols();
    const unsigned Br = B.rows(), Bc = B.cols();

    std::vector<std::vector<R>> out(Ar * Br, std::vector<R>(Ac * Bc, R{}));

    for (unsigned i = 0; i < Ar; ++i)
    {
        for (unsigned j = 0; j < Ac; ++j)
        {
            const R aij = static_cast<R>(A.at(i, j));
            for (unsigned k = 0; k < Br; ++k)
            {
                for (unsigned l = 0; l < Bc; ++l)
                {
                    out[i * Br + k][j * Bc + l] = aij * static_cast<R>(B.at(k, l));
                }
            }
        }
    }

    return matrix<R>(out);
}

std::complex<float> vdot(const matrix<std::complex<float>> &v, const matrix<std::complex<float>> &w)
{
    if (v.cols() != w.cols() || v.rows() != w.rows())
        throw std::runtime_error("size mismatch");

    std::complex<float> s = std::complex<float>(0.0f, 0.0f);

    for (unsigned i = 0; i < w.rows(); i++)
    {
        for (unsigned j = 0; j < w.cols(); j++)
        {
            s = s + std::conj(v.at(i, j)) * w.at(i, j);
        }
    }

    return s;
}

static std::complex<float> vdot(const std::vector<std::complex<float>> &a, const std::vector<std::complex<float>> &b)
{
    if (a.size() != b.size())
        throw std::runtime_error("vdot: size mismatch");
    std::complex<float> s{0.f, 0.f};
    for (size_t i = 0; i < a.size(); ++i)
        s += std::conj(a[i]) * b[i];
    return s;
}

static float norm2(const std::vector<std::complex<float>> &v)
{
    float s = 0.f;
    for (auto &z : v)
        s += std::norm(z); // |z|^2
    return std::sqrt(s);
}

static void axpy(std::vector<std::complex<float>> &y, std::complex<float> a, const std::vector<std::complex<float>> &x)
{
    if (y.size() != x.size())
        throw std::runtime_error("axpy: size mismatch");
    for (size_t i = 0; i < y.size(); ++i)
        y[i] += a * x[i];
}

float vec_norm(const std::vector<std::complex<float>> &v)
{
    float s = 0.f;
    for (auto &z : v)
        s += std::norm(z);
    return std::sqrt(s);
}

std::vector<std::complex<float>> matvec_dense(const matrix<std::complex<float>> &A, const std::vector<std::complex<float>> &x)
{
    if (A.cols() != x.size())
        throw std::runtime_error("matvec_dense: shape mismatch");
    std::vector<std::complex<float>> y(A.rows(), std::complex<float>{0, 0});
    for (unsigned i = 0; i < A.rows(); ++i)
    {
        std::complex<float> s{0, 0};
        for (unsigned j = 0; j < A.cols(); ++j)
            s += A.at(i, j) * x[j];
        y[i] = s;
    }
    return y;
}

std::vector<std::complex<float>> expm_multiply(const matrix<std::complex<float>> &A, const std::vector<std::complex<float>> &v, std::complex<float> alpha, int m = 30, float tol = 1e-6f)
{
    const size_t n = v.size();
    if (A.rows() != A.cols() || A.rows() != n)
        throw std::runtime_error("expm_multiply_krylov: A must be nxn and match v");

    float beta = vec_norm(v);
    if (beta == 0.f)
        return std::vector<std::complex<float>>(n, std::complex<float>{0.0f, 0.0f});

    std::vector<std::vector<std::complex<float>>> V;
    V.reserve(m + 1);

    std::vector<std::complex<float>> v0 = v;
    for (auto &z : v0)
        z /= beta;
    V.push_back(std::move(v0));

    std::vector<std::vector<std::complex<float>>> H(m + 1, std::vector<std::complex<float>>(m + 1, std::complex<float>{0, 0}));

    int k = 0;
    for (; k < m; ++k)
    {
        std::vector<std::complex<float>> w = matvec_dense(A, V[k]);

        for (int j = 0; j <= k; ++j)
        {
            std::complex<float> hj = vdot(V[j], w);
            H[j][k] = hj;
            axpy(w, -hj, V[j]);
        }

        float h_next = vec_norm(w);
        H[k + 1][k] = std::complex<float>{h_next, 0.f};

        if (h_next < tol)
            break;

        for (auto &z : w)
            z /= h_next;
        V.push_back(std::move(w));
    }

    const int kk = std::min(k + 1, m);
    // Hk (kk x kk) jako Twoja matrix
    std::vector<std::vector<std::complex<float>>> hkd(kk, std::vector<std::complex<float>>(kk, std::complex<float>{0.0f, 0.0f}));
    for (int i = 0; i < kk; ++i)
        for (int j = 0; j < kk; ++j)
            hkd[i][j] = H[i][j];

    matrix<std::complex<float>> Hk(hkd);
    Hk = Hk * alpha;

    matrix<std::complex<float>> E = expm(Hk);

    std::vector<std::complex<float>> coeff(kk, std::complex<float>{0.0f, 0.0f});
    for (int i = 0; i < kk; ++i)
        coeff[i] = E.at(i, 0);

    std::vector<std::complex<float>> y(n, std::complex<float>{0, 0});
    for (int j = 0; j < kk; ++j)
    {
        axpy(y, std::complex<float>{beta, 0.f} * coeff[j], V[j]);
    }
    return y;
}

// zwraca macierz 2x2 podanych wartości
matrix<std::complex<float>> get_open22(std::complex<float> a, std::complex<float> b, std::complex<float> c, std::complex<float> d)
{
    return std::vector<std::vector<std::complex<float>>>({{a, b}, {c, d}});
}

// zwraca macierze paulisa cwelixa
std::vector<matrix<std::complex<float>>> paulis()
{
    using cd = std::complex<float>;
    return std::vector<matrix<std::complex<float>>>({
        get_open22(cd(0.0f, 0.0f), cd(1.0f, 0.0f), cd(1.0f, 0.0f), cd(0.0f, 0.0f)),  // sigma x
        get_open22(cd(0.0f, 0.0f), cd(0.0f, -1.0f), cd(0.0f, 1.0f), cd(0.0f, 0.0f)), // sigma y
        get_open22(cd(1.0f, 0.0f), cd(0.0f, 0.0f), cd(0.0f, 0.0f), cd(-1.0f, 0.0f)), // sigma z
        get_open22(cd(1.0f, 0.0f), cd(0.0f, 0.0f), cd(0.0f, 0.0f), cd(1.0f, 0.0f))   // identyczność
    });
}

matrix<std::complex<float>> op_on_site(matrix<std::complex<float>> &op, unsigned site, unsigned n, matrix<std::complex<float>> &I2)
{
    matrix<std::complex<float>> out;
    matrix<std::complex<float>> a;
    bool is_kroned = false;
    for (unsigned i = 0; i < n; i++)
    {
        a = (i == site) ? op : I2;
        if (!is_kroned)
        {
            out = a;
            is_kroned = true;
        }
        else
        {
            out = kron(out, a);
        }
    }

    return out;
}

auto precompiute_bonds(unsigned n)
{
    std::vector<matrix<std::complex<float>>> sigmas = paulis();
    std::vector<matrix<std::complex<float>>> B = std::vector<matrix<std::complex<float>>>(n - 1);

    for (unsigned i = 0; i < n - 1; i++)
    {
        matrix<std::complex<float>> sx_i = op_on_site(sigmas[0], i, n, sigmas[3]);
        matrix<std::complex<float>> sx_j = op_on_site(sigmas[0], i + 1, n, sigmas[3]);

        matrix<std::complex<float>> sy_i = op_on_site(sigmas[1], i, n, sigmas[3]);
        matrix<std::complex<float>> sy_j = op_on_site(sigmas[1], i + 1, n, sigmas[3]);

        matrix<std::complex<float>> sz_i = op_on_site(sigmas[2], i, n, sigmas[3]);
        matrix<std::complex<float>> sz_j = op_on_site(sigmas[2], i + 1, n, sigmas[3]);

        B[i] = sx_i * sx_j + sy_i * sy_j + sz_i * sz_j;
    }

    return B;
}

matrix<std::complex<float>> build_H(std::vector<matrix<std::complex<float>>> &B, float J1, float JN, float Jbulk)
{
    using cd = std::complex<float>;
    const size_t n_minus_1 = B.size();

    matrix<cd> H(B[0].rows(), B[0].cols(), cd(0.f, 0.f));
    H = H + (B[0] * cd(J1, 0.f));

    for (size_t i = 1; i + 1 < n_minus_1; ++i)
        H = H + (B[i] * cd(Jbulk, 0.f));

    H = H + (B[n_minus_1 - 1] * cd(JN, 0.f));
    return H;
}

float prob_target(const std::vector<std::complex<float>>& psi_t, const std::vector<std::complex<float>>& psi_target)
{
    const std::complex<float> overlap = vdot(psi_target, psi_t); // <target|psi>
    const float overlap2 = std::norm(overlap);                   // |.|^2

    float n2 = 0.f;
    for (const auto &z : psi_t) n2 += std::norm(z);              // ||psi||^2

    return (n2 > 0.f) ? (overlap2 / n2) : 0.f;  
}

float fitness(float j1, float j2, std::vector<matrix<std::complex<float>>> &B, std::vector<std::complex<float>> &psi0, std::vector<std::complex<float>> psi_targ, std::vector<float> t_grid, float jbulk = 1.0f, float lambda_time = 0.05f)
{
    matrix<std::complex<float>> H = build_H(B, j1, j2, jbulk);
    float best = -1e9;

    for (size_t i = 0; i < t_grid.size(); ++i)
    {
        float t = t_grid[i];
        auto psi_t = expm_multiply(H, psi0, std::complex<float>(0.0f, -t)); // exp(-i H t)
        float score = prob_target(psi_t, psi_targ) - lambda_time * t;
        if (score > best)
            best = score;
    }
    return best;
}

std::pair<float, float> rnd_float_pair(float upper, float lower)
{
    return std::pair<float, float>(lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (upper - lower))), lower + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (upper - lower))));
}

std::vector<std::pair<float, float>> gen_random_pop(float upper, float lower, unsigned size)
{
    std::vector<std::pair<float, float>> pop = std::vector<std::pair<float, float>>(size, std::pair<float, float>(0.0f, 0.0f));
    for (size_t i = 0; i < size; i++)
    {
        pop[i] = rnd_float_pair(upper, lower);
    }
    return pop;
}

void sort_by_scores_and_take_elite(std::vector<std::pair<float, float>> &pop, std::vector<float> &scores, std::vector<std::pair<float, float>> &new_pop, size_t elite)
{
    if (pop.size() != scores.size())
        throw std::runtime_error("pop i scores maja rozne rozmiary");

    const size_t n = pop.size();
    elite = std::min(elite, n);

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](size_t a, size_t b)
              { return scores[a] > scores[b]; });
    std::vector<std::pair<float, float>> pop_sorted;
    std::vector<float> scores_sorted;
    pop_sorted.reserve(n);
    scores_sorted.reserve(n);

    for (size_t idx : order)
    {
        pop_sorted.push_back(pop[idx]);
        scores_sorted.push_back(scores[idx]);
    }

    pop = std::move(pop_sorted);
    scores = std::move(scores_sorted);

    new_pop.clear();
    new_pop.reserve(n);
    for (size_t i = 0; i < elite; ++i)
    {
        new_pop.push_back(pop[i]);
    }

    for (size_t i = 0; i < elite; ++i)
        new_pop.push_back(pop[i]);

    for (size_t i = elite; i < n - elite; ++i)
        new_pop.push_back(std::pair<float, float>(0.0f, 0.0f));
}

size_t turnament(const std::vector<std::pair<float, float>> &pop, const std::vector<float> &scores, size_t k)
{
    std::vector<size_t> idxs = std::vector<size_t>(k, 0);
    for (unsigned i = 0; i < k; i++)
    {
        idxs[i] = static_cast<size_t>(rand() % (pop.size()));
    }

    std::vector<float> best_scores = std::vector<float>(k, -1.0f);
    for (size_t i = 0; i < k; i++)
    {
        best_scores[i] = scores[idxs[i]];
    }

    return static_cast<size_t>(std::distance(best_scores.begin(), std::max_element(best_scores.begin(), best_scores.end())));
}

std::pair<float, float> cross_and_mutate(const std::pair<float, float> &parent_1, const std::pair<float, float> &parent_2, const ga_args &args)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-args.mut_p, args.mut_p);
    float mult = 2.0f * (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) - 1.0f;;

    std::pair<float, float> child = std::pair<float, float>(mult * parent_1.first + (1 - mult) * parent_2.first + dis(gen), mult * parent_1.second + (1 - mult) * parent_2.second + dis(gen));

    if (child.first < args.lower_bound)
    {
        child.first = args.lower_bound;
    }
    else if (child.first > args.upper_bound)
    {
        child.first = args.upper_bound;
    }

    if (child.second < args.lower_bound)
    {
        child.second = args.lower_bound;
    }
    else if (child.second > args.upper_bound)
    {
        child.second = args.upper_bound;
    }

    return child;
}

std::pair<float, float> GA(std::vector<matrix<std::complex<float>>> &B, std::vector<std::complex<float>> psi0, std::vector<std::complex<float>> psi_targ, std::vector<float> t_grid, ga_args args = ga_args())
{
    std::vector<std::pair<float, float>> population = gen_random_pop(args.upper_bound, args.lower_bound, args.population_size);
    std::vector<std::pair<float, float>> new_population = std::vector<std::pair<float, float>>(args.population_size, std::pair<float, float>(0.0f, 0.0f));
    std::vector<float> scores = std::vector<float>(args.population_size, -1e9);

    for (size_t i = 0; i < args.population_size; i++)
    {
        scores[i] = fitness(population[i].first, population[i].second, B, psi0, psi_targ, t_grid, args.jbulk, args.lambda_time);
    }

    for (unsigned generation = 0; generation < args.generations; generation++)
    {
        sort_by_scores_and_take_elite(population, scores, new_population, args.elite);
        for (size_t i = args.elite - 1; i < args.population_size; i++)
        {
            new_population[i] = cross_and_mutate(population[turnament(population, scores, args.turnament_k)], population[turnament(population, scores, args.turnament_k)], args);
        }

        population = new_population;
        for (size_t i = 0; i < args.population_size; i++)
        {
            scores[i] = fitness(population[i].first, population[i].second, B, psi0, psi_targ, t_grid, args.jbulk, args.lambda_time);
        }

        size_t best_i = std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()));
        std::cout << "Generation: " << generation << ": best_i = " << best_i << ", best_val = (J1 = " << population[best_i].first << ", J2 = " << population[best_i].second << "), best_score = " << scores[best_i] << std::endl;
    }

    return population[std::distance(scores.begin(), std::max_element(scores.begin(), scores.end()))];
}

int idxone(int site, int n)
{
    return 1 << (n - site - 1);
}

std::vector<float> get_times(float start, float end, int num_points)
{
    if (num_points < 2)
        return {start};
    std::vector<float> t(num_points);
    float dt = (end - start) / float(num_points - 1);
    for (int i = 0; i < num_points; ++i)
        t[i] = start + dt * float(i);
    return t;
}

bool is_finite_complex(const std::complex<float>& z) {
    return std::isfinite(z.real()) && std::isfinite(z.imag());
}

bool is_finite_vec(const std::vector<std::complex<float>>& v) {
    for (auto &z : v) if (!is_finite_complex(z)) return false;
    return true;
}

void report_individual(const std::pair<float,float>& ind, std::vector<matrix<std::complex<float>>> &B, const std::vector<std::complex<float>> &psi0, const std::vector<std::complex<float>> &psi_targ, const std::vector<float> &t_grid, float Jbulk = 1.0f, float lambda_time = 0.05f, int krylov_m = 30, float krylov_tol = 1e-6f){
    const float J1 = ind.first;
    const float JN = ind.second;

    auto H = build_H(B, J1, JN, Jbulk);

    float bestP = -1.0f, bestP_t = std::numeric_limits<float>::quiet_NaN();
    float bestScore = -1e30f, bestScore_t = std::numeric_limits<float>::quiet_NaN();

    float sumP = 0.0f, sumP2 = 0.0f;
    float sumT = 0.0f;
    float sumTP = 0.0f;
    float sumP_for_meanT = 0.0f;

    float auc = 0.0f;
    bool any_nonfinite = false;

    std::vector<float> Ps;
    Ps.reserve(t_grid.size());

    float t_first_over_01 = std::numeric_limits<float>::quiet_NaN();
    float t_first_over_05 = std::numeric_limits<float>::quiet_NaN();

    for (size_t i = 0; i < t_grid.size(); ++i) {
        float t = t_grid[i];

        auto psi_t = expm_multiply(H, psi0, std::complex<float>(0.0f, -t), krylov_m, krylov_tol);
        if (!is_finite_vec(psi_t)) {
            any_nonfinite = true;
            Ps.push_back(std::numeric_limits<float>::quiet_NaN());
            continue;
        }

        float P = prob_target(psi_t, psi_targ);
        Ps.push_back(P);

        if (std::isfinite(P)) {
            sumP  += P;
            sumP2 += P * P;
            sumT  += t;

            sumTP += t * P;
            sumP_for_meanT += P;

            float score = P - lambda_time * t;

            if (P > bestP) { bestP = P; bestP_t = t; }
            if (score > bestScore) { bestScore = score; bestScore_t = t; }

            if (!std::isfinite(t_first_over_01) && P >= 0.1f) t_first_over_01 = t;
            if (!std::isfinite(t_first_over_05) && P >= 0.5f) t_first_over_05 = t;
        }

        if (i > 0 && std::isfinite(Ps[i-1]) && std::isfinite(Ps[i])) {
            float dt = t_grid[i] - t_grid[i-1];
            auc += 0.5f * dt * (Ps[i-1] + Ps[i]);
        }
    }

    const float n = static_cast<float>(t_grid.size());
    float meanP = sumP / n;
    float varP  = (sumP2 / n) - meanP * meanP;
    float stdP  = (varP > 0.f) ? std::sqrt(varP) : 0.f;

    float meanT = sumT / n;
    float meanT_weighted_by_P = (sumP_for_meanT > 0.f) ? (sumTP / sumP_for_meanT) : std::numeric_limits<float>::quiet_NaN();

    float t_first_over_90max = std::numeric_limits<float>::quiet_NaN();
    if (bestP > 0.f && std::isfinite(bestP)) {
        float thr = 0.9f * bestP;
        for (size_t i = 0; i < Ps.size(); ++i) {
            if (std::isfinite(Ps[i]) && Ps[i] >= thr) { t_first_over_90max = t_grid[i]; break; }
        }
    }

    float norm_at_bestP = std::numeric_limits<float>::quiet_NaN();
    if (std::isfinite(bestP_t)) {
        auto psi_best = expm_multiply(H, psi0, std::complex<float>(0.0f, -bestP_t), krylov_m, krylov_tol);
        if (is_finite_vec(psi_best)) norm_at_bestP = vec_norm(psi_best);
    }

    std::cout << "\n=== Individual report ===\n";
    std::cout << "J1=" << J1 << "  JN=" << JN << "  (Jbulk=" << Jbulk << ")\n";
    std::cout << "lambda_time=" << lambda_time << "  krylov(m=" << krylov_m << ", tol=" << krylov_tol << ")\n";

    if (any_nonfinite) {
        std::cout << "WARNING: non-finite values encountered for some t (NaN/inf in psi).\n";
    }

    std::cout << "Best P(t):        " << bestP << " at t=" << bestP_t << "\n";
    std::cout << "Best score:       " << bestScore << " at t=" << bestScore_t
              << "   (score = P - lambda*t)\n";

    std::cout << "Mean P:           " << meanP << "   std(P): " << stdP << "\n";
    std::cout << "AUC(P over t):    " << auc << "   (większe = częściej trafia w czasie)\n";

    std::cout << "Mean t (grid):    " << meanT << "\n";
    std::cout << "Mean t weighted P:" << meanT_weighted_by_P << "   (\"średni czas trafienia\")\n";

    std::cout << "First t with P>=0.1: " << t_first_over_01 << "\n";
    std::cout << "First t with P>=0.5: " << t_first_over_05 << "\n";
    std::cout << "First t with P>=0.9*maxP: " << t_first_over_90max << "\n";

    std::cout << "Norm(psi) at t*=argmax P: " << norm_at_bestP << "   (powinno ~1)\n";
    std::cout << "=========================\n";
}

int main(int argv, char **argc)
{
    std::vector<matrix<std::complex<float>>> B = precompiute_bonds(N);

    unsigned dim = static_cast<long>(pow(2, N));

    std::vector<std::complex<float>> psi0 = std::vector<std::complex<float>>(dim, std::complex(0.0f, 0.0f));
    psi0[idxone(start_state, N)] = std::complex<float>(1.0f, 0.0f);

    std::vector<std::complex<float>> target_psi = std::vector<std::complex<float>>(dim, std::complex(0.0f, 0.0f));
    target_psi[idxone(target_state, N)] = std::complex<float>(1.0f, 0.0f);

    auto t_grid = get_times(0.0f, 6.0f, 120);
    auto best = GA(B, psi0, target_psi, t_grid);
    report_individual(best, B, psi0, target_psi, t_grid);


    return EXIT_SUCCESS;
}