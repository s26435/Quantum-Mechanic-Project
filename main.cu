#include <cuda_runtime.h>
#include <cstdlib>
#include <complex>
#include <vector>
#include <iostream>
#include <type_traits>


#define CHECK_CUDA(call)                                                                                                \
    {                                                                                                                   \
        cudaError_t err = call;                                                                                         \
        if (err != cudaSuccess)                                                                                         \
        {                                                                                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                         \
        }                                                                                                               \
    }

struct complx
{
    float real;
    float imag;

    __host__ __device__ complx(float a = 0.0f, float b = 0.0f) : real(a), imag(b) {}

    __host__ __device__ complx operator+(const complx &o) const
    {
        return complx(this->real + o.real, this->imag + o.imag);
    }

    __host__ __device__ complx operator-(const complx &o) const
    {
        return complx(this->real - o.real, this->imag - o.imag);
    }

    __host__ __device__ complx operator*(const complx &o) const
    {
        return complx(this->real * o.real - this->imag * o.imag, this->real * o.imag + this->imag * o.real);
    }

    __host__ __device__ complx conj() const { return {real, -imag}; }

    __host__ __device__ float abs2() const { return real * real + imag * imag; }

    __host__ __device__ float abs() const { return sqrtf(abs2()); }

    __host__ __device__ complx operator/(const complx &o) const
    {
        float denom = o.real * o.real + o.imag * o.imag;
        return complx((real * o.real + imag * o.imag) / denom,
                      (imag * o.real - real * o.imag) / denom);
    }

    __host__ __device__
    complx& operator+=(const complx& o) {
        real += o.real;
        imag += o.imag;
        return *this;
    }

    __host__ __device__
    complx& operator-=(const complx& o) {
        real -= o.real;
        imag -= o.imag;
        return *this;
    }

    __host__ __device__
    complx& operator*=(const complx& o) {
        float r = real * o.real - imag * o.imag;
        float i = real * o.imag + imag * o.real;
        real = r;
        imag = i;
        return *this;
    }
};

inline std::ostream& operator<<(std::ostream& os, const complx& z) {
    os << z.real;
    if (z.imag >= 0) os << " + " << z.imag << "i";
    else             os << " - " << -z.imag << "i";
    return os;
}

template <class T>
class tensor
{
public:
    int n_dim = 0;
    int *shape = nullptr;
    size_t curr_size;

private:
    T *data = nullptr;
    bool owns_memory = false;

public:
    __host__ __device__ int numel() const
    {
        int a = 1;
        for (short i = 0; i < this->n_dim; i++)
            a *= shape[i];
        return a;
    }

    __host__ __device__ T *raw()
    {
        return data;
    }

    __host__ __device__ const T *raw() const
    {
        return data;
    }

    __host__ tensor() = default;

    __host__ tensor(int _n_dim, int *_shape, T *_data) : n_dim(_n_dim), shape(_shape), data(_data)
    {
        this->curr_size = this->numel() * sizeof(T);
    }

    __host__ static tensor owning(int _n_dim, const int *_shape)
    {
        tensor out;
        out.n_dim = _n_dim;
        out.owns_memory = true;

        out.shape = (int *)std::malloc(_n_dim * sizeof(int));
        if (!out.shape)
            throw std::bad_alloc();

        for (int d = 0; d < _n_dim; ++d)
            out.shape[d] = _shape[d];

        size_t n = out.numel();
        out.data = (T *)std::malloc(n * sizeof(T));
        if (!out.data)
            throw std::bad_alloc();

        out.curr_size = n * sizeof(T);
        return out;
    }

    __host__ bool same_shape(const tensor &o) const
    {
        if (this->n_dim != o.n_dim)
            return false;
        for (int d = 0; d < this->n_dim; d++)
        {
            if (this->shape[d] != o.shape[d])
                return false;
        }
        return true;
    }

    __host__ tensor operator+(const tensor &o) const
    {
        if (!this->same_shape(o))
            throw std::runtime_error("tensor shape missmatch");

        tensor out;
        out.n_dim = this->n_dim;
        out.shape = (int *)malloc(n_dim * sizeof(int));

        for (int d = 0; d < this->n_dim; d++)
            out.shape[d] = this->shape[d];

        int n = this->numel();
        out.data = (T *)malloc(n * sizeof(T));
        out.owns_memory = true;
        for (int i = 0; i < n; i++)
            out.data[i] = this->data[i] + o.data[i];

        return out;
    }

    __host__ tensor operator-(const tensor &o) const
    {
        if (!this->same_shape(o))
            throw std::runtime_error("tensor shape missmatch");

        tensor out;
        out.n_dim = this->n_dim;
        out.shape = (int *)malloc(n_dim * sizeof(int));

        for (int d = 0; d < this->n_dim; d++)
            out.shape[d] = this->shape[d];

        int n = this->numel();
        out.data = (T *)malloc(n * sizeof(T));
        out.owns_memory = true;
        for (int i = 0; i < n; i++)
            out.data[i] = this->data[i] - o.data[i];

        return out;
    }

    __host__ ~tensor()
    {
        if (this->owns_memory)
        {
            std::free(shape);
            std::free(data);
        }
    }

    tensor(const tensor &) = delete;
    tensor &operator=(const tensor &) = delete;

    tensor(tensor &&o) noexcept
    {
        *this = std::move(o);
    }

    tensor &operator=(tensor &&o) noexcept
    {
        if (this != &o)
        {
            if (owns_memory)
            {
                std::free(shape);
                std::free(data);
            }
            n_dim = o.n_dim;
            shape = o.shape;
            data = o.data;
            curr_size = o.curr_size;
            owns_memory = o.owns_memory;

            o.n_dim = 0;
            o.shape = nullptr;
            o.data = nullptr;
            o.curr_size = 0;
            o.owns_memory = false;
        }
        return *this;
    }

    __host__ void print_tensor()
    {
        for (int i = 0; i < this->numel(); i++)
        {
            std::cout << "Tensor[" << i << "] = " << this->data[i] << std::endl;
        }
    }
};

template <class T>
__global__ void tensor_mul_kernel1D(const T *a, const T *b, T *out, size_t n)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = a[i] * b[i];
}

template <class T>
__global__ void tensor_mul_kernel2D(const T *A, const T *B, T *C, int M, int K, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < n)
    {
        T sum = (T)0;
        for (int i = 0; i < K; i++)
        {
            sum += A[row * K + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

template <class T>
tensor<T> mul_cuda(const tensor<T> &A, const tensor<T> &B)
{
    if (A.n_dim == 1)
    {
        if (!A.same_shape(B))
            throw std::runtime_error("1D elemwise requires same shape");

        tensor<T> out = tensor<T>::owning(A.n_dim, A.shape);

        size_t n = (size_t)A.numel();
        size_t bytes = n * sizeof(T);

        T *a_d = nullptr, *b_d = nullptr, *o_d = nullptr;
        CHECK_CUDA(cudaMalloc(&a_d, bytes));
        CHECK_CUDA(cudaMalloc(&b_d, bytes));
        CHECK_CUDA(cudaMalloc(&o_d, bytes));

        CHECK_CUDA(cudaMemcpy(a_d, A.raw(), bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(b_d, B.raw(), bytes, cudaMemcpyHostToDevice));

        int block = 256;
        int grid = (int)((n + block - 1) / block);

        tensor_mul_kernel1D<<<grid, block>>>(a_d, b_d, o_d, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(out.raw(), o_d, bytes, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(a_d));
        CHECK_CUDA(cudaFree(b_d));
        CHECK_CUDA(cudaFree(o_d));

        return out;
    }

    if (A.n_dim == 2)
    {
        if (B.n_dim != 2)
            throw std::runtime_error("2D matmul requires both tensors 2D");

        int M = A.shape[0];
        int K = A.shape[1];
        int K2 = B.shape[0];
        int N = B.shape[1];

        if (K != K2)
            throw std::runtime_error("2D matmul shape mismatch: A[M,K] vs B[K,N]");

        int out_shape[2] = {M, N};
        tensor<T> out = tensor<T>::owning(2, out_shape);

        size_t bytesA = (size_t)M * K * sizeof(T);
        size_t bytesB = (size_t)K * N * sizeof(T);
        size_t bytesC = (size_t)M * N * sizeof(T);

        T *a_d = nullptr, *b_d = nullptr, *c_d = nullptr;
        CHECK_CUDA(cudaMalloc(&a_d, bytesA));
        CHECK_CUDA(cudaMalloc(&b_d, bytesB));
        CHECK_CUDA(cudaMalloc(&c_d, bytesC));

        CHECK_CUDA(cudaMemcpy(a_d, A.raw(), bytesA, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(b_d, B.raw(), bytesB, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);

        tensor_mul_kernel2D<<<grid, block>>>(a_d, b_d, c_d, M, K, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(out.raw(), c_d, bytesC, cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(a_d));
        CHECK_CUDA(cudaFree(b_d));
        CHECK_CUDA(cudaFree(c_d));

        return out;
    }

    throw std::runtime_error("mul_cuda supports only n_dim == 1 (elemwise) or 2 (matmul)");
}
template <typename T>
void init_data(T *data, size_t len, T val)
{
    for (int i = 0; i < len; i++)
    {
        data[i] = val;
    }
}

int main(int argv, char **argc)
{
    int n_dim = 2;
    int *shape_a, *shape_b;
    complx *data_a, *data_b;

    shape_a = (int *)malloc(n_dim * sizeof(int));
    shape_b = (int *)malloc(n_dim * sizeof(int));

    for (int i = 0; i < 2; i++)
    {
        shape_a[i] = 2;
    }

    for (int i = 0; i < 2; i++)
    {
        shape_b[i] = 2;
    }

    data_a = (complx *)malloc(4 * sizeof(complx));
    data_b = (complx *)malloc(4 * sizeof(complx));

    init_data<complx>(data_a, 4, complx(1, 1));
    init_data<complx>(data_b, 4, complx(1, 1));

    tensor<complx> a = tensor<complx>(n_dim, shape_a, data_a);
    tensor<complx> b = tensor<complx>(n_dim, shape_b, data_b);
    tensor<complx> c = mul_cuda<complx>(a, b);

    c.print_tensor();

    free(shape_a);
    free(shape_b);
    free(data_a);
    free(data_b);

    return EXIT_SUCCESS;
}