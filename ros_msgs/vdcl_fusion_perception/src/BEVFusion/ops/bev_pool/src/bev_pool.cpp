#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

// CUDA function declarations
void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
    const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out);

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad);


/*
  Function: pillar pooling (forward, cuda)
  Args:
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    out              : output features, FloatTensor[b, d, h, w, c]
*/
at::Tensor bev_pool_forward(
  const at::Tensor _x,
  const at::Tensor _geom_feats,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _x.size(0);
  int c = _x.size(1);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_x));
  const float* x = _x.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_x.dtype()).device(_x.device());
  at::Tensor _out = torch::zeros({b, d, h, w, c}, options);
  float* out = _out.data_ptr<float>();
  bev_pool(
    b, d, h, w, n, c, n_intervals, x,
    geom_feats, interval_starts, interval_lengths, out
  );
  return _out;
}


/*
  Function: pillar pooling (backward, cuda)
  Args:
    out_grad         : input features, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
    x_grad           : output features, FloatTensor[n, 4]
*/
at::Tensor bev_pool_backward(
  const at::Tensor _out_grad,
  const at::Tensor _geom_feats,
  const at::Tensor _interval_lengths,
  const at::Tensor _interval_starts,
  int b, int d, int h, int w
) {
  int n = _geom_feats.size(0);
  int c = _out_grad.size(4);
  int n_intervals = _interval_lengths.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(_out_grad));
  const float* out_grad = _out_grad.data_ptr<float>();
  const int* geom_feats = _geom_feats.data_ptr<int>();
  const int* interval_lengths = _interval_lengths.data_ptr<int>();
  const int* interval_starts = _interval_starts.data_ptr<int>();

  auto options =
      torch::TensorOptions().dtype(_out_grad.dtype()).device(_out_grad.device());
  at::Tensor _x_grad = torch::zeros({n, c}, options);
  float* x_grad = _x_grad.data_ptr<float>();

  bev_pool_grad(
    b, d, h, w, n, c, n_intervals, out_grad,
    geom_feats, interval_starts, interval_lengths, x_grad
  );

  return _x_grad;
}

// ───────────────────────────────────────────────────────────────
// CUDA 함수 (앞서 작성한 템플릿 런처 선언)
//   • use_half == true  → __half 버전 호출
//   • use_half == false → float  버전 호출
// ───────────────────────────────────────────────────────────────
void bev_mean_pool( int b,int d,int h,int w,int n,int c,int n_i,
                    const void* x,const int* geom,const int* starts,const int* lens,
                    void* out, cudaStream_t stream, bool use_half );

void bev_mean_pool_grad( int b,int d,int h,int w,int n,int c,int n_i,
                         const void* gout,const int* geom,const int* starts,const int* lens,
                         void* gx, cudaStream_t stream, bool use_half );

// ───────────────────────────────────────────────────────────────
// Forward
// ───────────────────────────────────────────────────────────────
at::Tensor bev_mean_pool_forward( const at::Tensor _x,
                                  const at::Tensor _geom,
                                  const at::Tensor _lens,
                                  const at::Tensor _starts,
                                  int B, int D, int H, int W )
{
    TORCH_CHECK(_x.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(_x.is_contiguous(), "input must be contiguous");

    int N   = _x.size(0);
    int C   = _x.size(1);
    int N_i = _lens.size(0);

    auto opts = _x.options();
    at::Tensor out = at::zeros({B, D, H, W, C}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool use_half = (_x.scalar_type() == at::kHalf);

    bev_mean_pool(B,D,H,W, N,C,N_i,
                  _x.data_ptr(), _geom.data_ptr<int>(),
                  _starts.data_ptr<int>(), _lens.data_ptr<int>(),
                  out.data_ptr(), stream, use_half);
    return out;
}

// ───────────────────────────────────────────────────────────────
// Backward
// ───────────────────────────────────────────────────────────────
at::Tensor bev_mean_pool_backward( const at::Tensor _gout,
                                   const at::Tensor _geom,
                                   const at::Tensor _lens,
                                   const at::Tensor _starts,
                                   int B,int D,int H,int W )
{
    TORCH_CHECK(_gout.is_cuda(), "grad_out must be CUDA tensor");
    TORCH_CHECK(_gout.is_contiguous(), "grad_out must be contiguous");

    int N   = _geom.size(0);
    int C   = _gout.size(4);
    int N_i = _lens.size(0);

    auto opts = _gout.options();
    at::Tensor gx = at::zeros({N, C}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bool use_half = (_gout.scalar_type() == at::kHalf);

    bev_mean_pool_grad(B,D,H,W, N,C,N_i,
                       _gout.data_ptr(), _geom.data_ptr<int>(),
                       _starts.data_ptr<int>(), _lens.data_ptr<int>(),
                       gx.data_ptr(), stream, use_half);
    return gx;
}

// ───────────────────────────────────────────────────────────────
// 기존 bev_pool(float 전용) 바인딩은 그대로 유지
// ───────────────────────────────────────────────────────────────
extern at::Tensor bev_pool_forward(const at::Tensor, const at::Tensor,
                                   const at::Tensor,const at::Tensor,int,int,int,int);
extern at::Tensor bev_pool_backward(const at::Tensor, const at::Tensor,
                                    const at::Tensor,const at::Tensor,int,int,int,int);

// ───────────────────────────────────────────────────────────────
// PYBIND11 모듈
// ───────────────────────────────────────────────────────────────
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bev_pool_forward",   &bev_pool_forward,   "bev_pool_forward");
    m.def("bev_pool_backward",  &bev_pool_backward,  "bev_pool_backward");
    m.def("bev_mean_pool_forward",  &bev_mean_pool_forward,  "bev_mean_pool_forward (float/half)");
    m.def("bev_mean_pool_backward", &bev_mean_pool_backward, "bev_mean_pool_backward (float/half)");
}