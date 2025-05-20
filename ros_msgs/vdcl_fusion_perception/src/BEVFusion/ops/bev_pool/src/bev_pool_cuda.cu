#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <type_traits>


/*
  Function: pillar pooling
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    x                : input features, FloatTensor[n, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    out              : output features, FloatTensor[b, d, h, w, c]
*/
__global__ void bev_pool_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ x,
                                  const int *__restrict__ geom_feats,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];
  const int* cur_geom_feats = geom_feats + interval_start * 4;
  const float* cur_x = x + interval_start * c + cur_c;
  float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
    cur_geom_feats[1] * c + cur_c;
  float psum = 0;
  for(int i = 0; i < interval_length; i++){
    psum += cur_x[i * c];
  }
  *cur_out = psum;
}

/*
  Function: pillar pooling backward
  Args:
    b                : batch size
    d                : depth of the feature map
    h                : height of pooled feature map
    w                : width of pooled feature map
    n                : number of input points
    c                : number of channels
    n_intervals      : number of unique points
    out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
    geom_feats       : input coordinates, IntTensor[n, 4]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
    x_grad           : gradient of the image fmap, FloatTensor
*/
__global__ void bev_pool_grad_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                  const float *__restrict__ out_grad,
                                  const int *__restrict__ geom_feats,
                                  const int *__restrict__ interval_starts,
                                  const int *__restrict__ interval_lengths,
                                  float* __restrict__ x_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  const int* cur_geom_feats = geom_feats + interval_start * 4;
  float* cur_x_grad = x_grad + interval_start * c + cur_c;

  const float* cur_out_grad = out_grad + cur_geom_feats[3] * d * h * w * c +
    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
    cur_geom_feats[1] * c + cur_c;
  for(int i = 0; i < interval_length; i++){
    cur_x_grad[i * c] = *cur_out_grad;
  }
}

// // MEAN pooling
// __global__ void bev_pool_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
//                                 const float *__restrict__ x,
//                                 const int *__restrict__ geom_feats,
//                                 const int *__restrict__ interval_starts,
//                                 const int *__restrict__ interval_lengths,
//                                 float* __restrict__ out) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int index = idx / c;
//   int cur_c = idx % c;
//   if (index >= n_intervals) return;
  
//   int interval_start = interval_starts[index];
//   int interval_length = interval_lengths[index];

//   if (interval_length == 0) return;
  
//   const int* cur_geom_feats = geom_feats + interval_start * 4;
//   const float* cur_x = x + interval_start * c + cur_c;
//   float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
//                    cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
//                    cur_geom_feats[1] * c + cur_c;

//   float psum = 0;
//   for (int i = 0; i < interval_length; i++) {
//       psum += cur_x[i * c];
//   }
  
//   *cur_out = psum / interval_length;
// }

// __global__ void bev_pool_grad_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
//                                      const float *__restrict__ out_grad,
//                                      const int *__restrict__ geom_feats,
//                                      const int *__restrict__ interval_starts,
//                                      const int *__restrict__ interval_lengths,
//                                      float* __restrict__ x_grad) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   int index = idx / c;
//   int cur_c = idx % c;
//   if (index >= n_intervals) return;

//   int interval_start = interval_starts[index];
//   int interval_length = interval_lengths[index];

//   if (interval_length == 0) return;

//   const int* cur_geom_feats = geom_feats + interval_start * 4;
//   float* cur_x_grad = x_grad + interval_start * c + cur_c;

//   const float* cur_out_grad = out_grad + cur_geom_feats[3] * d * h * w * c +
//                               cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
//                               cur_geom_feats[1] * c + cur_c;

//   for (int i = 0; i < interval_length; i++) {
//     cur_x_grad[i * c] = *cur_out_grad / interval_length;
//   }
// }

void bev_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out) {
  bev_pool_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts, interval_lengths, out
  );
}

void bev_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
  const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad) {
  bev_pool_grad_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    b, d, h, w, n, c, n_intervals, out_grad, geom_feats, interval_starts, interval_lengths, x_grad
  );
}


__global__ void bev_mean_pool_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                const float *__restrict__ x,
                                const int *__restrict__ geom_feats,
                                const int *__restrict__ interval_starts,
                                const int *__restrict__ interval_lengths,
                                float* __restrict__ out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;
  
  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  if (interval_length == 0) return;
  
  const int* cur_geom_feats = geom_feats + interval_start * 4;
  const float* cur_x = x + interval_start * c + cur_c;
  float* cur_out = out + cur_geom_feats[3] * d * h * w * c +
                   cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
                   cur_geom_feats[1] * c + cur_c;

  float psum = 0;
  for (int i = 0; i < interval_length; i++) {
      psum += cur_x[i * c];
  }
  
  *cur_out = psum / interval_length;
}

__global__ void bev_mean_pool_grad_kernel(int b, int d, int h, int w, int n, int c, int n_intervals,
                                     const float *__restrict__ out_grad,
                                     const int *__restrict__ geom_feats,
                                     const int *__restrict__ interval_starts,
                                     const int *__restrict__ interval_lengths,
                                     float* __restrict__ x_grad) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_intervals) return;

  int interval_start = interval_starts[index];
  int interval_length = interval_lengths[index];

  if (interval_length == 0) return;

  const int* cur_geom_feats = geom_feats + interval_start * 4;
  float* cur_x_grad = x_grad + interval_start * c + cur_c;

  const float* cur_out_grad = out_grad + cur_geom_feats[3] * d * h * w * c +
                              cur_geom_feats[2] * h * w * c + cur_geom_feats[0] * w * c +
                              cur_geom_feats[1] * c + cur_c;

  for (int i = 0; i < interval_length; i++) {
    cur_x_grad[i * c] = *cur_out_grad / interval_length;
  }
}

// void bev_mean_pool(int b, int d, int h, int w, int n, int c, int n_intervals, const float* x,
//   const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* out) {
//   bev_mean_pool_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
//     b, d, h, w, n, c, n_intervals, x, geom_feats, interval_starts, interval_lengths, out
//   );
// }

// void bev_mean_pool_grad(int b, int d, int h, int w, int n, int c, int n_intervals, const float* out_grad,
//   const int* geom_feats, const int* interval_starts, const int* interval_lengths, float* x_grad) {
//   bev_mean_pool_grad_kernel<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
//     b, d, h, w, n, c, n_intervals, out_grad, geom_feats, interval_starts, interval_lengths, x_grad
//   );
// }

// ────────── 타입 안전 변환 헬퍼 ──────────
template <typename T>
__device__ inline float toFloat(T v) {      // 기본: 이미 float
    return v;
}
template <>
__device__ inline float toFloat<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ inline T toScalar(float v) {     // 기본: 이미 float
    return v;
}
template <>
__device__ inline __half toScalar<__half>(float v) {
    return __float2half(v);
}

template <typename scalar_t>
__global__ void bev_mean_pool_kernel_t(
    int b,int d,int h,int w,int n,int c,int n_intervals,
    const scalar_t * __restrict__ x,
    const int      * __restrict__ geom,
    const int      * __restrict__ starts,
    const int      * __restrict__ lens,
    scalar_t       * __restrict__ out)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int iv    = idx / c;          // interval index
    int ch    = idx % c;          // channel
    if (iv >= n_intervals)  return;

    int s  = starts[iv];
    int L  = lens[iv];
    if (L == 0)  return;

    const int *g = geom + s * 4;
    const scalar_t* src = x   + s * c + ch;
    scalar_t*       dst = out + g[3]*d*h*w*c + g[2]*h*w*c +
                          g[0]*w*c        + g[1]*c + ch;

    // 누적은 float32 로
    float acc = 0.f;
    #pragma unroll
    for(int i=0;i<L;++i)
        acc += toFloat(src[i*c]);   // 타입별 변환

    *dst = toScalar<scalar_t>(acc / L);
}

template <typename scalar_t>
__global__ void bev_mean_pool_grad_kernel_t(
    int b,int d,int h,int w,int n,int c,int n_intervals,
    const scalar_t * __restrict__ grad_out,
    const int      * __restrict__ geom,
    const int      * __restrict__ starts,
    const int      * __restrict__ lens,
    scalar_t       * __restrict__ grad_x)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int iv  = idx / c;
    int ch  = idx % c;
    if (iv >= n_intervals) return;

    int s = starts[iv];
    int L = lens[iv];
    if (L == 0) return;

    const int* g   = geom + s * 4;
    scalar_t*  dst = grad_x + s * c + ch;
    const scalar_t* go = grad_out + g[3]*d*h*w*c + g[2]*h*w*c +
                         g[0]*w*c        + g[1]*c + ch;
    scalar_t gval = toScalar<scalar_t>( toFloat(*go) / L );
    #pragma unroll
    for(int i=0;i<L;++i) dst[i*c] = gval;
}

// -----------------------------------------------------------------------------
// 런처: dtype 검사 후 템플릿 인스턴스 선택
// -----------------------------------------------------------------------------
void bev_mean_pool(int b,int d,int h,int w,int n,int c,int n_intervals,
                   const void* x, const int* geom,
                   const int* starts,const int* lens, void* out,
                   cudaStream_t stream, bool use_half)
{
    int threads = 256;
    int blocks  = (n_intervals * c + threads - 1) / threads;
    if (use_half)
        bev_mean_pool_kernel_t<__half><<<blocks,threads,0,stream>>>(
            b,d,h,w,n,c,n_intervals,
            static_cast<const __half*>(x), geom, starts, lens,
            static_cast<__half*>(out));
    else
        bev_mean_pool_kernel_t<float><<<blocks,threads,0,stream>>>(
            b,d,h,w,n,c,n_intervals,
            static_cast<const float*>(x), geom, starts, lens,
            static_cast<float*>(out));
}

void bev_mean_pool_grad(int b,int d,int h,int w,int n,int c,int n_intervals,
                        const void* grad_out, const int* geom,
                        const int* starts,const int* lens, void* grad_x,
                        cudaStream_t stream, bool use_half)
{
    int threads = 256;
    int blocks  = (n_intervals * c + threads - 1) / threads;
    if (use_half)
        bev_mean_pool_grad_kernel_t<__half><<<blocks,threads,0,stream>>>(
            b,d,h,w,n,c,n_intervals,
            static_cast<const __half*>(grad_out), geom, starts, lens,
            static_cast<__half*>(grad_x));
    else
        bev_mean_pool_grad_kernel_t<float><<<blocks,threads,0,stream>>>(
            b,d,h,w,n,c,n_intervals,
            static_cast<const float*>(grad_out), geom, starts, lens,
            static_cast<float*>(grad_x));
}