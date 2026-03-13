#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void rotary_emb_kernel(
    const scalar_t* __restrict__ x,
    const c10::complex<float>* __restrict__ freqs,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t seq,
    int64_t heads,
    int64_t dim
) {
    const int64_t half_dim = dim >> 1;
    const int64_t total_pairs = bsz * seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t seq_idx = t % seq;
    const int64_t batch_idx = t / seq;

    const int64_t base = ((batch_idx * seq + seq_idx) * heads + head_idx) * dim + (pair_idx << 1);
    const float x0 = static_cast<float>(x[base]);
    const float x1 = static_cast<float>(x[base + 1]);
    const c10::complex<float> f = freqs[seq_idx * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const float y0 = x0 * fr - x1 * fi;
    const float y1 = x0 * fi + x1 * fr;
    out[base] = static_cast<scalar_t>(y0);
    out[base + 1] = static_cast<scalar_t>(y1);
}

template <typename scalar_t>
__global__ void rotary_emb_pair_kernel(
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    const c10::complex<float>* __restrict__ freqs,
    scalar_t* __restrict__ out1,
    scalar_t* __restrict__ out2,
    int64_t bsz,
    int64_t seq,
    int64_t heads,
    int64_t dim
) {
    const int64_t half_dim = dim >> 1;
    const int64_t total_pairs = bsz * seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t seq_idx = t % seq;
    const int64_t batch_idx = t / seq;

    const int64_t base = ((batch_idx * seq + seq_idx) * heads + head_idx) * dim + (pair_idx << 1);
    const c10::complex<float> f = freqs[seq_idx * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const float x10 = static_cast<float>(x1[base]);
    const float x11 = static_cast<float>(x1[base + 1]);
    out1[base] = static_cast<scalar_t>(x10 * fr - x11 * fi);
    out1[base + 1] = static_cast<scalar_t>(x10 * fi + x11 * fr);

    const float x20 = static_cast<float>(x2[base]);
    const float x21 = static_cast<float>(x2[base + 1]);
    out2[base] = static_cast<scalar_t>(x20 * fr - x21 * fi);
    out2[base + 1] = static_cast<scalar_t>(x20 * fi + x21 * fr);
}

torch::Tensor rotary_emb_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis) {
    auto out = torch::empty_like(x);
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t heads = x.size(2);
    const int64_t dim = x.size(3);
    const int64_t total_pairs = bsz * seq * heads * (dim >> 1);

    if (total_pairs == 0) {
        return out;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_rotary_emb_cuda",
        [&] {
            rotary_emb_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                freqs_cis.data_ptr<c10::complex<float>>(),
                out.data_ptr<scalar_t>(),
                bsz,
                seq,
                heads,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> rotary_emb_pair_forward_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor freqs_cis) {
    auto out1 = torch::empty_like(x1);
    auto out2 = torch::empty_like(x2);
    const int64_t bsz = x1.size(0);
    const int64_t seq = x1.size(1);
    const int64_t heads = x1.size(2);
    const int64_t dim = x1.size(3);
    const int64_t total_pairs = bsz * seq * heads * (dim >> 1);

    if (total_pairs == 0) {
        return {out1, out2};
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x1.scalar_type(),
        "qwen_image_rotary_emb_pair_cuda",
        [&] {
            rotary_emb_pair_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x1.data_ptr<scalar_t>(),
                x2.data_ptr<scalar_t>(),
                freqs_cis.data_ptr<c10::complex<float>>(),
                out1.data_ptr<scalar_t>(),
                out2.data_ptr<scalar_t>(),
                bsz,
                seq,
                heads,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out1, out2};
}

template <typename scalar_t, typename index_t>
__global__ void rotary_emb_indexed_kernel(
    const scalar_t* __restrict__ x,
    const c10::complex<float>* __restrict__ freqs,
    const index_t* __restrict__ token_indices,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t seq,
    int64_t heads,
    int64_t dim,
    int64_t total_freqs
) {
    const int64_t half_dim = dim >> 1;
    const int64_t total_pairs = bsz * seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t seq_idx = t % seq;
    const int64_t batch_idx = t / seq;

    const int64_t freq_row = static_cast<int64_t>(token_indices[seq_idx]);
    if (freq_row < 0 || freq_row >= total_freqs) {
        return;
    }
    const int64_t base = ((batch_idx * seq + seq_idx) * heads + head_idx) * dim + (pair_idx << 1);
    const float x0 = static_cast<float>(x[base]);
    const float x1 = static_cast<float>(x[base + 1]);
    const c10::complex<float> f = freqs[freq_row * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const float y0 = x0 * fr - x1 * fi;
    const float y1 = x0 * fi + x1 * fr;
    out[base] = static_cast<scalar_t>(y0);
    out[base + 1] = static_cast<scalar_t>(y1);
}

template <typename scalar_t, typename index_t>
__global__ void rotary_emb_pair_indexed_kernel(
    const scalar_t* __restrict__ x1,
    const scalar_t* __restrict__ x2,
    const c10::complex<float>* __restrict__ freqs,
    const index_t* __restrict__ token_indices,
    scalar_t* __restrict__ out1,
    scalar_t* __restrict__ out2,
    int64_t bsz,
    int64_t seq,
    int64_t heads,
    int64_t dim,
    int64_t total_freqs
) {
    const int64_t half_dim = dim >> 1;
    const int64_t total_pairs = bsz * seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t seq_idx = t % seq;
    const int64_t batch_idx = t / seq;

    const int64_t freq_row = static_cast<int64_t>(token_indices[seq_idx]);
    if (freq_row < 0 || freq_row >= total_freqs) {
        return;
    }
    const int64_t base = ((batch_idx * seq + seq_idx) * heads + head_idx) * dim + (pair_idx << 1);
    const c10::complex<float> f = freqs[freq_row * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const float x10 = static_cast<float>(x1[base]);
    const float x11 = static_cast<float>(x1[base + 1]);
    out1[base] = static_cast<scalar_t>(x10 * fr - x11 * fi);
    out1[base + 1] = static_cast<scalar_t>(x10 * fi + x11 * fr);

    const float x20 = static_cast<float>(x2[base]);
    const float x21 = static_cast<float>(x2[base + 1]);
    out2[base] = static_cast<scalar_t>(x20 * fr - x21 * fi);
    out2[base + 1] = static_cast<scalar_t>(x20 * fi + x21 * fr);
}

torch::Tensor rotary_emb_indexed_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis, torch::Tensor token_indices) {
    auto out = torch::empty_like(x);
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t heads = x.size(2);
    const int64_t dim = x.size(3);
    const int64_t total_freqs = freqs_cis.size(0);
    const int64_t total_pairs = bsz * seq * heads * (dim >> 1);

    if (total_pairs == 0) {
        return out;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_rotary_emb_indexed_cuda",
        [&] {
            if (token_indices.scalar_type() == at::kLong) {
                rotary_emb_indexed_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    freqs_cis.data_ptr<c10::complex<float>>(),
                    token_indices.data_ptr<int64_t>(),
                    out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    heads,
                    dim,
                    total_freqs
                );
            } else {
                rotary_emb_indexed_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    freqs_cis.data_ptr<c10::complex<float>>(),
                    token_indices.data_ptr<int>(),
                    out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    heads,
                    dim,
                    total_freqs
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> rotary_emb_pair_indexed_forward_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor freqs_cis,
    torch::Tensor token_indices
) {
    auto out1 = torch::empty_like(x1);
    auto out2 = torch::empty_like(x2);
    const int64_t bsz = x1.size(0);
    const int64_t seq = x1.size(1);
    const int64_t heads = x1.size(2);
    const int64_t dim = x1.size(3);
    const int64_t total_freqs = freqs_cis.size(0);
    const int64_t total_pairs = bsz * seq * heads * (dim >> 1);

    if (total_pairs == 0) {
        return {out1, out2};
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x1.scalar_type(),
        "qwen_image_rotary_emb_pair_indexed_cuda",
        [&] {
            if (token_indices.scalar_type() == at::kLong) {
                rotary_emb_pair_indexed_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                    x1.data_ptr<scalar_t>(),
                    x2.data_ptr<scalar_t>(),
                    freqs_cis.data_ptr<c10::complex<float>>(),
                    token_indices.data_ptr<int64_t>(),
                    out1.data_ptr<scalar_t>(),
                    out2.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    heads,
                    dim,
                    total_freqs
                );
            } else {
                rotary_emb_pair_indexed_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
                    x1.data_ptr<scalar_t>(),
                    x2.data_ptr<scalar_t>(),
                    freqs_cis.data_ptr<c10::complex<float>>(),
                    token_indices.data_ptr<int>(),
                    out1.data_ptr<scalar_t>(),
                    out2.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    heads,
                    dim,
                    total_freqs
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out1, out2};
}

template <typename scalar_t>
__global__ void rotary_cat_qkv_kernel(
    const scalar_t* __restrict__ img_q,
    const scalar_t* __restrict__ img_k,
    const scalar_t* __restrict__ img_v,
    const scalar_t* __restrict__ txt_q,
    const scalar_t* __restrict__ txt_k,
    const scalar_t* __restrict__ txt_v,
    const c10::complex<float>* __restrict__ img_freqs,
    const c10::complex<float>* __restrict__ txt_freqs,
    scalar_t* __restrict__ joint_q,
    scalar_t* __restrict__ joint_k,
    scalar_t* __restrict__ joint_v,
    int64_t bsz,
    int64_t txt_seq,
    int64_t img_seq,
    int64_t heads,
    int64_t dim
) {
    const int64_t half_dim = dim >> 1;
    const int64_t joint_seq = txt_seq + img_seq;
    const int64_t total_pairs = bsz * joint_seq * heads * half_dim;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int64_t t = idx;
    const int64_t pair_idx = t % half_dim;
    t /= half_dim;
    const int64_t head_idx = t % heads;
    t /= heads;
    const int64_t joint_s = t % joint_seq;
    const int64_t b = t / joint_seq;

    const bool is_txt = joint_s < txt_seq;
    const int64_t src_s = is_txt ? joint_s : (joint_s - txt_seq);
    const int64_t src_base = ((b * (is_txt ? txt_seq : img_seq) + src_s) * heads + head_idx) * dim + (pair_idx << 1);
    const int64_t dst_base = ((b * joint_seq + joint_s) * heads + head_idx) * dim + (pair_idx << 1);

    const c10::complex<float> f =
        is_txt ? txt_freqs[src_s * half_dim + pair_idx] : img_freqs[src_s * half_dim + pair_idx];
    const float fr = f.real();
    const float fi = f.imag();

    const scalar_t* src_q = is_txt ? txt_q : img_q;
    const scalar_t* src_k = is_txt ? txt_k : img_k;
    const scalar_t* src_v = is_txt ? txt_v : img_v;

    const float q0 = static_cast<float>(src_q[src_base]);
    const float q1 = static_cast<float>(src_q[src_base + 1]);
    joint_q[dst_base] = static_cast<scalar_t>(q0 * fr - q1 * fi);
    joint_q[dst_base + 1] = static_cast<scalar_t>(q0 * fi + q1 * fr);

    const float k0 = static_cast<float>(src_k[src_base]);
    const float k1 = static_cast<float>(src_k[src_base + 1]);
    joint_k[dst_base] = static_cast<scalar_t>(k0 * fr - k1 * fi);
    joint_k[dst_base + 1] = static_cast<scalar_t>(k0 * fi + k1 * fr);

    joint_v[dst_base] = src_v[src_base];
    joint_v[dst_base + 1] = src_v[src_base + 1];
}

std::vector<torch::Tensor> rotary_cat_qkv_forward_cuda(
    torch::Tensor img_q,
    torch::Tensor img_k,
    torch::Tensor img_v,
    torch::Tensor txt_q,
    torch::Tensor txt_k,
    torch::Tensor txt_v,
    torch::Tensor img_freqs,
    torch::Tensor txt_freqs
) {
    const int64_t bsz = img_q.size(0);
    const int64_t img_seq = img_q.size(1);
    const int64_t txt_seq = txt_q.size(1);
    const int64_t heads = img_q.size(2);
    const int64_t dim = img_q.size(3);
    const int64_t joint_seq = txt_seq + img_seq;

    auto joint_q = torch::empty({bsz, joint_seq, heads, dim}, img_q.options());
    auto joint_k = torch::empty_like(joint_q);
    auto joint_v = torch::empty_like(joint_q);

    const int64_t total_pairs = bsz * joint_seq * heads * (dim >> 1);
    if (total_pairs == 0) {
        return {joint_q, joint_k, joint_v};
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_pairs + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        img_q.scalar_type(),
        "qwen_image_rotary_cat_qkv_cuda",
        [&] {
            rotary_cat_qkv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                img_q.data_ptr<scalar_t>(),
                img_k.data_ptr<scalar_t>(),
                img_v.data_ptr<scalar_t>(),
                txt_q.data_ptr<scalar_t>(),
                txt_k.data_ptr<scalar_t>(),
                txt_v.data_ptr<scalar_t>(),
                img_freqs.data_ptr<c10::complex<float>>(),
                txt_freqs.data_ptr<c10::complex<float>>(),
                joint_q.data_ptr<scalar_t>(),
                joint_k.data_ptr<scalar_t>(),
                joint_v.data_ptr<scalar_t>(),
                bsz,
                txt_seq,
                img_seq,
                heads,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {joint_q, joint_k, joint_v};
}


template <typename scalar_t>
__global__ void modulate_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ scale,
    scalar_t* __restrict__ modulated,
    int64_t bsz,
    int64_t seq,
    int64_t dim
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = bsz * seq * dim;
    if (idx >= total) return;

    const int64_t d = idx % dim;
    const int64_t t = idx / dim;
    const int64_t b = t / seq;
    const int64_t base_vec = b * dim + d;

    const scalar_t xv = x[idx];
    const scalar_t shiftv = shift[base_vec];
    const scalar_t scalev = scale[base_vec];
    modulated[idx] = xv * (static_cast<scalar_t>(1.0f) + scalev) + shiftv;
}


template <typename scalar_t, typename index_t>
__global__ void modulate_indexed_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ shift,
    const scalar_t* __restrict__ scale,
    const scalar_t* __restrict__ gate,
    const index_t* __restrict__ index,
    scalar_t* __restrict__ modulated,
    scalar_t* __restrict__ gate_out,
    int64_t bsz,
    int64_t seq,
    int64_t dim,
    int64_t index_bsz
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = bsz * seq * dim;
    if (idx >= total) return;

    const int64_t d = idx % dim;
    const int64_t t = idx / dim;
    const int64_t s = t % seq;
    const int64_t b = t / seq;

    const int64_t ib = (index_bsz == 1) ? 0 : b;
    const int64_t index_offset = (ib * seq + s);
    const int64_t select = static_cast<int64_t>(index[index_offset]) == 0 ? b : (b + bsz);
    const int64_t base_vec = select * dim + d;

    const scalar_t xv = x[idx];
    const scalar_t shiftv = shift[base_vec];
    const scalar_t scalev = scale[base_vec];
    modulated[idx] = xv * (static_cast<scalar_t>(1.0f) + scalev) + shiftv;
    gate_out[idx] = gate[base_vec];
}


std::vector<torch::Tensor> modulate_forward_cuda(torch::Tensor x, torch::Tensor mod_params) {
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t dim = x.size(2);
    auto modulated = torch::empty_like(x);

    const int64_t total = bsz * seq * dim;
    if (total == 0) {
        auto gate_out = mod_params.narrow(1, dim * 2, dim).unsqueeze(1).contiguous();
        return {modulated, gate_out};
    }

    auto shift = mod_params.narrow(1, 0, dim).contiguous();
    auto scale = mod_params.narrow(1, dim, dim).contiguous();
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_modulate_cuda",
        [&] {
            modulate_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                shift.data_ptr<scalar_t>(),
                scale.data_ptr<scalar_t>(),
                modulated.data_ptr<scalar_t>(),
                bsz,
                seq,
                dim
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto gate_out = mod_params.narrow(1, dim * 2, dim).unsqueeze(1).contiguous();
    return {modulated, gate_out};
}


std::vector<torch::Tensor> modulate_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor mod_params,
    torch::Tensor index
) {
    const int64_t bsz = x.size(0);
    const int64_t seq = x.size(1);
    const int64_t dim = x.size(2);
    const int64_t index_bsz = index.size(0);
    auto modulated = torch::empty_like(x);
    auto gate_out = torch::empty_like(x);

    const int64_t total = bsz * seq * dim;
    if (total == 0) {
        return {modulated, gate_out};
    }

    auto shift = mod_params.narrow(1, 0, dim).contiguous();
    auto scale = mod_params.narrow(1, dim, dim).contiguous();
    auto gate = mod_params.narrow(1, dim * 2, dim).contiguous();
    auto index_2d = index.squeeze(-1);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        x.scalar_type(),
        "qwen_image_modulate_indexed_cuda",
        [&] {
            if (index_2d.scalar_type() == at::kLong) {
                modulate_indexed_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    gate.data_ptr<scalar_t>(),
                    index_2d.data_ptr<int64_t>(),
                    modulated.data_ptr<scalar_t>(),
                    gate_out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    dim,
                    index_bsz
                );
            } else {
                modulate_indexed_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
                    x.data_ptr<scalar_t>(),
                    shift.data_ptr<scalar_t>(),
                    scale.data_ptr<scalar_t>(),
                    gate.data_ptr<scalar_t>(),
                    index_2d.data_ptr<int>(),
                    modulated.data_ptr<scalar_t>(),
                    gate_out.data_ptr<scalar_t>(),
                    bsz,
                    seq,
                    dim,
                    index_bsz
                );
            }
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {modulated, gate_out};
}


template <typename scalar_t>
__global__ void gated_residual_kernel(
    const scalar_t* __restrict__ base,
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ update,
    scalar_t* __restrict__ out,
    int64_t bsz,
    int64_t seq,
    int64_t dim,
    int64_t gate_seq
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = bsz * seq * dim;
    if (idx >= total) return;

    const int64_t d = idx % dim;
    const int64_t t = idx / dim;
    const int64_t s = t % seq;
    const int64_t b = t / seq;
    const int64_t gs = (gate_seq == 1) ? 0 : s;
    const int64_t gate_idx = (b * gate_seq + gs) * dim + d;

    out[idx] = base[idx] + gate[gate_idx] * update[idx];
}


torch::Tensor gated_residual_forward_cuda(torch::Tensor base, torch::Tensor gate, torch::Tensor update) {
    auto out = torch::empty_like(base);
    const int64_t bsz = base.size(0);
    const int64_t seq = base.size(1);
    const int64_t dim = base.size(2);
    const int64_t gate_seq = gate.size(1);
    const int64_t total = bsz * seq * dim;

    if (total == 0) {
        return out;
    }

    constexpr int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        base.scalar_type(),
        "qwen_image_gated_residual_cuda",
        [&] {
            gated_residual_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                base.data_ptr<scalar_t>(),
                gate.data_ptr<scalar_t>(),
                update.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                bsz,
                seq,
                dim,
                gate_seq
            );
        }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
