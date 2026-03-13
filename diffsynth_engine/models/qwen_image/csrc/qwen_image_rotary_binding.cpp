#include <torch/extension.h>

torch::Tensor rotary_emb_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis);
torch::Tensor rotary_emb_indexed_forward_cuda(torch::Tensor x, torch::Tensor freqs_cis, torch::Tensor token_indices);
std::vector<torch::Tensor> rotary_emb_pair_forward_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor freqs_cis);
std::vector<torch::Tensor> rotary_emb_pair_indexed_forward_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor freqs_cis,
    torch::Tensor token_indices
);
std::vector<torch::Tensor> rotary_cat_qkv_forward_cuda(
    torch::Tensor img_q,
    torch::Tensor img_k,
    torch::Tensor img_v,
    torch::Tensor txt_q,
    torch::Tensor txt_k,
    torch::Tensor txt_v,
    torch::Tensor img_freqs,
    torch::Tensor txt_freqs
);
std::vector<torch::Tensor> modulate_forward_cuda(torch::Tensor x, torch::Tensor mod_params);
std::vector<torch::Tensor> modulate_indexed_forward_cuda(
    torch::Tensor x,
    torch::Tensor mod_params,
    torch::Tensor index
);
torch::Tensor gated_residual_forward_cuda(torch::Tensor base, torch::Tensor gate, torch::Tensor update);

torch::Tensor rotary_emb_forward(torch::Tensor x, torch::Tensor freqs_cis) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(freqs_cis.is_cuda(), "freqs_cis must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(freqs_cis.is_contiguous(), "freqs_cis must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, H, D]");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must have shape [S, D/2]");
    TORCH_CHECK(x.size(1) == freqs_cis.size(0), "x sequence dimension must match freqs_cis");
    TORCH_CHECK((x.size(3) % 2) == 0, "last dim of x must be even");
    TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(x.size(3) / 2 == freqs_cis.size(1), "freqs_cis second dim must be D/2");
    return rotary_emb_forward_cuda(x, freqs_cis);
}

torch::Tensor rotary_emb_indexed_forward(torch::Tensor x, torch::Tensor freqs_cis, torch::Tensor token_indices) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(freqs_cis.is_cuda(), "freqs_cis must be a CUDA tensor");
    TORCH_CHECK(token_indices.is_cuda(), "token_indices must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(freqs_cis.is_contiguous(), "freqs_cis must be contiguous");
    TORCH_CHECK(token_indices.is_contiguous(), "token_indices must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have shape [B, S, H, D]");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must have shape [S_all, D/2]");
    TORCH_CHECK(token_indices.dim() == 1, "token_indices must be 1D");
    TORCH_CHECK(x.size(1) == token_indices.size(0), "x sequence dimension must equal token_indices length");
    TORCH_CHECK((x.size(3) % 2) == 0, "last dim of x must be even");
    TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(x.size(3) / 2 == freqs_cis.size(1), "freqs_cis second dim must be D/2");
    TORCH_CHECK(token_indices.scalar_type() == at::kLong || token_indices.scalar_type() == at::kInt,
                "token_indices must be int32 or int64");
    return rotary_emb_indexed_forward_cuda(x, freqs_cis, token_indices);
}

std::vector<torch::Tensor> rotary_emb_pair_forward(torch::Tensor x1, torch::Tensor x2, torch::Tensor freqs_cis) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda() && freqs_cis.is_cuda(), "x1/x2/freqs_cis must be CUDA tensors");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous() && freqs_cis.is_contiguous(), "x1/x2/freqs_cis must be contiguous");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    TORCH_CHECK(x1.dim() == 4, "x1 must have shape [B, S, H, D]");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must have shape [S, D/2]");
    TORCH_CHECK(x1.size(1) == freqs_cis.size(0), "x1 sequence dimension must match freqs_cis");
    TORCH_CHECK((x1.size(3) % 2) == 0, "last dim of x1 must be even");
    TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "x1 and x2 dtypes must match");
    TORCH_CHECK(
        (x1.scalar_type() == at::kFloat) || (x1.scalar_type() == at::kHalf) || (x1.scalar_type() == at::kBFloat16),
        "x1/x2 dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(x1.size(3) / 2 == freqs_cis.size(1), "freqs_cis second dim must be D/2");
    return rotary_emb_pair_forward_cuda(x1, x2, freqs_cis);
}

std::vector<torch::Tensor> rotary_emb_pair_indexed_forward(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor freqs_cis,
    torch::Tensor token_indices
) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda() && freqs_cis.is_cuda() && token_indices.is_cuda(),
                "x1/x2/freqs_cis/token_indices must be CUDA tensors");
    TORCH_CHECK(x1.is_contiguous() && x2.is_contiguous() && freqs_cis.is_contiguous() && token_indices.is_contiguous(),
                "x1/x2/freqs_cis/token_indices must be contiguous");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");
    TORCH_CHECK(x1.dim() == 4, "x1 must have shape [B, S, H, D]");
    TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must have shape [S_all, D/2]");
    TORCH_CHECK(token_indices.dim() == 1, "token_indices must be 1D");
    TORCH_CHECK(x1.size(1) == token_indices.size(0), "x1 sequence dimension must equal token_indices length");
    TORCH_CHECK((x1.size(3) % 2) == 0, "last dim of x1 must be even");
    TORCH_CHECK(freqs_cis.scalar_type() == at::kComplexFloat, "freqs_cis must be complex64");
    TORCH_CHECK(x1.scalar_type() == x2.scalar_type(), "x1 and x2 dtypes must match");
    TORCH_CHECK(
        (x1.scalar_type() == at::kFloat) || (x1.scalar_type() == at::kHalf) || (x1.scalar_type() == at::kBFloat16),
        "x1/x2 dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(x1.size(3) / 2 == freqs_cis.size(1), "freqs_cis second dim must be D/2");
    TORCH_CHECK(token_indices.scalar_type() == at::kLong || token_indices.scalar_type() == at::kInt,
                "token_indices must be int32 or int64");
    return rotary_emb_pair_indexed_forward_cuda(x1, x2, freqs_cis, token_indices);
}

std::vector<torch::Tensor> rotary_cat_qkv_forward(
    torch::Tensor img_q,
    torch::Tensor img_k,
    torch::Tensor img_v,
    torch::Tensor txt_q,
    torch::Tensor txt_k,
    torch::Tensor txt_v,
    torch::Tensor img_freqs,
    torch::Tensor txt_freqs
) {
    TORCH_CHECK(img_q.is_cuda() && img_k.is_cuda() && img_v.is_cuda(), "image q/k/v must be CUDA tensors");
    TORCH_CHECK(txt_q.is_cuda() && txt_k.is_cuda() && txt_v.is_cuda(), "text q/k/v must be CUDA tensors");
    TORCH_CHECK(img_freqs.is_cuda() && txt_freqs.is_cuda(), "img_freqs/txt_freqs must be CUDA tensors");
    TORCH_CHECK(
        img_q.is_contiguous() && img_k.is_contiguous() && img_v.is_contiguous() && txt_q.is_contiguous() &&
            txt_k.is_contiguous() && txt_v.is_contiguous() && img_freqs.is_contiguous() && txt_freqs.is_contiguous(),
        "all inputs must be contiguous"
    );
    TORCH_CHECK(img_q.dim() == 4 && txt_q.dim() == 4, "q tensors must be [B, S, H, D]");
    TORCH_CHECK(img_k.sizes() == img_q.sizes() && img_v.sizes() == img_q.sizes(), "img k/v must match img q shape");
    TORCH_CHECK(txt_k.sizes() == txt_q.sizes() && txt_v.sizes() == txt_q.sizes(), "txt k/v must match txt q shape");
    TORCH_CHECK(img_q.size(0) == txt_q.size(0), "image/text batch must match");
    TORCH_CHECK(img_q.size(2) == txt_q.size(2) && img_q.size(3) == txt_q.size(3), "image/text H/D must match");
    TORCH_CHECK((img_q.size(3) % 2) == 0, "last dim must be even");
    TORCH_CHECK(img_q.scalar_type() == txt_q.scalar_type(), "image/text dtypes must match");
    TORCH_CHECK(img_q.scalar_type() == img_k.scalar_type() && img_q.scalar_type() == img_v.scalar_type() &&
                    img_q.scalar_type() == txt_k.scalar_type() && img_q.scalar_type() == txt_v.scalar_type(),
                "all q/k/v dtypes must match");
    TORCH_CHECK(
        (img_q.scalar_type() == at::kFloat) || (img_q.scalar_type() == at::kHalf) || (img_q.scalar_type() == at::kBFloat16),
        "q/k/v dtype must be float32, float16, or bfloat16"
    );
    TORCH_CHECK(img_freqs.dim() == 2 && txt_freqs.dim() == 2, "freqs must be [S, D/2]");
    TORCH_CHECK(img_freqs.scalar_type() == at::kComplexFloat && txt_freqs.scalar_type() == at::kComplexFloat,
                "freqs must be complex64");
    TORCH_CHECK(img_freqs.size(0) == img_q.size(1), "img_freqs sequence must match img sequence");
    TORCH_CHECK(txt_freqs.size(0) == txt_q.size(1), "txt_freqs sequence must match txt sequence");
    TORCH_CHECK(img_freqs.size(1) == img_q.size(3) / 2 && txt_freqs.size(1) == txt_q.size(3) / 2,
                "freqs second dim must be D/2");
    return rotary_cat_qkv_forward_cuda(img_q, img_k, img_v, txt_q, txt_k, txt_v, img_freqs, txt_freqs);
}

std::vector<torch::Tensor> modulate_forward(torch::Tensor x, torch::Tensor mod_params) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mod_params.is_cuda(), "mod_params must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mod_params.is_contiguous(), "mod_params must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, D]");
    TORCH_CHECK(mod_params.dim() == 2, "mod_params must have shape [B, 3D]");
    TORCH_CHECK(x.size(0) == mod_params.size(0), "mod_params batch must equal x batch");
    TORCH_CHECK(mod_params.size(1) == x.size(2) * 3, "mod_params last dim must be 3 * D");
    TORCH_CHECK(mod_params.scalar_type() == x.scalar_type(), "mod_params dtype must match x dtype");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    return modulate_forward_cuda(x, mod_params);
}

std::vector<torch::Tensor> modulate_indexed_forward(torch::Tensor x, torch::Tensor mod_params, torch::Tensor index) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(mod_params.is_cuda(), "mod_params must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(mod_params.is_contiguous(), "mod_params must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must have shape [B, S, D]");
    TORCH_CHECK(mod_params.dim() == 2, "mod_params must have shape [2B, 3D]");
    TORCH_CHECK(index.dim() == 3, "index must have shape [B or 1, S, 1]");
    TORCH_CHECK(index.size(1) == x.size(1), "index sequence dim must match x sequence dim");
    TORCH_CHECK(index.size(2) == 1, "index last dim must be 1");
    TORCH_CHECK(mod_params.size(0) == x.size(0) * 2, "mod_params batch must be 2 * x batch");
    TORCH_CHECK(mod_params.size(1) == x.size(2) * 3, "mod_params last dim must be 3 * D");
    TORCH_CHECK(mod_params.scalar_type() == x.scalar_type(), "mod_params dtype must match x dtype");
    TORCH_CHECK(index.scalar_type() == at::kLong || index.scalar_type() == at::kInt, "index must be int32 or int64");
    TORCH_CHECK(
        (x.scalar_type() == at::kFloat) || (x.scalar_type() == at::kHalf) || (x.scalar_type() == at::kBFloat16),
        "x dtype must be float32, float16, or bfloat16"
    );
    return modulate_indexed_forward_cuda(x, mod_params, index);
}

torch::Tensor gated_residual_forward(torch::Tensor base, torch::Tensor gate, torch::Tensor update) {
    TORCH_CHECK(base.is_cuda(), "base must be a CUDA tensor");
    TORCH_CHECK(gate.is_cuda(), "gate must be a CUDA tensor");
    TORCH_CHECK(update.is_cuda(), "update must be a CUDA tensor");
    TORCH_CHECK(base.is_contiguous(), "base must be contiguous");
    TORCH_CHECK(gate.is_contiguous(), "gate must be contiguous");
    TORCH_CHECK(update.is_contiguous(), "update must be contiguous");
    TORCH_CHECK(base.dim() == 3, "base must have shape [B, S, D]");
    TORCH_CHECK(update.sizes() == base.sizes(), "update must have same shape as base");
    TORCH_CHECK(gate.dim() == 3, "gate must have shape [B, 1 or S, D]");
    TORCH_CHECK(gate.size(0) == base.size(0), "gate batch dim must match base");
    TORCH_CHECK((gate.size(1) == 1) || (gate.size(1) == base.size(1)), "gate sequence dim must be 1 or match base");
    TORCH_CHECK(gate.size(2) == base.size(2), "gate hidden dim must match base");
    TORCH_CHECK(base.scalar_type() == update.scalar_type(), "base and update dtypes must match");
    TORCH_CHECK(base.scalar_type() == gate.scalar_type(), "base and gate dtypes must match");
    TORCH_CHECK(
        (base.scalar_type() == at::kFloat) || (base.scalar_type() == at::kHalf) || (base.scalar_type() == at::kBFloat16),
        "base dtype must be float32, float16, or bfloat16"
    );
    return gated_residual_forward_cuda(base, gate, update);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotary_emb_forward", &rotary_emb_forward, "Qwen rotary embedding forward (CUDA)");
    m.def("rotary_emb_indexed_forward", &rotary_emb_indexed_forward, "Qwen indexed rotary embedding forward (CUDA)");
    m.def("rotary_emb_pair_forward", &rotary_emb_pair_forward, "Qwen paired rotary embedding forward (CUDA)");
    m.def(
        "rotary_emb_pair_indexed_forward",
        &rotary_emb_pair_indexed_forward,
        "Qwen paired indexed rotary embedding forward (CUDA)"
    );
    m.def("rotary_cat_qkv_forward", &rotary_cat_qkv_forward, "Qwen rotary+concat qkv forward (CUDA)");
    m.def("modulate_forward", &modulate_forward, "Qwen modulation forward (CUDA)");
    m.def("modulate_indexed_forward", &modulate_indexed_forward, "Qwen indexed modulation forward (CUDA)");
    m.def("gated_residual_forward", &gated_residual_forward, "Qwen gated residual forward (CUDA)");
}
