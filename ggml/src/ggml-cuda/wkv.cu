#include "common.cuh"
#include "wkv.cuh"

template <int block_size>
static __global__ void rwkv_wkv_f32(const int B, const int T, const int C, const int H, const float * k, const float * v, const float * r, const float * tf, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    __syncthreads();
    _tf[tid] = tf[head_i * head_size + tid];
    __syncthreads();

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            y += r.x * (tf.x * kv.x + s.x);
            y += r.y * (tf.y * kv.y + s.y);
            y += r.z * (tf.z * kv.z + s.z);
            y += r.w * (tf.w * kv.w + s.w);

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

template <int block_size>
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

#ifndef GGML_USE_MUSA
    #pragma unroll
#endif
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& a = (float4&)(_a[j]);
            const float4& s = (float4&)(state[j]);
            sa += a.x * s.x;
            sa += a.y * s.y;
            sa += a.z * s.z;
            sa += a.w * s.w;
        }

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& b = (float4&)(_b[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * w.x + kv.x + sa * b.x;
            s.y = s.y * w.y + kv.y + sa * b.y;
            s.z = s.z * w.z + kv.z + sa * b.z;
            s.w = s.w * w.w + kv.w + sa * b.w;

            y += s.x * r.x;
            y += s.y * r.y;
            y += s.z * r.z;
            y += s.w * r.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

// Power Retention kernel:
// Each block handles one (batch, kv_head) pair. Thread tid handles row tid of the S×S state.
// k,v: [S, H_kv, T]  q: [S, H, T]  g: [H_kv, T]  state_in: [S*S*H_kv, n_seqs]
// output: [S*H*T + S*S*H_kv*n_seqs] (attention output | updated state)
// Compute expanded phi dimension for power retention (degree=2, IB=16, OB=8)
static __host__ __device__ int64_t power_retention_expanded_dim(int64_t head_dim) {
    const int64_t IB = 16, OB = 8;
    return ((IB/OB + head_dim/OB) * (head_dim/IB) / 2) * (IB * OB);
}

// Cooperative phi computation: each thread writes elements where flat_idx % S == tid.
// For key phi: IB=16, OB=8, off-diagonal blocks get 2x multiplier.
// For query phi: IB=16, OB=1, no multiplier.
// x_src: input vector in shared memory [S], phi_out: output in shared memory [D]
static __device__ void compute_phi_cooperative(
        const float * x_src, float * phi_out, int S, int D, int tid,
        int OB, bool is_key) {
    const int IB = 16;
    const int n_inner = S / IB;
    const int n_outer = S / OB;
    int flat_idx = 0;

    for (int y_idx = 0; y_idx < n_inner; y_idx++) {
        const int x_end_val = (y_idx + 1) * IB / OB;
        const int x_end = x_end_val < n_outer ? x_end_val : n_outer;
        for (int x_idx = 0; x_idx < x_end; x_idx++) {
            const float mult = (is_key && !((x_idx + 1) * OB > y_idx * IB)) ? 2.0f : 1.0f;
            for (int a = 0; a < OB; a++) {
                const float xa = mult * x_src[x_idx * OB + a];
                for (int bb = 0; bb < IB; bb++) {
                    if (flat_idx % S == tid) {
                        phi_out[flat_idx] = xa * x_src[y_idx * IB + bb];
                    }
                    flat_idx++;
                }
            }
        }
    }
}

// Block-level reduction to compute sum across all threads
static __device__ float block_reduce_sum(float val, int tid, int block_size, float * warp_buf) {
    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;
    if (lane == 0) warp_buf[warp_id] = val;
    __syncthreads();
    float result = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < (block_size + warpSize - 1) / warpSize; w++) {
            result += warp_buf[w];
        }
        warp_buf[0] = result;
    }
    __syncthreads();
    return warp_buf[0];
}

// Power Retention CUDA kernel with symmetric power feature map.
// Each block handles one (batch, kv_head). S threads per block, each owns column tid.
// State [D, S] + normalizer [D] live in global memory.
static __global__ void power_retention_f32(
        const int B, const int T, const int S, const int H_kv, const int H,
        const int D, const int64_t state_per_head,
        const float * k, const float * v, const float * q, const float * g,
        const float * state_in, float * dst) {
    const int tid    = threadIdx.x;
    const int bid    = blockIdx.x;
    const int b      = bid / H_kv;
    const int h_kv   = bid % H_kv;
    const int groups = H / H_kv;
    const int n_seq_tokens = T / B;

    float * out_attn  = dst;
    float * out_state = dst + (int64_t)S * H * T;

    const int64_t state_offset = (int64_t)b * state_per_head * H_kv + (int64_t)h_kv * state_per_head;
    float * S_state = out_state + state_offset;
    float * s_norm  = out_state + state_offset + (int64_t)D * S;

    // Copy initial state
    const float * s_in_base = state_in + state_offset;
    for (int i = 0; i < D; i++) {
        S_state[i * S + tid] = s_in_base[i * S + tid];
    }
    for (int i = tid; i < D; i += S) {
        s_norm[i] = s_in_base[D * S + i];
    }
    __syncthreads();

    // Dynamic shared memory layout: phi[D] + vec[S] + warp_buf[32]
    extern __shared__ float shmem[];
    float * sh_phi    = shmem;
    float * sh_vec    = shmem + D;       // reused for k, v, q
    float * warp_buf  = shmem + D + S;   // [32]

    for (int t = b * n_seq_tokens; t < (b + 1) * n_seq_tokens; t++) {
        const float g_raw = g[t * H_kv + h_kv];
        const float decay = 1.0f / (1.0f + expf(-g_raw));

        // Load k into shared memory and compute phi_key
        __syncthreads();
        sh_vec[tid] = k[t * H_kv * S + h_kv * S + tid];
        __syncthreads();

        compute_phi_cooperative(sh_vec, sh_phi, S, D, tid, 8, true);
        __syncthreads();

        // Load v[tid]
        const float v_tid = v[t * H_kv * S + h_kv * S + tid];

        // State update: S_state[i, tid] = decay * S_state[i, tid] + phi_k[i] * v[tid]
        for (int i = 0; i < D; i++) {
            S_state[i * S + tid] = decay * S_state[i * S + tid] + sh_phi[i] * v_tid;
        }
        // Normalizer update (distributed)
        for (int i = tid; i < D; i += S) {
            s_norm[i] = decay * s_norm[i] + sh_phi[i];
        }
        __syncthreads();

        // Query output for each group
        for (int gi = 0; gi < groups; gi++) {
            const int h_q = h_kv * groups + gi;

            // Load q into shared mem and compute phi_query
            __syncthreads();
            sh_vec[tid] = q[t * H * S + h_q * S + tid];
            __syncthreads();

            compute_phi_cooperative(sh_vec, sh_phi, S, D, tid, 1, false);
            __syncthreads();

            // Denominator: sum_i(phi_q[i] * s_norm[i])
            float denom_partial = 0.0f;
            for (int i = tid; i < D; i += S) {
                denom_partial += sh_phi[i] * s_norm[i];
            }
            float denom = block_reduce_sum(denom_partial, tid, S, warp_buf);
            if (fabsf(denom) < 1e-12f) denom = 1e-12f;
            const float inv_denom = 1.0f / denom;

            // Numerator for column tid: sum_i(phi_q[i] * S_state[i, tid])
            float num = 0.0f;
            for (int i = 0; i < D; i++) {
                num += sh_phi[i] * S_state[i * S + tid];
            }
            out_attn[t * H * S + h_q * S + tid] = num * inv_denom;
        }
    }
}

void ggml_cuda_op_power_retention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d = (const float *)dst->src[0]->data;
    const float * v_d = (const float *)dst->src[1]->data;
    const float * q_d = (const float *)dst->src[2]->data;
    const float * g_d = (const float *)dst->src[3]->data;
    const float * s_d = (const float *)dst->src[4]->data;

    const int64_t S    = dst->src[0]->ne[0];
    const int64_t H_kv = dst->src[0]->ne[1];
    const int64_t T    = dst->src[0]->ne[2];
    const int64_t H    = dst->src[2]->ne[1];
    const int64_t B    = dst->src[4]->ne[1];
    const int64_t D    = power_retention_expanded_dim(S);
    const int64_t state_per_head = D * (S + 1);

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(S % 16 == 0);
    GGML_ASSERT(H % H_kv == 0);

    // Dynamic shared memory: phi[D] + vec[S] + warp_buf[32]
    const size_t shmem_size = (D + S + 32) * sizeof(float);

    power_retention_f32<<<B * H_kv, S, shmem_size, stream>>>(
        B, T, S, H_kv, H, D, state_per_head, k_d, v_d, q_d, g_d, s_d, dst_d);
}

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * tf_d = (const float *)dst->src[3]->data;
    const float * td_d = (const float *)dst->src[4]->data;
    const float * s_d  = (const float *)dst->src[5]->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    } else {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    }
}

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * r_d = (const float *)dst->src[0]->data;
    const float * w_d = (const float *)dst->src[1]->data;
    const float * k_d = (const float *)dst->src[2]->data;
    const float * v_d = (const float *)dst->src[3]->data;
    const float * a_d = (const float *)dst->src[4]->data;
    const float * b_d = (const float *)dst->src[5]->data;
    const float * s_d = (const float *)dst->src[6]->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d);
    }
}
