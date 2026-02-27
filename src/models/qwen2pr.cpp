#include "models.h"

#include "llama-memory-recurrent.h"

llm_build_qwen2pr::llm_build_qwen2pr(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const auto * mctx_cur = static_cast<const llama_memory_recurrent_context *>(mctx);

    const int64_t n_embd_head = hparams.n_embd_head_v;
    const int64_t head_size   = hparams.wkv_head_size;
    const int64_t n_seqs      = ubatch.n_seqs;

    GGML_ASSERT(n_embd_head == head_size);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    auto * rs_inp = build_rs_inp();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // power retention attention
        {
            // Q: [n_embd, T] -> [head_dim, n_heads, T]
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            }

            // K: [n_embd, T] -> [head_dim, n_kv_heads, T]
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur", il);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            }

            // V: same layout as K
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            if (model.layers[il].attn_q_norm) {
                Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            }
            if (model.layers[il].attn_k_norm) {
                Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            }

            // Compute per-KV-head gate logits: g = wg @ x (raw logits)
            // wg: [n_embd, n_kv_heads], cur: [n_embd, T]
            // result: [n_kv_heads, T] — kernel applies sigmoid internally
            ggml_tensor * gcur = ggml_mul_mat(ctx0, model.layers[il].wg, cur);
            cb(gcur, "gcur", il);

            // Load recurrent state for this layer: [n_embd_s, n_seqs]
            ggml_tensor * wkv_state = build_rs(rs_inp, mctx_cur->get_s_l(il), hparams.n_embd_s(), n_seqs);

            // Power retention op
            // K/V/Q must be contiguous
            Kcur = ggml_cont(ctx0, Kcur);
            Vcur = ggml_cont(ctx0, Vcur);
            Qcur = ggml_cont(ctx0, Qcur);
            gcur = ggml_cont(ctx0, gcur);

            ggml_tensor * pr_out = ggml_power_retention(ctx0, Kcur, Vcur, Qcur, gcur, wkv_state);

            // Extract attention output: first n_embd * n_tokens floats
            cur = ggml_view_2d(ctx0, pr_out, n_embd, n_tokens,
                               n_embd * ggml_element_size(pr_out), 0);
            cb(cur, "pr_attn_out", il);

            // Extract updated state and write back
            const int64_t n_embd_s = hparams.n_embd_s();
            ggml_tensor * new_state = ggml_view_1d(ctx0, pr_out,
                                                   n_embd_s * n_seqs,
                                                   n_embd * n_tokens * ggml_element_size(pr_out));

            const int32_t kv_head = mctx_cur->get_head();
            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                         ggml_view_1d(ctx0, mctx_cur->get_s_l(il),
                                      n_embd_s * n_seqs,
                                      (int64_t)n_embd_s * kv_head * ggml_element_size(mctx_cur->get_s_l(il)))));

            // Project output
            cur = build_lora_mm(model.layers[il].wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // feed-forward network
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    if (model.output_b) {
        cur = ggml_add(ctx0, cur, model.output_b);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
