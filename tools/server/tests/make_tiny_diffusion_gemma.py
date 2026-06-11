#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate the tiny random-weight DiffusionGemma test model used by unit/test_diffusion.py.

This builds a fully synthetic HuggingFace checkpoint (config, tokenizer and weights are all
generated here - nothing is derived from any released Gemma artifact) and converts it with the
repository's own convert_hf_to_gguf.py, so the GGUF exercises the exact conversion path
(conversion/diffusion_gemma.py) that real DiffusionGemma checkpoints take.

The model is *not* meant to produce meaningful text: the weights are seeded random numbers.
It is meant to
  - load through the DIFFUSION_GEMMA arch (canvas split, region-aware mask, MoE backbone),
  - denoise garbage-but-valid canvases deterministically for a fixed seed,
  - emit tokens from a small "answer alphabet" (NATO phonetic words) plus the channel-close
    marker "<channel|>", so the server's scaffold-extraction path is reachable,
  - never emit EOG tokens (their embedding rows are scaled to ~0), so blocks run full and
    multi-block generation is testable.

Usage (from tools/server/tests):
    python make_tiny_diffusion_gemma.py [--hf-dir DIR] [--outfile GGUF] [--seed N]

The committed asset assets/tiny-diffusion-gemma-f32.gguf was generated with the default seed.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent

CANVAS_LENGTH = 32
N_EMBD = 64
N_LAYER = 2  # one sliding_attention + one full_attention layer, to cover both mask paths
N_FF = 128
N_EXPERT = 4
N_EXPERT_USED = 2
N_FF_EXPERT = 32
N_HEAD = 4
N_HEAD_KV = 2
HEAD_DIM = 16

# control tokens (special=True -> CONTROL): never rendered, <eos>/<turn|> are EOG
CONTROL_TOKENS = [
    "<pad>", "<unk>", "<bos>", "<eos>",
    "<|turn>", "<turn|>",
    "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
]
# channel markers (special=False -> USER_DEFINED via conversion): always rendered,
# "<channel|>" is the scaffold close the server extracts the final answer after
MARKER_TOKENS = ["<|channel>", "<channel|>"]
# the "answer alphabet": with their embedding rows boosted, denoised canvases are
# pseudo-random sequences over these words (plus the channel-close marker)
ANSWER_TOKENS = [
    "▁alpha", "▁bravo", "▁charlie", "▁delta",
    "▁echo", "▁foxtrot", "▁golf", "▁hotel",
    "▁india", "▁juliet", "▁kilo", "▁lima",
    "▁mike", "▁november", "▁oscar", "▁papa",
]
# tokens produced by the toy BPE merges below
MERGED_TOKENS = ["al", "er", "on", "th"]
MERGES = [["a", "l"], ["e", "r"], ["o", "n"], ["t", "h"]]


def build_vocab() -> list[str]:
    tokens: list[str] = []
    tokens += CONTROL_TOKENS
    tokens += MARKER_TOKENS
    tokens += [f"<0x{b:02X}>" for b in range(256)]  # byte fallback
    # single characters: '▁' stands in for space (SPM-style escape), plus newline
    tokens += ["▁", "\n"]
    tokens += [chr(c) for c in range(0x21, 0x7F)]  # printable ASCII minus space
    tokens += MERGED_TOKENS
    tokens += ANSWER_TOKENS
    assert len(tokens) == len(set(tokens))
    return tokens


def write_tokenizer(hf_dir: Path, vocab: list[str]) -> None:
    added = []
    for tok in CONTROL_TOKENS:
        added.append({
            "id": vocab.index(tok), "content": tok, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False, "special": True,
        })
    for tok in MARKER_TOKENS:
        added.append({
            "id": vocab.index(tok), "content": tok, "single_word": False,
            "lstrip": False, "rstrip": False, "normalized": False, "special": False,
        })

    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
        "pre_tokenizer": None,
        "post_processor": None,
        "decoder": {
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
            ],
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<unk>",
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": True,
            "byte_fallback": True,
            "ignore_merges": False,
            "vocab": {tok: i for i, tok in enumerate(vocab)},
            "merges": MERGES,
        },
    }
    (hf_dir / "tokenizer.json").write_text(json.dumps(tokenizer_json, ensure_ascii=False, indent=1))

    # minimal turn-based chat template (written for this test model; not a Gemma template)
    chat_template = (
        "{% for message in messages %}"
        "<|turn>{{ message.role }}\n{{ message.content }}<turn|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
    )
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "chat_template": chat_template,
    }
    (hf_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=1))
    (hf_dir / "special_tokens_map.json").write_text(json.dumps({
        "bos_token": "<bos>", "eos_token": "<eos>", "unk_token": "<unk>", "pad_token": "<pad>",
    }, indent=1))


def write_config(hf_dir: Path, n_vocab: int) -> None:
    config = {
        "architectures": ["DiffusionGemmaForBlockDiffusion"],
        "model_type": "diffusion_gemma",
        "canvas_length": CANVAS_LENGTH,
        "vocab_size": n_vocab,
        "hidden_size": N_EMBD,
        "intermediate_size": N_FF,
        "num_hidden_layers": N_LAYER,
        "num_attention_heads": N_HEAD,
        "num_key_value_heads": N_HEAD_KV,
        "num_global_key_value_heads": N_HEAD_KV,
        "head_dim": HEAD_DIM,
        "global_head_dim": HEAD_DIM,
        "layer_types": ["sliding_attention", "full_attention"],
        "sliding_window": 32,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "num_local_experts": N_EXPERT,
        "num_experts_per_tok": N_EXPERT_USED,
        "expert_intermediate_size": N_FF_EXPERT,
        "partial_rotary_factor": 1.0,
        "rope_parameters": {
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
            },
            "sliding_attention": {"rope_theta": 10000.0},
        },
        "torch_dtype": "float32",
    }
    (hf_dir / "config.json").write_text(json.dumps(config, indent=1))

    # small denoiser budget so server tests stay fast; keys mirror conversion/diffusion_gemma.py
    generation_config = {
        "max_denoising_steps": 12,
        "t_min": 0.4,
        "t_max": 0.8,
        "stability_threshold": 1,
        "confidence_threshold": 0.005,
        "sampler_config": {"entropy_bound": 0.1},
    }
    (hf_dir / "generation_config.json").write_text(json.dumps(generation_config, indent=1))


def write_weights(hf_dir: Path, vocab: list[str], seed: int) -> None:
    import torch
    from safetensors.torch import save_file

    rng = np.random.default_rng(seed)

    def rand(*shape, std=0.05):
        return torch.from_numpy(rng.normal(0.0, std, size=shape).astype(np.float32))

    def ones(*shape):
        return torch.ones(*shape, dtype=torch.float32)

    n_vocab = len(vocab)
    tensors: dict[str, "torch.Tensor"] = {}

    # token embedding (tied lm_head): shape the row norms so that
    #   - answer-alphabet words and the channel-close marker dominate every argmax,
    #   - control tokens (incl. all EOG) are never produced.
    embd = rand(n_vocab, N_EMBD)
    boosted = [vocab.index(t) for t in ANSWER_TOKENS] + [vocab.index("<channel|>")]
    suppressed = [vocab.index(t) for t in CONTROL_TOKENS] + [vocab.index("<|channel>")]
    embd[boosted] *= 40.0
    embd[suppressed] *= 1e-3
    tensors["model.decoder.embed_tokens.weight"] = embd

    tensors["model.decoder.norm.weight"] = ones(N_EMBD)

    # self-conditioning gated MLP (kept small so SC feedback does not collapse the canvas)
    tensors["model.decoder.self_conditioning.pre_norm.weight"] = ones(N_EMBD)
    tensors["model.decoder.self_conditioning.gate_proj.weight"] = rand(N_FF, N_EMBD, std=0.02)
    tensors["model.decoder.self_conditioning.up_proj.weight"] = rand(N_FF, N_EMBD, std=0.02)
    tensors["model.decoder.self_conditioning.down_proj.weight"] = rand(N_EMBD, N_FF, std=0.02)

    layer_types = ["sliding_attention", "full_attention"]
    for i in range(N_LAYER):
        p = f"model.decoder.layers.{i}."
        tensors[p + "input_layernorm.weight"] = ones(N_EMBD)
        tensors[p + "self_attn.q_proj.weight"] = rand(N_HEAD * HEAD_DIM, N_EMBD, std=0.08)
        tensors[p + "self_attn.k_proj.weight"] = rand(N_HEAD_KV * HEAD_DIM, N_EMBD, std=0.08)
        if layer_types[i] == "sliding_attention":
            # global (full_attention) layers have no v_proj: V reuses k_proj
            tensors[p + "self_attn.v_proj.weight"] = rand(N_HEAD_KV * HEAD_DIM, N_EMBD, std=0.08)
        tensors[p + "self_attn.o_proj.weight"] = rand(N_EMBD, N_HEAD * HEAD_DIM, std=0.08)
        tensors[p + "self_attn.q_norm.weight"] = ones(HEAD_DIM)
        tensors[p + "self_attn.k_norm.weight"] = ones(HEAD_DIM)
        tensors[p + "post_attention_layernorm.weight"] = ones(N_EMBD)

        # dense MLP (shared expert)
        tensors[p + "pre_feedforward_layernorm.weight"] = ones(N_EMBD)
        tensors[p + "mlp.gate_proj.weight"] = rand(N_FF, N_EMBD, std=0.08)
        tensors[p + "mlp.up_proj.weight"] = rand(N_FF, N_EMBD, std=0.08)
        tensors[p + "mlp.down_proj.weight"] = rand(N_EMBD, N_FF, std=0.08)
        tensors[p + "post_feedforward_layernorm.weight"] = ones(N_EMBD)
        tensors[p + "post_feedforward_layernorm_1.weight"] = ones(N_EMBD)
        tensors[p + "pre_feedforward_layernorm_2.weight"] = ones(N_EMBD)
        tensors[p + "post_feedforward_layernorm_2.weight"] = ones(N_EMBD)

        # MoE router + fused gate_up experts
        tensors[p + "router.proj.weight"] = rand(N_EXPERT, N_EMBD, std=0.5)
        tensors[p + "router.scale"] = ones(N_EMBD)
        tensors[p + "experts.gate_up_proj"] = rand(N_EXPERT, 2 * N_FF_EXPERT, N_EMBD, std=0.08)
        tensors[p + "experts.down_proj"] = rand(N_EXPERT, N_EMBD, N_FF_EXPERT, std=0.08)

        # region-aware per-layer scalars (decoder = canvas, encoder = prompt)
        tensors[p + "layer_scalar"] = ones(1)
        tensors[f"model.encoder.layers.{i}.layer_scalar"] = ones(1)

    save_file(tensors, str(hf_dir / "model.safetensors"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-dir", type=Path, default=HERE / "tmp" / "tiny-diffusion-gemma-hf")
    ap.add_argument("--outfile", type=Path, default=HERE / "assets" / "tiny-diffusion-gemma-f32.gguf")
    ap.add_argument("--seed", type=int, default=20260611)
    args = ap.parse_args()

    args.hf_dir.mkdir(parents=True, exist_ok=True)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab()
    write_tokenizer(args.hf_dir, vocab)
    write_config(args.hf_dir, len(vocab))
    write_weights(args.hf_dir, vocab, args.seed)
    print(f"synthetic HF checkpoint written to {args.hf_dir} (vocab = {len(vocab)})")

    cmd = [
        sys.executable, str(REPO_ROOT / "convert_hf_to_gguf.py"), str(args.hf_dir),
        "--outfile", str(args.outfile), "--outtype", "f32",
    ]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"GGUF written to {args.outfile} ({args.outfile.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
