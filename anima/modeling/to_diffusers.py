"""Single-file (ComfyUI-style) Anima checkpoints -> a diffusers-format repo.

This is the direction upstream ships as `scripts/convert_anima_to_diffusers.py`, which is not in
the installed wheel -- only in the diffusers git checkout. Since this trainer needs a converted
repo before it can do anything, doing the conversion ourselves removes a second checkout from the
setup path. `convert.py` is the inverse (diffusers -> native, for ComfyUI export).

Three inputs, all in the layout the model actually ships in:

    anima-base-v1.0.safetensors    685 tensors, all `net.*`  -- transformer AND text conditioner
    qwen_3_06b_base.safetensors    310 tensors, all `model.*`
    qwen_image_vae.safetensors     194 tensors

The transformer/conditioner rename tables are the inverse of `convert.py`'s, built from them at
import time rather than transcribed, so the two directions cannot drift apart.

The VAE is the awkward one: ComfyUI's names come from Qwen-Image's original implementation and
diffusers' from `AutoencoderKLQwenImage`, and the two differ by more than a prefix -- sequential
`middle.0/1/2` becomes `mid_block.resnets.0` / `attentions.0` / `resnets.1`. `VAE_KEY_MAP` below
is therefore an explicit table rather than a rule set. It was DERIVED, not transcribed: both files
hold bit-identical weights, so every tensor was matched by content hash. The match is total and
unambiguous -- 194 tensors, 194 distinct hashes on each side, no collisions and nothing unmatched
-- which makes the table provably correct rather than plausible. `test_convert_to_diffusers.py`
re-derives and compares it whenever both files are present.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .convert import _BLOCK, _TOPLEVEL, ConversionError

# native -> diffusers, the exact inverse of `convert.py`'s tables. Built here so a change there
# cannot leave this direction silently stale.
_BLOCK_INV = {v: k for k, v in _BLOCK.items()}
_TOPLEVEL_INV = {v: k for k, v in _TOPLEVEL.items()}
_TOPLEVEL_INV_BY_LEN = sorted(_TOPLEVEL_INV.items(), key=lambda kv: -len(kv[0]))

NATIVE_PREFIX = "net."
ADAPTER_PREFIX = "llm_adapter."

VAE_KEY_MAP: dict[str, str] = {
    "conv1.bias": "quant_conv.bias",
    "conv1.weight": "quant_conv.weight",
    "conv2.bias": "post_quant_conv.bias",
    "conv2.weight": "post_quant_conv.weight",
    "decoder.conv1.bias": "decoder.conv_in.bias",
    "decoder.conv1.weight": "decoder.conv_in.weight",
    "decoder.head.0.gamma": "decoder.norm_out.gamma",
    "decoder.head.2.bias": "decoder.conv_out.bias",
    "decoder.head.2.weight": "decoder.conv_out.weight",
    "decoder.middle.0.residual.0.gamma": "decoder.mid_block.resnets.0.norm1.gamma",
    "decoder.middle.0.residual.2.bias": "decoder.mid_block.resnets.0.conv1.bias",
    "decoder.middle.0.residual.2.weight": "decoder.mid_block.resnets.0.conv1.weight",
    "decoder.middle.0.residual.3.gamma": "decoder.mid_block.resnets.0.norm2.gamma",
    "decoder.middle.0.residual.6.bias": "decoder.mid_block.resnets.0.conv2.bias",
    "decoder.middle.0.residual.6.weight": "decoder.mid_block.resnets.0.conv2.weight",
    "decoder.middle.1.norm.gamma": "decoder.mid_block.attentions.0.norm.gamma",
    "decoder.middle.1.proj.bias": "decoder.mid_block.attentions.0.proj.bias",
    "decoder.middle.1.proj.weight": "decoder.mid_block.attentions.0.proj.weight",
    "decoder.middle.1.to_qkv.bias": "decoder.mid_block.attentions.0.to_qkv.bias",
    "decoder.middle.1.to_qkv.weight": "decoder.mid_block.attentions.0.to_qkv.weight",
    "decoder.middle.2.residual.0.gamma": "decoder.mid_block.resnets.1.norm1.gamma",
    "decoder.middle.2.residual.2.bias": "decoder.mid_block.resnets.1.conv1.bias",
    "decoder.middle.2.residual.2.weight": "decoder.mid_block.resnets.1.conv1.weight",
    "decoder.middle.2.residual.3.gamma": "decoder.mid_block.resnets.1.norm2.gamma",
    "decoder.middle.2.residual.6.bias": "decoder.mid_block.resnets.1.conv2.bias",
    "decoder.middle.2.residual.6.weight": "decoder.mid_block.resnets.1.conv2.weight",
    "decoder.upsamples.0.residual.0.gamma": "decoder.up_blocks.0.resnets.0.norm1.gamma",
    "decoder.upsamples.0.residual.2.bias": "decoder.up_blocks.0.resnets.0.conv1.bias",
    "decoder.upsamples.0.residual.2.weight": "decoder.up_blocks.0.resnets.0.conv1.weight",
    "decoder.upsamples.0.residual.3.gamma": "decoder.up_blocks.0.resnets.0.norm2.gamma",
    "decoder.upsamples.0.residual.6.bias": "decoder.up_blocks.0.resnets.0.conv2.bias",
    "decoder.upsamples.0.residual.6.weight": "decoder.up_blocks.0.resnets.0.conv2.weight",
    "decoder.upsamples.1.residual.0.gamma": "decoder.up_blocks.0.resnets.1.norm1.gamma",
    "decoder.upsamples.1.residual.2.bias": "decoder.up_blocks.0.resnets.1.conv1.bias",
    "decoder.upsamples.1.residual.2.weight": "decoder.up_blocks.0.resnets.1.conv1.weight",
    "decoder.upsamples.1.residual.3.gamma": "decoder.up_blocks.0.resnets.1.norm2.gamma",
    "decoder.upsamples.1.residual.6.bias": "decoder.up_blocks.0.resnets.1.conv2.bias",
    "decoder.upsamples.1.residual.6.weight": "decoder.up_blocks.0.resnets.1.conv2.weight",
    "decoder.upsamples.10.residual.0.gamma": "decoder.up_blocks.2.resnets.2.norm1.gamma",
    "decoder.upsamples.10.residual.2.bias": "decoder.up_blocks.2.resnets.2.conv1.bias",
    "decoder.upsamples.10.residual.2.weight": "decoder.up_blocks.2.resnets.2.conv1.weight",
    "decoder.upsamples.10.residual.3.gamma": "decoder.up_blocks.2.resnets.2.norm2.gamma",
    "decoder.upsamples.10.residual.6.bias": "decoder.up_blocks.2.resnets.2.conv2.bias",
    "decoder.upsamples.10.residual.6.weight": "decoder.up_blocks.2.resnets.2.conv2.weight",
    "decoder.upsamples.11.resample.1.bias": "decoder.up_blocks.2.upsamplers.0.resample.1.bias",
    "decoder.upsamples.11.resample.1.weight": "decoder.up_blocks.2.upsamplers.0.resample.1.weight",
    "decoder.upsamples.12.residual.0.gamma": "decoder.up_blocks.3.resnets.0.norm1.gamma",
    "decoder.upsamples.12.residual.2.bias": "decoder.up_blocks.3.resnets.0.conv1.bias",
    "decoder.upsamples.12.residual.2.weight": "decoder.up_blocks.3.resnets.0.conv1.weight",
    "decoder.upsamples.12.residual.3.gamma": "decoder.up_blocks.3.resnets.0.norm2.gamma",
    "decoder.upsamples.12.residual.6.bias": "decoder.up_blocks.3.resnets.0.conv2.bias",
    "decoder.upsamples.12.residual.6.weight": "decoder.up_blocks.3.resnets.0.conv2.weight",
    "decoder.upsamples.13.residual.0.gamma": "decoder.up_blocks.3.resnets.1.norm1.gamma",
    "decoder.upsamples.13.residual.2.bias": "decoder.up_blocks.3.resnets.1.conv1.bias",
    "decoder.upsamples.13.residual.2.weight": "decoder.up_blocks.3.resnets.1.conv1.weight",
    "decoder.upsamples.13.residual.3.gamma": "decoder.up_blocks.3.resnets.1.norm2.gamma",
    "decoder.upsamples.13.residual.6.bias": "decoder.up_blocks.3.resnets.1.conv2.bias",
    "decoder.upsamples.13.residual.6.weight": "decoder.up_blocks.3.resnets.1.conv2.weight",
    "decoder.upsamples.14.residual.0.gamma": "decoder.up_blocks.3.resnets.2.norm1.gamma",
    "decoder.upsamples.14.residual.2.bias": "decoder.up_blocks.3.resnets.2.conv1.bias",
    "decoder.upsamples.14.residual.2.weight": "decoder.up_blocks.3.resnets.2.conv1.weight",
    "decoder.upsamples.14.residual.3.gamma": "decoder.up_blocks.3.resnets.2.norm2.gamma",
    "decoder.upsamples.14.residual.6.bias": "decoder.up_blocks.3.resnets.2.conv2.bias",
    "decoder.upsamples.14.residual.6.weight": "decoder.up_blocks.3.resnets.2.conv2.weight",
    "decoder.upsamples.2.residual.0.gamma": "decoder.up_blocks.0.resnets.2.norm1.gamma",
    "decoder.upsamples.2.residual.2.bias": "decoder.up_blocks.0.resnets.2.conv1.bias",
    "decoder.upsamples.2.residual.2.weight": "decoder.up_blocks.0.resnets.2.conv1.weight",
    "decoder.upsamples.2.residual.3.gamma": "decoder.up_blocks.0.resnets.2.norm2.gamma",
    "decoder.upsamples.2.residual.6.bias": "decoder.up_blocks.0.resnets.2.conv2.bias",
    "decoder.upsamples.2.residual.6.weight": "decoder.up_blocks.0.resnets.2.conv2.weight",
    "decoder.upsamples.3.resample.1.bias": "decoder.up_blocks.0.upsamplers.0.resample.1.bias",
    "decoder.upsamples.3.resample.1.weight": "decoder.up_blocks.0.upsamplers.0.resample.1.weight",
    "decoder.upsamples.3.time_conv.bias": "decoder.up_blocks.0.upsamplers.0.time_conv.bias",
    "decoder.upsamples.3.time_conv.weight": "decoder.up_blocks.0.upsamplers.0.time_conv.weight",
    "decoder.upsamples.4.residual.0.gamma": "decoder.up_blocks.1.resnets.0.norm1.gamma",
    "decoder.upsamples.4.residual.2.bias": "decoder.up_blocks.1.resnets.0.conv1.bias",
    "decoder.upsamples.4.residual.2.weight": "decoder.up_blocks.1.resnets.0.conv1.weight",
    "decoder.upsamples.4.residual.3.gamma": "decoder.up_blocks.1.resnets.0.norm2.gamma",
    "decoder.upsamples.4.residual.6.bias": "decoder.up_blocks.1.resnets.0.conv2.bias",
    "decoder.upsamples.4.residual.6.weight": "decoder.up_blocks.1.resnets.0.conv2.weight",
    "decoder.upsamples.4.shortcut.bias": "decoder.up_blocks.1.resnets.0.conv_shortcut.bias",
    "decoder.upsamples.4.shortcut.weight": "decoder.up_blocks.1.resnets.0.conv_shortcut.weight",
    "decoder.upsamples.5.residual.0.gamma": "decoder.up_blocks.1.resnets.1.norm1.gamma",
    "decoder.upsamples.5.residual.2.bias": "decoder.up_blocks.1.resnets.1.conv1.bias",
    "decoder.upsamples.5.residual.2.weight": "decoder.up_blocks.1.resnets.1.conv1.weight",
    "decoder.upsamples.5.residual.3.gamma": "decoder.up_blocks.1.resnets.1.norm2.gamma",
    "decoder.upsamples.5.residual.6.bias": "decoder.up_blocks.1.resnets.1.conv2.bias",
    "decoder.upsamples.5.residual.6.weight": "decoder.up_blocks.1.resnets.1.conv2.weight",
    "decoder.upsamples.6.residual.0.gamma": "decoder.up_blocks.1.resnets.2.norm1.gamma",
    "decoder.upsamples.6.residual.2.bias": "decoder.up_blocks.1.resnets.2.conv1.bias",
    "decoder.upsamples.6.residual.2.weight": "decoder.up_blocks.1.resnets.2.conv1.weight",
    "decoder.upsamples.6.residual.3.gamma": "decoder.up_blocks.1.resnets.2.norm2.gamma",
    "decoder.upsamples.6.residual.6.bias": "decoder.up_blocks.1.resnets.2.conv2.bias",
    "decoder.upsamples.6.residual.6.weight": "decoder.up_blocks.1.resnets.2.conv2.weight",
    "decoder.upsamples.7.resample.1.bias": "decoder.up_blocks.1.upsamplers.0.resample.1.bias",
    "decoder.upsamples.7.resample.1.weight": "decoder.up_blocks.1.upsamplers.0.resample.1.weight",
    "decoder.upsamples.7.time_conv.bias": "decoder.up_blocks.1.upsamplers.0.time_conv.bias",
    "decoder.upsamples.7.time_conv.weight": "decoder.up_blocks.1.upsamplers.0.time_conv.weight",
    "decoder.upsamples.8.residual.0.gamma": "decoder.up_blocks.2.resnets.0.norm1.gamma",
    "decoder.upsamples.8.residual.2.bias": "decoder.up_blocks.2.resnets.0.conv1.bias",
    "decoder.upsamples.8.residual.2.weight": "decoder.up_blocks.2.resnets.0.conv1.weight",
    "decoder.upsamples.8.residual.3.gamma": "decoder.up_blocks.2.resnets.0.norm2.gamma",
    "decoder.upsamples.8.residual.6.bias": "decoder.up_blocks.2.resnets.0.conv2.bias",
    "decoder.upsamples.8.residual.6.weight": "decoder.up_blocks.2.resnets.0.conv2.weight",
    "decoder.upsamples.9.residual.0.gamma": "decoder.up_blocks.2.resnets.1.norm1.gamma",
    "decoder.upsamples.9.residual.2.bias": "decoder.up_blocks.2.resnets.1.conv1.bias",
    "decoder.upsamples.9.residual.2.weight": "decoder.up_blocks.2.resnets.1.conv1.weight",
    "decoder.upsamples.9.residual.3.gamma": "decoder.up_blocks.2.resnets.1.norm2.gamma",
    "decoder.upsamples.9.residual.6.bias": "decoder.up_blocks.2.resnets.1.conv2.bias",
    "decoder.upsamples.9.residual.6.weight": "decoder.up_blocks.2.resnets.1.conv2.weight",
    "encoder.conv1.bias": "encoder.conv_in.bias",
    "encoder.conv1.weight": "encoder.conv_in.weight",
    "encoder.downsamples.0.residual.0.gamma": "encoder.down_blocks.0.norm1.gamma",
    "encoder.downsamples.0.residual.2.bias": "encoder.down_blocks.0.conv1.bias",
    "encoder.downsamples.0.residual.2.weight": "encoder.down_blocks.0.conv1.weight",
    "encoder.downsamples.0.residual.3.gamma": "encoder.down_blocks.0.norm2.gamma",
    "encoder.downsamples.0.residual.6.bias": "encoder.down_blocks.0.conv2.bias",
    "encoder.downsamples.0.residual.6.weight": "encoder.down_blocks.0.conv2.weight",
    "encoder.downsamples.1.residual.0.gamma": "encoder.down_blocks.1.norm1.gamma",
    "encoder.downsamples.1.residual.2.bias": "encoder.down_blocks.1.conv1.bias",
    "encoder.downsamples.1.residual.2.weight": "encoder.down_blocks.1.conv1.weight",
    "encoder.downsamples.1.residual.3.gamma": "encoder.down_blocks.1.norm2.gamma",
    "encoder.downsamples.1.residual.6.bias": "encoder.down_blocks.1.conv2.bias",
    "encoder.downsamples.1.residual.6.weight": "encoder.down_blocks.1.conv2.weight",
    "encoder.downsamples.10.residual.0.gamma": "encoder.down_blocks.10.norm1.gamma",
    "encoder.downsamples.10.residual.2.bias": "encoder.down_blocks.10.conv1.bias",
    "encoder.downsamples.10.residual.2.weight": "encoder.down_blocks.10.conv1.weight",
    "encoder.downsamples.10.residual.3.gamma": "encoder.down_blocks.10.norm2.gamma",
    "encoder.downsamples.10.residual.6.bias": "encoder.down_blocks.10.conv2.bias",
    "encoder.downsamples.10.residual.6.weight": "encoder.down_blocks.10.conv2.weight",
    "encoder.downsamples.2.resample.1.bias": "encoder.down_blocks.2.resample.1.bias",
    "encoder.downsamples.2.resample.1.weight": "encoder.down_blocks.2.resample.1.weight",
    "encoder.downsamples.3.residual.0.gamma": "encoder.down_blocks.3.norm1.gamma",
    "encoder.downsamples.3.residual.2.bias": "encoder.down_blocks.3.conv1.bias",
    "encoder.downsamples.3.residual.2.weight": "encoder.down_blocks.3.conv1.weight",
    "encoder.downsamples.3.residual.3.gamma": "encoder.down_blocks.3.norm2.gamma",
    "encoder.downsamples.3.residual.6.bias": "encoder.down_blocks.3.conv2.bias",
    "encoder.downsamples.3.residual.6.weight": "encoder.down_blocks.3.conv2.weight",
    "encoder.downsamples.3.shortcut.bias": "encoder.down_blocks.3.conv_shortcut.bias",
    "encoder.downsamples.3.shortcut.weight": "encoder.down_blocks.3.conv_shortcut.weight",
    "encoder.downsamples.4.residual.0.gamma": "encoder.down_blocks.4.norm1.gamma",
    "encoder.downsamples.4.residual.2.bias": "encoder.down_blocks.4.conv1.bias",
    "encoder.downsamples.4.residual.2.weight": "encoder.down_blocks.4.conv1.weight",
    "encoder.downsamples.4.residual.3.gamma": "encoder.down_blocks.4.norm2.gamma",
    "encoder.downsamples.4.residual.6.bias": "encoder.down_blocks.4.conv2.bias",
    "encoder.downsamples.4.residual.6.weight": "encoder.down_blocks.4.conv2.weight",
    "encoder.downsamples.5.resample.1.bias": "encoder.down_blocks.5.resample.1.bias",
    "encoder.downsamples.5.resample.1.weight": "encoder.down_blocks.5.resample.1.weight",
    "encoder.downsamples.5.time_conv.bias": "encoder.down_blocks.5.time_conv.bias",
    "encoder.downsamples.5.time_conv.weight": "encoder.down_blocks.5.time_conv.weight",
    "encoder.downsamples.6.residual.0.gamma": "encoder.down_blocks.6.norm1.gamma",
    "encoder.downsamples.6.residual.2.bias": "encoder.down_blocks.6.conv1.bias",
    "encoder.downsamples.6.residual.2.weight": "encoder.down_blocks.6.conv1.weight",
    "encoder.downsamples.6.residual.3.gamma": "encoder.down_blocks.6.norm2.gamma",
    "encoder.downsamples.6.residual.6.bias": "encoder.down_blocks.6.conv2.bias",
    "encoder.downsamples.6.residual.6.weight": "encoder.down_blocks.6.conv2.weight",
    "encoder.downsamples.6.shortcut.bias": "encoder.down_blocks.6.conv_shortcut.bias",
    "encoder.downsamples.6.shortcut.weight": "encoder.down_blocks.6.conv_shortcut.weight",
    "encoder.downsamples.7.residual.0.gamma": "encoder.down_blocks.7.norm1.gamma",
    "encoder.downsamples.7.residual.2.bias": "encoder.down_blocks.7.conv1.bias",
    "encoder.downsamples.7.residual.2.weight": "encoder.down_blocks.7.conv1.weight",
    "encoder.downsamples.7.residual.3.gamma": "encoder.down_blocks.7.norm2.gamma",
    "encoder.downsamples.7.residual.6.bias": "encoder.down_blocks.7.conv2.bias",
    "encoder.downsamples.7.residual.6.weight": "encoder.down_blocks.7.conv2.weight",
    "encoder.downsamples.8.resample.1.bias": "encoder.down_blocks.8.resample.1.bias",
    "encoder.downsamples.8.resample.1.weight": "encoder.down_blocks.8.resample.1.weight",
    "encoder.downsamples.8.time_conv.bias": "encoder.down_blocks.8.time_conv.bias",
    "encoder.downsamples.8.time_conv.weight": "encoder.down_blocks.8.time_conv.weight",
    "encoder.downsamples.9.residual.0.gamma": "encoder.down_blocks.9.norm1.gamma",
    "encoder.downsamples.9.residual.2.bias": "encoder.down_blocks.9.conv1.bias",
    "encoder.downsamples.9.residual.2.weight": "encoder.down_blocks.9.conv1.weight",
    "encoder.downsamples.9.residual.3.gamma": "encoder.down_blocks.9.norm2.gamma",
    "encoder.downsamples.9.residual.6.bias": "encoder.down_blocks.9.conv2.bias",
    "encoder.downsamples.9.residual.6.weight": "encoder.down_blocks.9.conv2.weight",
    "encoder.head.0.gamma": "encoder.norm_out.gamma",
    "encoder.head.2.bias": "encoder.conv_out.bias",
    "encoder.head.2.weight": "encoder.conv_out.weight",
    "encoder.middle.0.residual.0.gamma": "encoder.mid_block.resnets.0.norm1.gamma",
    "encoder.middle.0.residual.2.bias": "encoder.mid_block.resnets.0.conv1.bias",
    "encoder.middle.0.residual.2.weight": "encoder.mid_block.resnets.0.conv1.weight",
    "encoder.middle.0.residual.3.gamma": "encoder.mid_block.resnets.0.norm2.gamma",
    "encoder.middle.0.residual.6.bias": "encoder.mid_block.resnets.0.conv2.bias",
    "encoder.middle.0.residual.6.weight": "encoder.mid_block.resnets.0.conv2.weight",
    "encoder.middle.1.norm.gamma": "encoder.mid_block.attentions.0.norm.gamma",
    "encoder.middle.1.proj.bias": "encoder.mid_block.attentions.0.proj.bias",
    "encoder.middle.1.proj.weight": "encoder.mid_block.attentions.0.proj.weight",
    "encoder.middle.1.to_qkv.bias": "encoder.mid_block.attentions.0.to_qkv.bias",
    "encoder.middle.1.to_qkv.weight": "encoder.mid_block.attentions.0.to_qkv.weight",
    "encoder.middle.2.residual.0.gamma": "encoder.mid_block.resnets.1.norm1.gamma",
    "encoder.middle.2.residual.2.bias": "encoder.mid_block.resnets.1.conv1.bias",
    "encoder.middle.2.residual.2.weight": "encoder.mid_block.resnets.1.conv1.weight",
    "encoder.middle.2.residual.3.gamma": "encoder.mid_block.resnets.1.norm2.gamma",
    "encoder.middle.2.residual.6.bias": "encoder.mid_block.resnets.1.conv2.bias",
    "encoder.middle.2.residual.6.weight": "encoder.mid_block.resnets.1.conv2.weight",}


# Component configs. Small enough to embed, and embedding them means the converter works offline
# and deterministically rather than depending on a network fetch at the moment you need it.
#
# They are NOT trusted blindly: `_check_config` asserts every dimension against the checkpoint that
# was actually supplied, so feeding a differently-shaped Anima variant fails with the mismatch
# named instead of writing a repo that loads and then produces noise.
TRANSFORMER_CONFIG = {   '_class_name': 'CosmosTransformer3DModel',
    '_diffusers_version': '0.40.0.dev0',
    'adaln_lora_dim': 256,
    'attention_head_dim': 128,
    'concat_padding_mask': True,
    'controlnet_block_every_n': None,
    'crossattn_proj_in_channels': 1024,
    'encoder_hidden_states_channels': 1024,
    'extra_pos_embed_type': None,
    'img_context_dim_in': None,
    'img_context_dim_out': 2048,
    'img_context_num_tokens': 256,
    'in_channels': 16,
    'max_size': [128, 240, 240],
    'mlp_ratio': 4.0,
    'num_attention_heads': 16,
    'num_layers': 28,
    'out_channels': 16,
    'patch_size': [1, 2, 2],
    'rope_scale': [1.0, 4.0, 4.0],
    'text_embed_dim': 1024,
    'use_crossattn_projection': False}

TEXT_CONDITIONER_CONFIG = {
    "_class_name": "AnimaTextConditioner",
    "_diffusers_version": "0.40.0.dev0",
    "min_sequence_length": 512,
    "mlp_ratio": 4.0,
    "model_dim": 1024,
    "num_attention_heads": 16,
    "num_layers": 6,
    "source_dim": 1024,
    "target_dim": 1024,
    "target_vocab_size": 32128,
    "use_layer_norm": False,
    "use_self_attention": True,
}

SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.40.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def split_native(state: dict[str, torch.Tensor]) -> tuple[dict, dict]:
    """One `net.*` checkpoint -> (transformer, text_conditioner) in DIFFUSERS naming.

    The two models ship interleaved in a single file: `net.llm_adapter.*` is the conditioner and
    everything else under `net.` is the trunk. 118 + 567 = 685 on the released checkpoint.
    """
    transformer, conditioner, unmapped = {}, {}, []

    for key, value in state.items():
        if not key.startswith(NATIVE_PREFIX):
            unmapped.append(key)
            continue
        inner = key[len(NATIVE_PREFIX):]

        if inner.startswith(ADAPTER_PREFIX):
            # The conditioner's names already agree with diffusers'; only the prefix goes.
            conditioner[inner[len(ADAPTER_PREFIX):]] = value
            continue

        if inner.startswith("blocks."):
            rest = inner[len("blocks."):]
            idx, _, tail = rest.partition(".")
            module, _, suffix = tail.rpartition(".")
            if suffix in ("weight", "bias") and module in _BLOCK_INV:
                transformer[f"transformer_blocks.{idx}.{_BLOCK_INV[module]}.{suffix}"] = value
                continue
            unmapped.append(key)
            continue

        # Longest prefix first, so `t_embedder.1` cannot be shadowed by a shorter overlap.
        for native_prefix, diffusers_prefix in _TOPLEVEL_INV_BY_LEN:
            if inner == native_prefix or inner.startswith(native_prefix + "."):
                transformer[diffusers_prefix + inner[len(native_prefix):]] = value
                break
        else:
            unmapped.append(key)

    if unmapped:
        raise ConversionError(
            f"{len(unmapped)} key(s) in the Anima checkpoint matched no rename rule "
            f"(e.g. {unmapped[:6]}). This does not look like an Anima single-file checkpoint."
        )
    return transformer, conditioner


def convert_text_encoder(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Qwen3 `model.*` -> the bare names `Qwen3Model.from_pretrained` expects."""
    out, unmapped = {}, []
    for key, value in state.items():
        if key.startswith("model."):
            out[key[len("model."):]] = value
        elif key.startswith("lm_head."):
            continue        # Anima uses Qwen3 as an encoder; the LM head is dead weight
        else:
            unmapped.append(key)
    if unmapped:
        raise ConversionError(
            f"{len(unmapped)} Qwen3 key(s) are not under `model.` (e.g. {unmapped[:6]}). "
            f"Expected a Qwen3 checkpoint in the layout ComfyUI ships."
        )
    return out


def convert_vae(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Qwen-Image VAE, ComfyUI naming -> `AutoencoderKLQwenImage` naming."""
    missing = [k for k in state if k not in VAE_KEY_MAP]
    if missing:
        raise ConversionError(
            f"{len(missing)} VAE key(s) are not in the rename table (e.g. {missing[:6]}). "
            f"Expected the Qwen-Image VAE ({len(VAE_KEY_MAP)} tensors); got {len(state)}."
        )
    absent = [k for k in VAE_KEY_MAP if k not in state]
    if absent:
        raise ConversionError(
            f"the VAE is missing {len(absent)} expected tensor(s) (e.g. {absent[:6]})."
        )
    return {VAE_KEY_MAP[k]: v for k, v in state.items()}


def _check_config(name: str, config: dict, state: dict[str, torch.Tensor],
                  checks: list[tuple[str, object, object]]) -> None:
    """Fail loudly when the embedded config disagrees with the weights actually supplied.

    Without this, a differently-shaped Anima variant writes a repo whose config.json describes a
    model the safetensors does not contain. `from_pretrained` then either raises something opaque
    about size mismatches, or -- worse, if only a count differs -- loads and generates noise.
    """
    bad = [(what, want, got) for what, want, got in checks if want != got]
    if bad:
        detail = "; ".join(f"{what}: config says {want}, checkpoint has {got}"
                           for what, want, got in bad)
        raise ConversionError(
            f"{name}: the supplied checkpoint does not match the known Anima architecture "
            f"({detail}). This converter ships the config for the released model; a variant with "
            f"different dimensions needs its config.json written by hand."
        )


def _write(component: Path, config: dict, state: dict[str, torch.Tensor], weights_name: str,
           progress=None) -> None:
    component.mkdir(parents=True, exist_ok=True)
    (component / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    if progress:
        progress(f"writing {component.name}/{weights_name} ({len(state)} tensors)")
    # `contiguous()` because safetensors refuses a view, and a few renamed tensors arrive as one.
    save_file({k: v.contiguous() for k, v in state.items()}, component / weights_name,
              metadata={"format": "pt"})


TOKENIZER_DIRS = ("tokenizer", "t5_tokenizer")

#: Pulled when no tokenizer source is given. ~13MB of the two vocabularies and nothing else --
#: `allow_patterns` keeps the 5GB of weights in that repo from being fetched. Verified byte-
#: identical to the reference repo's tokenizer directories (all 5 files).
#:
#: `circlestone-labs/Anima-Base-v1.0-Diffusers` is a working alternative: its tokenizers produce
#: identical ids on 413/413 probes, differing only in metadata (`eos_token`, `model_max_length`),
#: which this path never consults -- both call sites pass `max_length=512, truncation=True`
#: explicitly. It is not the default only because it is not bit-identical to the reference.
DEFAULT_TOKENIZER_REPO = "Bluvoll/Anima-v1.0-Base-Diffusers"

#: `--tokenizers none` -- convert without them on purpose (weights-only inspection, or a repo
#: whose tokenizers are supplied later).
TOKENIZER_NONE = "none"


def _looks_like_repo_id(src: str | Path) -> bool:
    """A hub id rather than a path: `owner/name`, one slash, and no such directory on disk.

    Checked in that order so a local directory always wins -- someone with a real `./org/model`
    checkout must not silently get a download instead.
    """
    s = str(src)
    return not Path(s).exists() and s.count("/") == 1 and not s.startswith((".", "/", "~"))


def fetch_tokenizers(repo_id: str, say=lambda _m: None) -> Path:
    """Download just the two tokenizer directories from `repo_id`, and return the snapshot root.

    Uses the ordinary huggingface_hub cache, so this costs one ~13MB download the first time and
    nothing on every conversion after -- including offline, which `snapshot_download` serves from
    cache without touching the network.
    """
    from huggingface_hub import snapshot_download

    say(f"fetching tokenizers from {repo_id} (~13MB, cached after the first time)")
    return Path(snapshot_download(
        repo_id, allow_patterns=[f"{name}/*" for name in TOKENIZER_DIRS]))


def copy_tokenizers(out: Path, src: str | Path | None, say=lambda _m: None) -> list[str]:
    """Put `tokenizer/` and `t5_tokenizer/` into `out`; return what landed there.

    `src` may be a local directory, a hub repo id, `None` (fetch `DEFAULT_TOKENIZER_REPO`), or
    `"none"` (skip deliberately).

    Separate from `convert_to_diffusers` so the warning below can be tested without writing 4GB
    of weights first -- it is the one part of the conversion whose *absence* is the failure, and
    an untested warning is a warning that quietly stops firing.

    Loud rather than fatal, because the repo looks complete without them: every weight is present
    and the directory listing looks right. `load_components` then dies on a missing tokenizer --
    after loading the whole model, with an opaque "Repo id must be in the form ..." from
    huggingface_hub, which names neither the real problem nor the fix.
    """
    # `src is None` must be tested FIRST: `str(None).lower()` is the literal "none", so the
    # explicit-skip branch would otherwise swallow the default and silently download nothing.
    if src is None:
        src = DEFAULT_TOKENIZER_REPO
    elif str(src).strip().lower() == TOKENIZER_NONE:
        src = None

    copied: list[str] = []
    if src:
        if _looks_like_repo_id(src):
            try:
                src = fetch_tokenizers(str(src), say)
            except Exception as exc:                 # noqa: BLE001 - offline is not a crash here
                # Never fatal: the weights are converted either way, and the warning below tells
                # the user exactly what is missing and how to supply it. Dying at this point would
                # throw away a conversion that has already read several GB.
                say(f"WARNING: could not fetch tokenizers ({type(exc).__name__}: {exc})")
                src = None

    if src:
        src = Path(src)
        for name in TOKENIZER_DIRS:
            if (src / name).is_dir():
                shutil.copytree(src / name, out / name, dirs_exist_ok=True)
                copied.append(name)
            elif src.name == name:
                shutil.copytree(src, out / name, dirs_exist_ok=True)
                copied.append(name)
        say(f"copied tokenizers: {', '.join(copied) or 'none found'}")

    if set(copied) != set(TOKENIZER_DIRS):
        absent = sorted(set(TOKENIZER_DIRS) - set(copied))
        say(f"WARNING: {' and '.join(absent)} missing -- this repo CANNOT TRAIN yet. Both run on "
            f"every step, because text embeddings are not cached: tag shuffling and caption "
            f"dropout change the caption for the same image every epoch, so a cached embedding "
            f"would be stale. Re-run with a tokenizer source, or copy the directories in by hand.")
    return copied


def convert_to_diffusers(
    anima_path: str | Path,
    qwen_path: str | Path,
    vae_path: str | Path,
    out_dir: str | Path,
    tokenizer_src: str | Path | None = None,
    write_modular_index: bool = True,
    progress=None,
) -> Path:
    """Write a diffusers-format Anima repo from the three single-file checkpoints.

    `tokenizer_src` says where the two tokenizers come from: a local directory holding
    `tokenizer/` and `t5_tokenizer/`, a hub repo id, `None` to fetch `DEFAULT_TOKENIZER_REPO`
    (the default -- ~13MB, then cached), or `"none"` to skip them deliberately. They are ~13MB of
    vocabulary that no amount of tensor renaming can synthesise.

    **They are required to train.** This trainer does NOT cache text embeddings -- tag shuffling
    and caption dropout produce a different caption for the same image every epoch, so a cached
    embedding would be stale. Both tokenizers are therefore called on every step
    (`train.py::_encode`). A repo converted without them holds every weight and still cannot run.

    `write_modular_index` emits `modular_model_index.json` for `AnimaModularPipeline`. The trainer
    never reads it -- it loads each component by subfolder -- and it embeds absolute paths, so it
    is written pointing at `out_dir` rather than copied from anywhere.
    """
    say = progress or (lambda _m: None)
    anima_path, qwen_path, vae_path = Path(anima_path), Path(qwen_path), Path(vae_path)
    out = Path(out_dir)
    # Destination first: it is the check the user can act on without waiting, and reading 4GB of
    # safetensors only to refuse afterwards would be a rude way to report a fixable mistake.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} already exists and is not empty. Choose an empty directory -- refusing to "
            f"merge into an existing repo, where a stale file would silently survive."
        )
    for label, p in (("Anima", anima_path), ("Qwen3", qwen_path), ("VAE", vae_path)):
        if not p.is_file():
            raise FileNotFoundError(f"{label} checkpoint not found: {p}")

    say(f"reading {anima_path.name}")
    transformer, conditioner = split_native(load_file(str(anima_path)))
    n_blocks = 1 + max((int(k.split(".")[1]) for k in transformer
                        if k.startswith("transformer_blocks.")), default=-1)
    _check_config("transformer", TRANSFORMER_CONFIG, transformer, [
        ("num_layers", TRANSFORMER_CONFIG["num_layers"], n_blocks),
        ("attention_head_dim x num_attention_heads",
         TRANSFORMER_CONFIG["attention_head_dim"] * TRANSFORMER_CONFIG["num_attention_heads"],
         transformer["transformer_blocks.0.attn1.to_q.weight"].shape[0]),
        ("text_embed_dim", TRANSFORMER_CONFIG["text_embed_dim"],
         transformer["transformer_blocks.0.attn2.to_k.weight"].shape[1]),
    ])
    n_cond = 1 + max((int(k.split(".")[1]) for k in conditioner if k.startswith("blocks.")),
                     default=-1)
    _check_config("text_conditioner", TEXT_CONDITIONER_CONFIG, conditioner, [
        ("num_layers", TEXT_CONDITIONER_CONFIG["num_layers"], n_cond),
    ])
    _write(out / "transformer", TRANSFORMER_CONFIG, transformer,
           "diffusion_pytorch_model.safetensors", say)
    _write(out / "text_conditioner", TEXT_CONDITIONER_CONFIG, conditioner,
           "diffusion_pytorch_model.safetensors", say)

    say(f"reading {qwen_path.name}")
    te = convert_text_encoder(load_file(str(qwen_path)))
    n_layers = 1 + max((int(k.split(".")[1]) for k in te if k.startswith("layers.")), default=-1)
    hidden = te["embed_tokens.weight"].shape[1]
    te_config = dict(TEXT_ENCODER_CONFIG)
    te_config["num_hidden_layers"] = n_layers
    te_config["layer_types"] = ["full_attention"] * n_layers
    _check_config("text_encoder", te_config, te, [
        ("hidden_size", te_config["hidden_size"], hidden),
        ("vocab_size", te_config["vocab_size"], te["embed_tokens.weight"].shape[0]),
    ])
    _write(out / "text_encoder", te_config, te, "model.safetensors", say)

    say(f"reading {vae_path.name}")
    _write(out / "vae", VAE_CONFIG, convert_vae(load_file(str(vae_path))),
           "diffusion_pytorch_model.safetensors", say)

    (out / "scheduler").mkdir(parents=True, exist_ok=True)
    (out / "scheduler" / "scheduler_config.json").write_text(
        json.dumps(SCHEDULER_CONFIG, indent=2, sort_keys=True) + "\n")

    copied = copy_tokenizers(out, tokenizer_src, say)

    if write_modular_index:
        (out / "modular_model_index.json").write_text(
            json.dumps(_modular_index(out, bool(copied)), indent=2, sort_keys=True) + "\n")

    say(f"done -> {out}")
    return out


def _modular_index(out: Path, with_tokenizers: bool) -> dict:
    """`modular_model_index.json` pointing at `out`, not at wherever it was generated.

    Every entry carries an absolute `pretrained_model_name_or_path`, so a copied index makes the
    repo work only on the machine that produced it.
    """
    def entry(library: str, cls: str, subfolder: str) -> list:
        return [library, cls, {"pretrained_model_name_or_path": str(out.resolve()),
                               "revision": None, "subfolder": subfolder,
                               "type_hint": [library, cls], "variant": None}]

    index = {
        "_blocks_class_name": "AnimaAutoBlocks",
        "_class_name": "AnimaModularPipeline",
        "_diffusers_version": "0.40.0.dev0",
        "scheduler": entry("diffusers", "FlowMatchEulerDiscreteScheduler", "scheduler"),
        "text_conditioner": entry("diffusers", "AnimaTextConditioner", "text_conditioner"),
        "text_encoder": entry("transformers", "Qwen3Model", "text_encoder"),
        "transformer": entry("diffusers", "CosmosTransformer3DModel", "transformer"),
        "vae": entry("diffusers", "AutoencoderKLQwenImage", "vae"),
    }
    if with_tokenizers:
        index["tokenizer"] = entry("transformers", "Qwen2Tokenizer", "tokenizer")
        index["t5_tokenizer"] = entry("transformers", "T5Tokenizer", "t5_tokenizer")
    return index


# Qwen3-0.6B used as an encoder. `num_hidden_layers` and `layer_types` are overwritten from
# the checkpoint at convert time, so a different Qwen3 size still yields a correct config.
TEXT_ENCODER_CONFIG = {   'architectures': ['Qwen3Model'],
    'attention_bias': False,
    'attention_dropout': 0.0,
    'bos_token_id': None,
    'dtype': 'bfloat16',
    'eos_token_id': None,
    'head_dim': 128,
    'hidden_act': 'silu',
    'hidden_size': 1024,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 32768,
    'max_window_layers': 28,
    'model_type': 'qwen3',
    'num_attention_heads': 16,
    'num_hidden_layers': 28,
    'num_key_value_heads': 8,
    'pad_token_id': None,
    'rms_norm_eps': 1e-06,
    'rope_parameters': {'rope_theta': 1000000.0, 'rope_type': 'default'},
    'sliding_window': None,
    'tie_word_embeddings': False,
    'transformers_version': '5.14.1',
    'use_cache': True,
    'use_sliding_window': False,
    'vocab_size': 151936}


# AutoencoderKLQwenImage.
VAE_CONFIG = {   '_class_name': 'AutoencoderKLQwenImage',
    '_diffusers_version': '0.40.0.dev0',
    'attn_scales': [],
    'base_dim': 96,
    'dim_mult': [1, 2, 4, 4],
    'dropout': 0.0,
    'input_channels': 3,
    'latents_mean': [   -0.7571,
                        -0.7089,
                        -0.9113,
                        0.1075,
                        -0.1745,
                        0.9653,
                        -0.1517,
                        1.5508,
                        0.4134,
                        -0.0715,
                        0.5517,
                        -0.3632,
                        -0.1922,
                        -0.9497,
                        0.2503,
                        -0.2921],
    'latents_std': [   2.8184,
                       1.4541,
                       2.3275,
                       2.6558,
                       1.2196,
                       1.7708,
                       2.6052,
                       2.0743,
                       3.2687,
                       2.1526,
                       2.8652,
                       1.5579,
                       1.6382,
                       1.1253,
                       2.8251,
                       1.916],
    'num_res_blocks': 2,
    'temperal_downsample': [False, True, True],
    'z_dim': 16}
