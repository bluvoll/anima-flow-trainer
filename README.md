# Anima trainer

Diffusers-native trainer for Anima (Cosmos-Predict2 DiT + LLMAdapter), built for my own 2×4090, by using SDNQ the vram usage in FullFinetune can be brought down to ~10GB.
Full finetune and LoRA/LoKr, sd-scripts bucketing, SDNQ quantization.

Everything runs from the repo root using its own venv. **Never use `python`/`pip` directly** —
always `venv/bin/python`, so a system or neighbouring environment cannot leak in, or activate the venv first to use local python / pip.

---

## 0. Install

```bash
./install.sh          # Linux
install.bat           # Windows
```

Both find a Python **3.11** or **3.12**, create `venv/`, install everything, and write two
launchers: `start-gui.sh` / `start-gui.bat` for the trainer GUI, and `start-converter.sh` /
`start-converter.bat` for the single-file → diffusers converter (§0b). Each forwards its arguments,
so a bare double-click opens the window while flags still reach the CLI. `--recreate` starts from
a clean venv.

| | |
|---|---|
| **3.11** | what every measurement below was taken on; the installer picks it when both are present |
| **3.12** | the whole graph resolves (77 packages, torch ships cp312 wheels for linux x86_64 and win_amd64) but nothing has been *run* on it — the installer prefers 3.11 when both are present and says so when it falls back |
| **3.13** | excluded: torch has wheels, SDNQ and triton are untested and the lock has never been resolved against it |

If `uv` is on PATH the installer syncs from `uv.lock`, which reproduces the exact resolved set. If
not it falls back to pip, which needs three things in order that a plain `pip install -e .` gets
wrong: torch from the cu128 index (PyPI's default build is CUDA 12.6) and diffusers from **git**
(Anima support exists only on main — a release build imports fine and then fails at model load).

Afterwards it imports every dependency and reports CUDA. Re-run it any time:

```bash
venv/bin/python -m anima.tools.check_install
```

---

## 0b. Convert the model

The trainer reads a **diffusers-format repo**, but Anima ships as single files. Convert once:

```bash
./start-converter.sh                                      # or start-converter.bat on Windows
venv/bin/python -m anima.tools.convert_model --gui        # the same window, without the launcher
```

or headlessly:

```bash
venv/bin/python -m anima.tools.convert_model \
    --anima  anima-base-v1.0.safetensors \
    --qwen   qwen_3_06b_base.safetensors \
    --vae    qwen_image_vae.safetensors \
    --out    /path/to/anima-diffusers
```

<img width="944" height="678" alt="imagen" src="https://github.com/user-attachments/assets/f6c159b3-bb0c-47bf-a169-bb768f2d99ac" />


The three inputs are only ever read. The output directory must be empty or absent — it refuses to
merge into an existing repo, where a stale file would silently survive a re-convert.

| input | contents |
|---|---|
| Anima checkpoint | 685 tensors under `net.*` — the transformer **and** the text conditioner in one file |
| Qwen3 0.6B | 310 tensors under `model.*`; the LM head, if present, is dropped since Anima uses Qwen3 as an encoder |
| Qwen-Image VAE | 194 tensors in ComfyUI's naming |

**Tokenizers are the one thing conversion cannot produce, and they are required to train** — so
they are fetched for you. `tokenizer/` and `t5_tokenizer/` are ~13 MB of vocabulary that no amount
of tensor renaming synthesises, so by default the converter downloads just those two directories
from `Bluvoll/Anima-v1.0-Base-Diffusers` (`allow_patterns` keeps the 5 GB of weights in that repo
out of it). They land in the ordinary Hugging Face cache, so it costs one 13 MB download the first
time and nothing — including offline — on every conversion after.

`--tokenizers` overrides that with a local directory or a different hub repo id; a directory on
disk always wins over a repo-id reading. `--tokenizers none` skips them deliberately, and an
unreachable repo warns rather than throwing away a conversion that has already read several GB.

Why it matters: this trainer does **not** cache text embeddings — tag shuffling and caption dropout
produce a different caption for the same image every epoch, so a cached embedding would be stale
(§3). Both tokenizers therefore run on *every step* (`train.py::_encode`). A repo converted without
them holds every weight, looks complete in a directory listing, and still cannot run — it dies
inside `load_components`, *after* loading all 4 GB, with huggingface_hub's opaque `Repo id must be
in the form ...`. The converter says so up front rather than letting you find out there.

`circlestone-labs/Anima-Base-v1.0-Diffusers` works equally well as a source: its tokenizers give
identical ids on 413/413 probes, differing only in metadata (`eos_token`, `model_max_length`) that
this path never consults — both call sites pass `max_length=512, truncation=True` explicitly.

`modular_model_index.json` is written by default for the diffusers inference pipeline and can be
skipped with `--no-modular-index`. The trainer never reads it; it loads each component by
subfolder. Note it embeds absolute paths, which is why it is generated pointing at your output
directory rather than copied from anywhere.

### How the mapping is known to be right

The transformer and conditioner tables are built at import time as the **inverse** of the
diffusers→native tables in `convert.py`, so the two directions cannot drift apart. The gate asserts
`native → diffusers → native` is the identity, which a one-entry error in either table breaks.

The VAE is different: ComfyUI's names come from Qwen-Image's original implementation, and they
differ from diffusers' by more than a prefix — sequential `middle.0/1/2` becomes
`mid_block.resnets.0` / `attentions.0` / `resnets.1`. Rules for that would reimplement module
flattening and encode assumptions about block counts, so `VAE_KEY_MAP` is an explicit 194-entry
table. It was **derived, not transcribed**: both files hold bit-identical weights, so every tensor
was matched by content hash — 194 tensors, 194 distinct hashes per side, no collisions and nothing
unmatched. `test_convert_to_diffusers.py` re-derives it from the real weights when they are
present, so the table is checked against the model rather than against someone's reading of it.

Measured on the released checkpoint: **1189/1189 tensors bit-identical** to a reference repo
produced by upstream's own script, and all four `config.json` files identical.

---

### How the GUI looks

<img width="1843" height="1185" alt="imagen" src="https://github.com/user-attachments/assets/7c311edf-bae2-4413-9c0c-1faf6533f238" />



## 1. Cache latents (required first)

<img width="769" height="154" alt="imagen" src="https://github.com/user-attachments/assets/b9b32bd8-5d37-44ec-a0f4-a4389b3b643b" />



Training reads latents, never images. Caching also freezes bucket assignment.

```bash
CUDA_VISIBLE_DEVICES=1 venv/bin/python -m anima.tools.cache_latents cache \
    /path/to/dataset --resolution 1024
```

Prints the bucket distribution before writing anything, and refuses to start if images lack a
`.txt` caption (override with `--allow-missing-captions`). Re-running skips what already exists;
`--overwrite` forces.

**`--resolution` is an AREA budget, not a side length.** `1024` means "about 1024×1024 worth of
pixels", so 1920×1080 becomes 1344×768. Only `--max-bucket-reso` is a per-side cap.

Add `--dry-run` to see the bucket plan and the cache size it would write, without writing it.

### Multi-resolution: several area budgets in one run

Pass several values to cache every image at every tier, then list the same tiers in the config:

```bash
venv/bin/python -m anima.tools.cache_latents cache \
    /path/to/dataset --resolution 768 1024 1280 --dry-run
```
```toml
[dataset]
resolutions = [768, 1024, 1280]     # replaces `resolution`; setting both is an error
```

<img width="1258" height="266" alt="imagen" src="https://github.com/user-attachments/assets/f7f7e034-e786-434a-b40a-39f713fb6541" />



**A tier is a repeat at a different resolution.** N tiers cost N× the epoch and N× the cache — the
model simply sees each image at N sizes instead of one. Nothing else changes: batches stay
bucket-homogeneous, step counting is unchanged, and DDP is unaffected.

**Choosing the ladder is the whole game, and it is a measurement, not a guess.** Because buckets
never upscale, a tier above an image's own native area produces the *same* bucket as the tier below
it — so a ladder sitting above your dataset's size distribution silently degrades into plain
`num_repeats` while costing the same VRAM, disk and wall-clock. `audit` reports the distribution and
proposes a ladder:

```bash
venv/bin/python -m anima.tools.cache_latents audit /path/to/dataset
```
```
source-area distribution (2000 sources)
   pct     MP    tier at this source's own ceiling
   p1      0.55     704px
   p10     1.24    1088px
   p50     5.00    2176px

   suggested ladder: resolutions = [640, 832, 1088]
   -> 190/2000 sources (9.5%) sit below the top rung
```
<img width="1103" height="423" alt="imagen" src="https://github.com/user-attachments/assets/c82c8f46-c391-4788-afa1-a62d8b292d7e" />




Rule of thumb: **bottom rung below p1–p5** so every tier genuinely shrinks something; the **gap
between the top rung and p10 is your collapse rate**. On the mixed set, `[640, 896, 1280]` collapses ~1%
where `[768, 1024, 1280]` collapses 8.2%, at identical cost.

Whatever collapses is reported, and warned about at startup:

```
tiers    2000 sources x 3 tier(s)
           768:   2000 entries
          1024:   1967 entries  (33 collapsed onto a lower tier)
          1280:   1836 entries  (164 collapsed onto a lower tier)
```

| key | default | notes |
|---|---|---|
| `resolutions` | unset | list of area budgets; unset = single tier at `resolution` |
| `tier_collapse` | `"dedup"` | `"dedup"` drops duplicate (image, bucket) pairs; `"repeat"` keeps them so every image gets exactly N repeats |
| `min_source_area` | `0.0` | drop sources below this fraction of the *smallest* tier's budget |

**Why `dedup` is the default.** Under `"repeat"`, a small image's repeat factor silently equals the
ladder length — add a 1536 rung and every collapsed image gains 4/3 weight while gaining no new
resolution. Nobody chooses that. `dedup` keeps the two decisions separate: the ladder controls how
many resolutions, `num_repeats` controls how much weight. Measured costs are small either way
(dedup: the 33 fully-collapsed images get 1 entry not 3; repeat: 2.05% of batches contain the
same image twice), so this is a low-stakes knob — the coupling is what decides it.

**`min_source_area` exists because Anima has no size micro-conditioning.** `CosmosTransformer3DModel`
takes no `original_size` / `crops_coords`, unlike SDXL, which fed exactly those into the timestep
embedding so the model could tell "this source was small" from "this concept has little detail".
Anima can't: the only resolution signal it gets is the RoPE grid. So a 256×384 source supplies
384-token supervision the model cannot attribute to source size. Dedup stops it being counted three
times; it does not make it useful data for a 1280px finetune. Setting `min_source_area = 1.0` drops
anything that cannot even fill the smallest tier.

**Do not reach for upscaling.** It is the only thing that would give small sources more pixels, but
LANCZOS-upscaling a 0.5 MP illustration to 1.6 MP teaches the model to emit soft detail at high
resolution — directly against the high-frequency token loss below.

### Maximum resolution: 1920 per side in spec, 2048 hard

Two different ceilings, both **per side**, neither an area limit:

| per side | patch units | result |
|---|---|---|
| 1920 | 120 | OK — the model's declared spatial maximum |
| 1984 / 2048 | 124 / 128 | runs, but spatial RoPE is **extrapolating** |
| 2112+ | 132+ | `RuntimeError` — hard failure |

The transformer config declares `max_size = [128, 240, 240]` (t, h, w), which becomes
`[128, 120, 120]` patch units — so **120 units × 16 px = 1920 px** is the configured spatial range.
But `CosmosRotaryPosEmbed.forward` builds its position sequence as `arange(max(self.max_size))`,
and that max is **128, from the temporal axis**. The spatial dims index into that longer sequence,
so they reach 2048 px before anything breaks.

This is why diffusion-pipe asserts at 1920 while diffusers works to 2048 — one follows the declared
limit, the other an implementation artifact. `verify_max_resolution()` errors above 2048 and warns
between 1920 and 2048. **Set `max_bucket_reso = 1920` to stay in spec.**

Note the area cost: a 1920×1088 bucket is ~2.09M pixels, so it needs `resolution ≈ 1440` to be
reachable at all — and at 8160 tokens it is in the same cost class as 1536².

**`max_bucket_reso` (default 1920) is enforced in both bucketing branches — shrink only.** The
area budget bounds `w*h` but neither side individually, so a wide source can sit under `max_area`
and still exceed the RoPE limit: at `resolution = 1536` a 21:9 image lands at 2346px. sd-scripts
has no such limit to respect and its no-upscale branch ignores `max_size`; we downscale uniformly
until the longest side fits.

This never upscales. A 512×512 image stays 512×512 regardless of `resolution`; only sources
*larger* than the cap are shrunk:

```
 512x512   ->  512x512     (untouched)
1920x1080  -> 1920x1024
3840x2160  -> 1920x1024    (downscaled to fit)
6000x800   -> 1920x256     (capped on the long side)
```

Because the widest side scales as `sqrt(area x aspect_ratio)`, high `resolution` plus wide sources
is what pushes into the cap. Largest side produced for a source above the area cap:

| resolution | 4:3 | 16:10 | 16:9 | 21:9 |
|---|---|---|---|---|
| 1280 | 1478 | 1619 | 1707 | 1955 |
| **1440** | 1663 | 1821 | **1920** | 2200 |
| 1536 | 1774 | 1943 | 2048 | 2346 |

Anything above 1920 in that table gets shrunk to fit rather than rejected, which costs area on
wide images — at `resolution = 1536` a 21:9 source ends up at ~0.86MP instead of 2.36MP.

Measured on a mixed-resolution set (2000 images, AR 0.31–3.20), `bucket_no_upscale = true`:

| config | widest | out-of-spec | hard-fail | upscaled | buckets |
|---|---|---|---|---|---|
| res 1024, max 1920 | 1792 | 0 | 0 | 0 | 92 |
| res 1440, max 1920 | 1920 | 0 | 0 | 0 | 186 |
| **res 1536, max 1920** | **1920** | **0** | **0** | **0** | 204 |

(Before the per-side cap, res 1536 produced 47 hard failures and the only workaround was
`bucket_no_upscale = false`, which upscaled 502 of 2000 images.)

Caches are written next to the images as `<stem>_WWWWxHHHH_anima.safetensors` (sd-scripts
convention). **Changing `--resolution`, `--bucket-reso-steps`, or `--upscale` invalidates them** —
the trainer will tell you rather than train on the wrong thing.

### Deleting images afterwards

```bash
venv/bin/python -m anima.tools.cache_latents audit /path/to/dataset
```

Only says `SAFE` when every image is cached *and* every cache has a caption. Caches are ~13×
smaller than the images. After deleting, training keeps working with no config change
(`source = "auto"` detects it).

### Training without a cache: `source = "encode"`

```toml
[dataset]
source = "encode"    # bucket from images, run the VAE every step
```

Normally the trainer never loads the VAE at all — latents are cached, so recomputing them every
epoch is pure waste, and `source = "images"` **hard-refuses to start** if any (image, bucket) pair
has no cache. `"encode"` is the exception: it buckets exactly like `"images"` but yields pixels and
encodes in the training loop.

It exists for modes where the crop is chosen *per step* and so cannot be precomputed — random-crop
augmentation, mirroring, and the texture curriculum. **For ordinary training it is strictly worse**
and you should cache.

Measured on a texture set (413 images, tier 1024, one 4090, identical batch schedule):

| | cached | encode |
|---|---|---|
| peak, batch 12 | 17.8 GB | 18.2 GB |
| peak, batch 8 | 13.6 GB | 13.9 GB |
| mean step, batch 12 | 5.61 s/it | 6.52 s/it |

So the price is **+0.3–0.4 GB and +16% step time**, and *no reduction in batch size* — batch 12
runs in both modes. The memory cost is nearly all the VAE's resident weights, because the encode
runs under `no_grad` and its activations are freed before the transformer forward allocates, making
the step peak `max(train_peak, encode_peak)` rather than their sum.

That is also why `train.vae_encode_chunk` exists and why it should stay at **1** — it decides which
of those two terms wins. Raising it buys no throughput at all (chunk 1/2/6 → 6.52/6.52/6.53 s/it)
while peak climbs 18.2 → 19.3 → 23.1 GB, and chunk 12 OOMs.

Encoding on the fly is numerically the same as caching, not merely similar: both paths go through
`LatentCacher.encode_tensor`, so the crop is bit-identical and the only difference is the VAE
posterior draw — measured at mean abs 1.41e-05 against a same-input noise floor of 1.40e-05.
One consequence worth knowing: that draw is *not* a meaningful source of augmentation. Re-encoding
the same image reproduces the cached latent to 5 decimal places, so "a fresh posterior sample each
epoch" buys nothing. Crop and flip variation is the only real argument for encoding live.

---

## 2. Train

```bash
# single GPU
CUDA_VISIBLE_DEVICES=1 venv/bin/python -m anima.training.train configs/lora.toml

# both GPUs
venv/bin/accelerate launch --num_processes 2 -m anima.training.train configs/full.toml

# or the GUI, which drives both of the above
venv/bin/python -m anima.gui
```

Both LoRA and full finetune work under DDP. Verified: after 6 steps on 2 ranks fed different data,
adapter checksums are **bit-identical** across ranks, so the gradient all-reduce is real.

**Sharding is Accelerate's job, not ours.** `accelerator.prepare(dataloader)` wraps the batch
sampler in `BatchSamplerShard`, which hands whole batches to each rank (preserving bucket
homogeneity) and pads with `even_batches` to keep ranks in lockstep. `BucketBatchSampler` is
deliberately rank-agnostic — sharding there too would shard twice and each rank would see roughly
`1/world_size²` of the data, visible only as a quietly reduced step count.

A trailing partial accumulation group still steps: Accelerate forces a sync at `end_of_dataloader`
(`Accelerator._do_sync`), which is the same behaviour sd-scripts relies on. **No batches are
dropped.** So `steps/epoch = ceil(ceil(batches / num_processes) / accum)`.

Worked example — 113 images, micro-batch 12, accum 2, 2 GPUs:

```
113 images -> 10 batches -> 5 per rank -> ceil(5/2) = 3 optimizer steps/epoch
images per optimizer step = 12 x 2 x 2 = 48   (last step of each epoch: 24, a partial group)
```

| run | per-rank peak | s/it | throughput vs 1 GPU |
|---|---|---|---|
| LoRA, 1 GPU | 7.1 GB | 1.56 | 1.00× |
| LoRA, 2 GPUs | 7.2 GB | 1.72 | 1.81× |
| full FT, 1 GPU | 14.2 GB | 2.19 | 1.00× |
| full FT, 2 GPUs | **17.7 GB** | 2.78 | 1.58× |

Full FT under DDP costs ~3.5 GB/rank extra for gradient buckets.

**Ranks are symmetric** Measured: torch peak 17.7 GB on *both*
ranks; `nvidia-smi` showed GPU:0 at 20152 MiB vs GPU:1 at 18134 MiB, but the pre-run baseline was
already 2078 vs 2 MiB. Net training allocation: 18074 vs 18132 MiB — identical to within 58 MiB.
The whole gap is the desktop on GPU:0.



### Interconnect is the scaling limit (PCIe 3.0 x8 here)

Full FT all-reduces 1.76B × 2 bytes ≈ **3.5 GB per optimizer step**. Ideal 2× scaling would be
2.19 s/it; measured 2.80 s/it, so ~0.6 s goes to communication — which is what 3.5 GB over a
~6 GB/s effective link costs. LoRA scales better (1.81×) purely because it all-reduces 45.9M
params, **38× less traffic**.

**The highest-value knob on a narrow link is `gradient_accumulation_steps`.** Accelerate skips
gradient sync on non-sync micro-batches, so all-reduce fires once per *optimizer* step regardless
of accumulation — more accumulation means proportionally less PCIe traffic per image. Measured,
full FT on 2 GPUs:

| `gradient_accumulation_steps` | images/step | s/it | **s/image** |
|---|---|---|---|
| 1 | 2 | 1.39 | 0.695 |
| 4 | 8 | 2.73 | **0.341** |

**2.04× throughput for free**, with peak memory essentially unchanged (17.4 → 17.7 GB). On PCIe 3.0
x8, do not run full-FT DDP with low accumulation. Freezing helps for the same reason:
`adaln`/`base`/`llm_adapter` frozen removes 330M params (16%) from every all-reduce.

Useful flags: `--max-steps N`, `--no-save`, `--dry-run` (build everything, one step, exit).
**Always `--dry-run` a new config first** — it catches config, cache, and VRAM problems in ~40s.

Config filenames used throughout this README (`configs/lora.toml`, `configs/full.toml`, …) are
illustrative. `configs/` is yours and ships empty — section 6 is the complete key reference, and
every key has a default, so a working config is short. Unknown keys are a hard error rather than a
warning, so a typo costs a `KeyError` at load and not a wasted run.

---

## 3. The knobs that matter

<img width="1282" height="691" alt="imagen" src="https://github.com/user-attachments/assets/69ea437f-48b4-4d95-928f-2b8ca18faf5a" />


### `[adapter]` — LoRA vs full finetune

> **The default is LoRA.** If you omit `[adapter]` entirely you get a rank-32 LoRA, not a full
> finetune. For full finetuning you must write `kind = "none"`.

| key | default | notes |
|---|---|---|
| `kind` | `"lora"` | `"lora"` \| `"lokr"` \| `"none"` (= full FT) |
| `rank` / `alpha` | 32 / 32.0 | LoKr reaches similar expressivity at far fewer params |
| `components` | `["self_attn","cross_attn","mlp"]` | see the table below |
| `train_llm_adapter` | `false` | adapts the text conditioner too |
| `train_text_encoder` | `false` | LoRA on Qwen3 itself (sd-scripts' `lora_te`) |

#### On LoKr

LoKr is a LyCORIS method, but it does not need LyCORIS here: **peft absorbed LoHa/LoKr/OFT**, so
`peft.LoKrConfig` is what this trainer uses (peft 0.20). It factorises the update as a Kronecker
product, reaching a given expressivity at far fewer parameters than LoRA at the same rank — which
matters when optimizer state is the binding constraint.

Where it lands is worth knowing before you commit a run to it:

| | LoRA | LoKr |
|---|---|---|
| trains | ✅ | ✅ |
| exports to ComfyUI (`diffusion_model.*`) | ✅ | ✅ *key names verified exact* |
| round-trips through diffusers `AnimaLoraLoaderMixin` | ✅ | ❌ **no LoKr support upstream** |
| kohya `lora_unet_*` export | ✅ | ❌ rejected — `lora_down`/`lora_up` have no LoKr analogue |

peft stores each LoKr factor as a bare `nn.Parameter` named `<marker>.<adapter_name>` with no
`.weight` suffix, unlike LoRA's `lora_A.default.weight`. The export left that `.default` in place
and emitted `diffusion_model.<path>.lokr_w1.default`, while ComfyUI looks up
`diffusion_model.<path>.lokr_w1` (`comfy/weight_adapter/lokr.py:211`) — so **LoKr checkpoints used
to load nowhere at all.** Fixed, and pinned by `test_param_groups.py`, which asserts the exported
key set matches ComfyUI's lookups exactly. What is verified is key-name compatibility and the
tensor layout peft produces (`lokr_w1` + `lokr_w2_a`/`lokr_w2_b`, the factored form ComfyUI reads);
an end-to-end generate in ComfyUI has not been run.

**Which components.** Measured share of the DiT's 1956M Linear parameters:

| component | share | what it carries |
|---|---|---|
| `mlp` | 48.0% | Bulk of Information |
| `self_attn` | 24.0% | Styles, some composition, some anatomy |
| `cross_attn` | 18.0% | prompt adherence — add for characters/concepts |
| `adaln` | 9.0% | timestep modulation; destabilises easily |
| `base` | 1.0% | in/out projections; rarely worth it |


### `[component_lr]` — full finetune **and** LoRA

<img width="1249" height="354" alt="imagen" src="https://github.com/user-attachments/assets/502eccfd-bc48-4481-a630-289760dea704" />


Per-component LRs. **`0.0` means freeze**, which allocates no gradient and no optimizer state —
that is a memory decision, not just a learning one.

```toml
[component_lr]
self_attn = 5e-6
cross_attn = 1e-5
mlp = 1e-5
adaln = 0.0        # frozen
base = 0.0
llm_adapter = 0.0
```

**This applies to LoRA too.** The adapter tensors are grouped by the component of the base module
each one wraps, so the same keys work — plus `text_encoder` for a Qwen3 adapter:

```toml
[adapter]
kind = "lora"
components = ["self_attn", "cross_attn", "mlp"]
train_text_encoder = true

[component_lr]
mlp = 4.5e-5           # style
cross_attn = 1.5e-5    # prompt adherence — often wants less
text_encoder = 5e-6    # see below
```

**Set `text_encoder` explicitly if you enable `train_text_encoder`.** It is the one component where
a shared LR is actively dangerous — sd-scripts has always exposed `text_encoder_lr` separately, and
the usual guidance is well below the trunk's. Left unset it inherits `optimizer.lr`, which for a
text encoder is normally far too high.

A LR on a component with no adapter injected is now **rejected** rather than ignored. So what you set is what you can see:

```
  self_attn       14.68M trainable      0.00M frozen  lr=3.00e-05
  cross_attn      12.85M trainable      0.00M frozen  lr=1.50e-05
  mlp             18.35M trainable      0.00M frozen  lr=4.50e-05
```

### `torch.compile`

<img width="1273" height="186" alt="imagen" src="https://github.com/user-attachments/assets/8e12ccdb-f417-43e9-882a-37c5940d53f5" />


```toml
[train]
compile = "default"        # "default" | "reduce-overhead" | "max-autotune"; omit to disable
compile_dynamic = true
compile_regional = true
```

Applied through Accelerate's `TorchDynamoPlugin`, so compilation happens *under* the DDP wrapper
rather than around it. Measured on uniform a 1920×1080 set, batch 2:

| | eager | compiled | speedup | compiles | break-even |
|---|---|---|---|---|---|
| 1 bucket (1344×768) | 3.06 s/it | **2.27 s/it** | 1.35× | 1 (~67 s) | ~85 steps |
| 3 buckets (640/768/1024 tiers) | 1.88 s/it | **1.42 s/it** | 1.32× | 2 (~46 s) | ~95 steps |

Losses match eager to 4 decimal places, and peak memory is marginally *lower* (7.9 vs 8.0 GB).

#### Compilation is lazy, and does not scale with bucket count

**Nothing is compiled at startup.** A shape is compiled on the step where it is first *needed*, so
step 1 always pays, and a later step can spike when a new shape first appears. If you see a mid-run
step jump to tens of seconds, that is a compile, not a hang.

The real risk was that every bucket pays its own recompile — which would make a mixed set's 45–207 buckets
hopeless. It does not happen. Note that **`batch_size` varies per tier, so the shape varies in
both dims**, and the trailing partial batch of each bucket is its own shape again: the 3-tier
config below produces **6** distinct `(batch, bucket)` shapes, not 3.

| shapes | inductor cache | compiles | at steps | cost |
|---|---|---|---|---|
| 1 (single bucket, fixed batch) | warm | 1 | 1 | 67 s |
| 2 (2 buckets, fixed batch) | warm | 2 | 1, 9 | 35 + 37 s |
| 3 (3 buckets, fixed batch) | warm | 2 | 1, 17 | 34 + 12 s |
| **6 (3 buckets, batch varies per bucket)** | warm | **1** | 1 | 34 s |
| **6 (3 buckets, batch varies per bucket)** | **cold** | **1** | 1 | **75 s** |

Counter-intuitively, *more* shape variety produced *fewer* compiles. That is dynamo's
automatic-dynamic behaviour: when varied shapes arrive early it generalises both dimensions during
the initial compile window, instead of specialising on the first shape and recompiling when the
second arrives. So varying batch size per bucket — exactly what the per-tier `batch_size` map does
— helps here rather than hurting.

Inductor caches compiled kernels on disk (`/tmp/torchinductor_$USER`, ~550 MB here), so the 75 s is
a **first-ever-run** cost; later runs on the same shapes pay about half. Verified to 6 shapes;
beyond that it is the documented mechanism rather than something measured here, and there is no
hard guarantee of exactly one — a guard failure can trigger another recompile later.

Two defaults carry the feature:

- **`compile_dynamic = true`.** Without it each bucket compiles its own graph and dynamo's
  `recompile_limit` (8) is exhausted within a handful of buckets, after which it *silently falls
  back to eager for the rest of the run*. The trainer also raises the limit to 32 for headroom.
- **`compile_regional = true`.** Compiles the repeated transformer block once instead of the whole
  28-block trunk. Full-model compile was abandoned after **25+ minutes without completing a single
  step** — not a practical option here.

So: worth enabling for any real run, not for a smoke test. If you see a mid-run step spike to tens
of seconds, that is the second compile, not a hang.

`setuptools` is required (inductor shells out to a C++ compiler); it is in `pyproject.toml`, so a
fresh `uv sync` has it. Without it, compile dies at step 1 with a bare `ModuleNotFoundError` buried
inside an `InductorError`.

### `[flow]` — timestep distribution


<img width="1269" height="349" alt="imagen" src="https://github.com/user-attachments/assets/eee74625-7769-442f-8337-6511a05a517f" />



| key | default | notes |
|---|---|---|
| `timestep_sample_method` | `"logit_normal"` | or `"uniform"` |
| `sigmoid_scale` | 1.0 | >1 widens toward both extremes. **`logit_normal` only** — see below |
| `shift` | none | 3.0 = the inference default; higher = noisier training |
| `flux_shift` | `false` | resolution-dependent shift; mutually exclusive with `shift` |
| `use_ot` | `false` | cosine optimal-transport noise pairing — see below |

**`sigmoid_scale` requires `timestep_sample_method = "logit_normal" AND shift=1.0`.** It is the width of the
logistic squash applied to a normal draw; the uniform sampler has no sigmoid to scale, so the value
was read and discarded. Setting both is now **rejected** rather than ignored. Measured effect at
`logit_normal` (20k draws):

| `sigmoid_scale` | p10 | p90 |
|---|---|---|
| 0.5 | 0.343 | 0.653 |
| 1.0 | 0.214 | 0.780 |
| 2.0 | 0.069 | 0.926 |

Higher pushes mass toward *both* ends — more nearly-clean and more nearly-pure-noise samples — at
a roughly unchanged mean. Your `1.3` sits just wider than default.

#### `use_ot` — optimal-transport noise pairing

Normally each sample gets an independent noise draw, so a batch's flow trajectories cross each
other. `use_ot = true` solves a linear assignment within the micro-batch, permuting the noise rows
so each latent is paired with its most cosine-similar noise. Straighter trajectories mean lower
gradient variance and, usually, faster early convergence.

Three practical points:

- **It only permutes noise inside a micro-batch**, so it does nothing at `batch_size = 1` and gets
  stronger with larger batches. Under gradient accumulation the pairing is per micro-batch, not
  across the accumulated batch.
- **It needs `scipy`** for `linear_sum_assignment` (`uv sync --extra ot`). Without scipy it falls
  back to the identity permutation — i.e. silently off, but not silently *wrong*, since the
  identity is exactly the no-OT behaviour.
- Cost is an O(n³) assignment on an n×n cosine matrix where n is the micro-batch — negligible next
  to the transformer forward at these batch sizes.
| `hf_scale` | 0.0 | high-frequency token loss weight; 0 = off |
| `hf_exponent` | 1.0 | how sharply that loss concentrates on detailed tokens |

**Tuning note:** Anima's latents normalise to std ≈0.69, not 1.0, so effective SNR is already lower
than a unit-variance schedule assumes — and `shift` pushes `t` higher still. On illustration data
**shift < 3 is worth an A/B.** `flux_shift` crosses over static `shift=3` at about 1024px.

#### High-frequency token loss (`hf_scale`) by Wiwi

<img width="395" height="157" alt="imagen" src="https://github.com/user-attachments/assets/26e2767c-faae-400a-bbb8-ad9d16aa8ea7" />



Plain velocity MSE weights every token equally, so flat regions — most of an illustration by area —
dominate the gradient, and the edges and texture that decide whether an image *looks* sharp
contribute in proportion to how little of the frame they occupy. This term adds a second copy of
the same error, reweighted per token by the local Laplacian energy of the **clean target**:

```
total = mse + hf_scale · mean(w_token · ‖x0̂_token − x0_token‖²)
```

It regresses the predicted clean estimate (`x0̂ = noisy − t·pred`), not the velocity — at low `t`
the velocity target is noise-dominated and "which token has detail" says nothing about it. The
consequence is that the term is inherently weak at small `t` (a `t²` factor); that is deliberate
and there is no timestep gate.

Measured weight distribution on real latents (24 images at 1344×768, 4032 tokens
each), which is what `hf_exponent` actually controls:

| `hf_exponent` | median token | p99 | max | share of weight in the top 10% of tokens |
|---|---|---|---|---|
| 0.5 | 0.81 | 2.75 | 4.6 | 22% |
| **1.0** | 0.50 | 5.54 | 16.4 | **38%** |
| 1.5 | 0.24 | 8.77 | 44.8 | 53% |
| 2.0 | 0.10 | 11.9 | 101 | 66% |
| 3.0 | 0.01 | 16.9 | 351 | 83% |

Weights always have per-sample mean exactly 1, so `hf_scale` keeps meaning the same thing as
`hf_exponent` is tuned. **1.0–1.5 is the usable band**; at 3.0 the median token is weighted 0.014,
i.e. you are training on edges and nothing else. Start at `hf_scale = 0.25`, `hf_exponent = 1.0`
and read the decomposed log line — the trainer prints `loss X (mse Y + hf Z)` so you can see the
term's actual contribution rather than guessing at the scale.

`hf_scale = 0.0` is gated in Python: no extra ops, no allocations, and **no RNG draws**, so
turning it on does not shift the timestep or caption-dropout streams. A same-seed A/B is a real
A/B. Verified by `anima/parity/test_hf_loss.py` (28 checks, including negative controls for
weights-from-prediction, zero padding, and the missing-eps NaN).

### `[[curriculum]]` — timestep phases over training progress by Inpaint a.k.a. the anon who loves migu and teto (?)

<img width="1278" height="385" alt="imagen" src="https://github.com/user-attachments/assets/d2946574-bf1d-4f66-b90a-3dfd27e5bef7" />


Ported from Inpaint's TrainTrain `train_ts_schedule`. A step function over progress: the **last** phase
whose `at ≤ progress` is active. Off by default; an empty curriculum is bit-identical to the
feature not existing (gated).

```toml
[[curriculum]]
at = 0.00
t_range = [0.0, 1.0]     # continuous; TrainTrain's "0 1000"
mode = "fullres"

[[curriculum]]
at = 0.15
t_range = [0.0, 0.6]     # low-noise / detail phase
mode = "fullres"
lr_mul = 0.5
```

**Turning it off is the absence of the key, not a flag.** No `[[curriculum]]` = plain
full-resolution training over the whole timestep range, which is the default and is bit-identical
to the feature not existing. In the GUI that is the **Enable curriculum** checkbox on the Dataset
tab: unchecked writes no curriculum at all but keeps the phases in the table, so switching a
texture config back to an ordinary LoRA run — and back again — costs one click and does not throw
the schedule away. (**Clear** discards it; the checkbox is the reversible route to the same config.)

The status bar distinguishes the two, since nothing else on screen does: `curriculum 9 phase(s),
8 texture` versus plain `LoRA/lora 1024px`. It also flags `source = "encode"` left set with no
texture phase to use it — that is +16% step time bought for nothing, and the tell is easy to miss
after switching a config back.

`t_range` is **continuous [0,1]**, because the transformer takes `t = step/num_train_timesteps`.
Writing `[0, 600]` is rejected with a message telling you to divide by 1000 — under the continuous
convention it would mean "clamp everything to pure noise", which trains nothing and looks like a
tuning problem rather than a config error.

**`flow.phase_mapping` decides what a range actually means**, and the two options are not close:

| on `[0, 0.6]`, `logit_normal(1.3)` | mean | p50 | p90 |
|---|---|---|---|
| full range (no curriculum) | 0.499 | 0.499 | 0.840 |
| `"rescale"` (default, = TrainTrain) | 0.300 | 0.299 | 0.504 |
| `"truncate"` | 0.338 | 0.344 | 0.548 |

`rescale` squeezes the whole distribution into the range, so shape is preserved exactly — every
quantile is `span ×` the full one. `truncate` keeps the pretraining density and discards
out-of-range draws, leaving mass skewed toward the top of the range. Truncation is implemented by
rejection and **raises** rather than hangs if a range holds almost no probability.

`shift` is applied **before** the range map, matching the reference. The orders are not
equivalent — shift is nonlinear — and the gate pins it.

**Two deliberate deviations from TrainTrain.** Its fifth column pins an absolute LR per phase, and
its own docstring notes that this fights a decaying scheduler; here `lr_mul` **multiplies** the
scheduler's current LR, so it composes with REX and `lr_mul = 1.0` is an exact no-op. And phases
are TOML tables rather than a text block, so the loader type-checks them.

### `mode = "texture"` — crop training by Inpaint a.k.a. the anon who loves migu and teto (?).

<img width="1265" height="376" alt="imagen" src="https://github.com/user-attachments/assets/d336980b-be89-4402-8d67-c435e3102247" />


A texture phase replaces the bucketed full image with a **canvas-sized crop chosen from the source
by detail**, captioned with a trigger word alone. There is no `texture = true` switch: a curriculum
phase is the only way in, which is why the setting lives in `[[curriculum]]` and not `[dataset]`.

It requires `dataset.source = "encode"` — a crop is chosen per sample per step, so no cached latent
can express it — and `load_config` rejects the combination rather than letting a run reach its
first texture batch and die there.

```toml
[dataset]
source = "encode"

[dataset.texture]
canvases = [[1024, 1024], [832, 1216], [1216, 832], [640, 1536], [1024, 1024]]
trigger  = ""            # "" = unconditional
oversize = "pad"
```

| key | default | notes |
|---|---|---|
| `canvases` | 5 presets | one drawn per batch so the batch stacks. **Duplicates are meaningful** — `1024x1024` twice gives it a 1/3 chance, not 1/5 |
| `trigger` | `""` | captions the crop *alone*, never the image's tags |
| `oversize` | `"pad"` | `"pad"` \| `"cover"` \| `"skip"` — see below |
| `mask_ratio` | `(0.5, 1.0)` | supervised sub-region as a fraction of the canvas |
| `feather_latent_px` | `4` | cosine taper on that region's edges, in latent px (1 latent px = 8 image px) |
| `energy_power` | `1.0` | crop position ∝ Laplacian detail ^ this. `0` = uniform, the honest control |
| `energy_downscale` | `8` | the Laplacian is taken at **full** resolution and only then averaged down; downscaling first low-passes away the texture this is meant to find |

**Why the trigger replaces the tags, rather than joining them.** Anima has no size
micro-conditioning (verified: `CosmosTransformer3DModel.forward` takes no `original_size` or
`crops_coords`), so it cannot tell a zoomed crop from a large subject. Captioning a 1024px crop of
a 4000px image with the whole image's tags teaches that confusion into every one of those tags.

**The oversize cascade.** The reference black-pads: PIL fills out-of-bounds crops with zeros, and
36–43% of the texture set cannot contain the square canvases (83% for `1536x640`), so a large share of
every texture batch carries a black band — 10.9% of the average crop, measured. That band is
learnable: a padded run emitted a black bar across the bottom of roughly **1 generation in 50**.

Naive `"cover"` — scale up to fill — cuts the black to 0.25% and **lost its A/B**, visibly, in
faces. Not from aspect distortion (the resize is uniform, measured at 0.04%) but from **scale**: it
fired on 57% of draws at a mean 1.25× and up to 1.85×, so most of every batch arrived magnified and
resampler-softened.

So the fix is a cascade of four rungs, each handling what the previous could not, all on by
default and all independently switchable:

| rung | key | default | what it does |
|---|---|---|---|
| 1 | `fit_aware` | `true` | draw the batch's canvas from those its images can actually **hold** |
| 2 | `cover_max_scale` | `1.15` | allow a *gentle* upscale rather than invent pixels; `1.0` disables |
| 3 | `pad_mode` | `"reflect"` | mirror real content into the overflow instead of zero-filling |
| 4 | `mask_padding` | `true` | whatever is still invented carries **no loss** |
| — | `oversize` | `"pad"` | terminal fallback: `"pad"` \| `"cover"` \| `"skip"` |

Measured on the texture set, 413 crops through the real sampler:

| | reference | cascade |
|---|---|---|
| crops needing invented pixels | 58.4% | **0.7%** |
| pure-black pixels in the crop | 11.6% | **0.4%** |

**Rung 1 does nearly all of it** — 99% of that set fits *some* canvas, so choosing among the ones
that fit removes the padding without upscaling anything. Rungs 2–4 fired on 3 crops out of 413.
The trade is that the canvas mix follows the dataset's shapes rather than the flat preset weighting
(`1024x1024` 40%, `832x1216` 37%, `1216x832` 11%, `640x1536` 9%, `1536x640` 2%), so extreme aspects
get rarer.

Setting `fit_aware=false, cover_max_scale=1.0, pad_mode="black", mask_padding=false` is
**byte-identical** to the pre-cascade trainer — verified 200/200 identical crops —
and `configs/reference_pad.toml` is exactly that.

> ⚠ **These defaults were reverted once and then restored, and the reason is worth knowing.** The
> first cascade run had visibly broken anatomy, and the cascade was blamed. It had been run under
> **DDP** while its baseline was single-GPU, so the comparison measured DDP. A DDP run with the
> cascade *off* reproduced the same damage; a single-GPU cascade run at the full step budget beat
> the baseline. See the DDP warning below.

**Crop freedom depends entirely on source resolution**, and it decides whether any of this does
anything. Median tightest slack for a `1024x1024` canvas: **1001px on the mixed set** (4.4MP sources) against
**176px on the texture set** (1.2MP). The startup report prints a per-canvas feasibility table so a run
that cannot crop says so rather than looking like it worked.

### ⚠ Texture mode under DDP is REFUSED — use one process

**The trainer refuses this combination**, rather than warning about it — a warning printed at
startup scrolls away in the first seconds of a 25-minute run, which is exactly how two runs were
lost. `train.allow_multi_gpu_texture = true` overrides it, and exists so re-testing the seed fix is
a deliberate act. It is not an "I know better" switch: if a run under it comes out clean against a
single-GPU control at matched steps, the gate should be **removed**, not the override left on.

The GUI blocks Start on the same combination, evaluated against the GPU checkboxes rather than the
config — the process count is the one input a TOML file cannot express.

DDP remains the right tool for fullres / cached-latent training, where the 1.75× held up.

Gate: `anima/parity/test_texture.py`, 50 checks.

⚠ **TrainTrain's reference run used `shift = 3.0`.** 

Gate: `anima/parity/test_curriculum.py`, 26 checks.

### `[optimizer]` — where full FT lives or dies

<img width="1262" height="353" alt="imagen" src="https://github.com/user-attachments/assets/1f2972c8-faa7-43cc-b7c8-52ce1c97b549" />

### These are imported from Disty's SDNQ and depend entirely on it, no pytorch_optimizer due to lacking energy to gate quantized optimizers and int8 behind disty's optims, deal with it.


| key | default | notes |
|---|---|---|
| `kind` | `"adamw"` | `"adamw8bit"`, `"adafactor"`, `"came"`, `"lion"`, `"muon"` are SDNQ |
| `lr` | 1e-5 | full FT: 5e-6–1e-5. LoRA: ~1e-4 |
| `quantize_state` | `false` | needs an SDNQ `kind`; ~4× off optimizer state |
| `offload_state` | `false` | moves it to host RAM |
| `use_kahan` | `false` | SDNQ only; see below |

**`betas` length is checked at load.** CAME keeps a third moment for its instability factor and
unpacks three; the two-element default otherwise sails through config loading and dies inside
`came_update` at the first step, after model load and dataset scan, with a bare "expected 3, got 2"
that names neither the key nor the optimizer. CAME's defaults are `[0.9, 0.999, 0.9999]`.

**`use_kahan` — for when the step size approaches a bf16 ulp.** SDNQ keeps a per-parameter residual
buffer and folds the part of each update the bf16 cast discarded back into the next one. It is
**not** an alternative to stochastic rounding: SDNQ applies both, SR on the cast and Kahan on its
remainder (`SDNQ/optim/utils.py:57`). SR makes the *expected* lost update zero; Kahan makes the
*realised* one zero.

When it matters: measured on a rank-8 LoRA at `lr = 2e-5`, step/ulp is 6.8 at peak LR — safe — but
falls below 1 late in a cosine decay, which is exactly where updates start vanishing. Cost is one
buffer per trainable parameter (2 bytes plain, 0.5 quantized) and **+4% step time** measured on
the texture set (1.02 → 1.06 s/it, peak unchanged at 8.5 GB). Rejected at load for `kind = "adamw"`, which
keeps fp32 masters and has nothing to correct.

Measured full-FT ladder (1024×576, bs1×accum4, 1.76B trainable):

| rung | peak | speed |
|---|---|---|
| fp32 AdamW, no grad ckpt | **OOM** | — |
| + gradient checkpointing | 18.0 GB | 1.83 s/it |
| + SDNQ adamw | 16.8 GB | 2.02 s/it |
| + quantized state | 14.2 GB | 2.19 s/it |
| + offloaded state | **9.8 GB** | 3.44 s/it |

The baseline genuinely OOMs. For full FT, `adamw8bit` + `quantize_state` is the sane starting point.

> ### ⚠ Precision is `train.dtype`, and nothing else
>
> The `Accelerator` is constructed with `mixed_precision="no"` **explicitly**. Left unset,
> Accelerate reads `~/.cache/huggingface/accelerate/default_config.yaml` — but only under
> `accelerate launch`, so `python -m anima.training.train` and the GUI (which always launches
> through accelerate) would run **different numerics from the same config file**.
>
> With `mixed_precision: bf16` there it is not merely redundant — every module is already loaded in
> `train.dtype` — it is broken. SDNQ's training-mode Linear keeps fp32 master weights and supplies
> its own compiled backward; under autocast that backward receives a bf16 `grad_output` against an
> fp32 input and dies at the **first** `backward()`:
>
> ```
> RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float
>   ... SDNQ/training/layers/linear/forward.py, in linear_backward
> ```
>
> Reproduced on a single GPU with `accelerate launch`, and fixed by `--mixed_precision no`. It only
> ever hit full finetunes, because `mode = "none"` installs no custom backward. After the fix the
> two launch paths give matching losses (0.1295 / 0.0734 against 0.1295 / 0.0733).
>
> If the environment asks for a different precision the trainer prints that it is ignoring it,
> rather than silently disagreeing with your accelerate config.

### `[quant]` — SDNQ

| key | default | notes |
|---|---|---|
| `mode` | `"none"` | `"frozen"` (LoRA only) \| `"training"` (full FT only) |
| `weights_dtype` | `"int8"` | int8 beats fp8 on both speed and error here |
| `use_quantized_matmul` | `"auto"` | **leave it on auto** — see below |
| `skip_policy` | `"default"` | or `"all_adaln"` for ~13% lower error, +0.13 GB |

> ### ⚠ Quantized matmul does not apply in `mode = "training"`
This needs fixing and help from Disty since Anima is finicky.


> ### ⚠ Quantized matmul is a 2× SLOWDOWN below 1024px (frozen mode)


**At 768px, quantization buys memory only** (~1.1–1.8 GB) for ~10% slower steps. It becomes a real
speed win at 1024px and up. If you train at 768 and have the VRAM, plain bf16 is the honest choice.

### Learning rates


The reference config, for calibration:

```toml
# known-good diffusion-pipe config
[adapter]  type = 'lora',  rank = 32
[optimizer] type = 'adamw', lr = 2.5e-5, betas = [0.9, 0.99], weight_decay = 0.01
            # AdamW8bitKahan alternative: lr = 5e-6
self_attn_lr = 2e-5 · cross_attn_lr = 2e-5 · mlp_lr = 2e-5 · mod_lr = 2e-5 · llm_adapter_lr = 0
micro_batch_size_per_gpu = 12,  epochs = 40
```

#### Starting points

| run | LR | why |
|---|---|---|
| LoRA r32, effective batch ~12 | `2.5e-5` | straight from the reference config |
| LoRA r32, effective batch 4 | `1.2e-5`–`1.5e-5` | √-scaled down from batch 12 |
| Full FT | `5e-6`–`1e-5` | matches the reference's AdamW8bitKahan line |
| Full FT, quantized/offloaded state | `5e-6` | stay at the low end |


#### Batch size changes the LR

This trainer's per-bucket micro-batches are much smaller than diffusion-pipe's 12, so an LR copied
straight across will be too high. Effective batch = `micro_batch × grad_accum × num_gpus`.

- **√ scaling** (safer, what the table above uses): `lr_new = lr_old × √(batch_new / batch_old)`
- **Linear scaling**: more aggressive; usually too hot below batch 8

Going 12 → 4 is √(1/3) ≈ 0.58×, so 2.5e-5 → ~1.45e-5.

#### Rank changes the LR (LoRA)

The effective update scales with `alpha / rank`. With `alpha = rank` (the default for this trainer), that
ratio is 1 and **LR is roughly rank-independent** — so r16 and r64 can share an LR. But if you set
`alpha = 2 × rank`, updates double and the LR should roughly halve. Change one or the other, not
both, or you will not know which moved.

#### Per-component ratios

Absolute values matter less than the ratios.  mlp and cross_attn equal,
self_attn ~2× higher, adaln equal-or-frozen, llm_adapter `0`.

```toml
[component_lr]
self_attn  = 2e-5     
cross_attn = 1.0e-5     # prompt adherence
mlp        = 1.0e-5     
adaln      = 0.0        # freeze first; unfreeze only if style will not move
llm_adapter = 0.0       # freeze — moving it invalidates prompts the base understands
```

Rules of thumb, in rough order of confidence:

- **`llm_adapter = 0` unless you have a specific reason.** It is only 6.4% of parameters and
  training it risks the text understanding the whole architecture is built around.
- **`adaln = 0` to start.** Modulation is global — it destabilises faster than anything else, and
  freezing it also saves gradient and optimizer memory.
- **`mlp` takes the lowest LR for styles but needs the same LR as self_attn for characters** It is 48% of the Linear parameters and where most content lives treat it nicely if you want to preserve character knowledge.


#### Schedule

```toml
[schedule]
kind = "cosine"          # constant | linear | cosine | rex | rerex
warmup_steps = 20        # LoRAs don't truly need warmup, for LoKr cosplaying as finetune or full-ft ~2-5% of total steps is a good baseline
min_lr_ratio = 0.0
```

**REX and ReREX** The difference from cosine is not subtle — mean LR
multiplier over a 1000-step run:

| | 250 | 500 | 750 | mean multiplier |
|---|---|---|---|---|
| `linear` | 0.75 | 0.50 | 0.25 | 0.50 |
| `cosine` | 0.85 | 0.50 | 0.15 | 0.50 |
| `rex` | 0.97 | 0.91 | 0.75 | **0.83** |
| `rerex` | 0.95 | 0.83 | 0.62 | **0.76** |

REX holds the LR near its peak for most of the run and then drops sharply at the end. That is why
it works well for LoRA — an adapter has few enough parameters that it tolerates a high LR for
longer, and the late collapse acts as the anneal. It also means **the same `lr` does roughly 1.7×
more total movement under `rex` than under `cosine`**, so port an LR across schedules and you have
silently raised it. Divide by about 1.6 when switching cosine → rex.

```toml
[schedule]
kind = "rex"
warmup_steps = 20
min_lr_ratio = 0.001     # sd-scripts hardcodes this; 0.0 decays fully to zero instead
d = 0.9                  # 0 == linear; higher holds the peak longer. sd-scripts' hardcoded value
```

```toml
[schedule]
kind = "rerex"
warmup_steps = 20
min_lr_ratio = 0.001
global_d = 0.78          # `d` of the outer curve the segment endpoints ride
local_d = 0.85           # `d` of the decay inside each segment
weight_power = 1.5       # step-budget skew toward early segments; 0 == equal lengths
num_segments = 8
```

ReREX is **not** a warm-restart schedule, despite the segmentation — segment *i* ends exactly where
*i+1* begins, so the curve is continuous and monotone. What the recursion buys is pacing control a
single `d` cannot express: `weight_power` stretches the high-LR region and compresses the tail,
`local_d` reshapes the descent within each segment. Net effect at the defaults: tracks REX early,
pulls below it after the midpoint. Every one of these is tunable here; sd-scripts hardcodes `d`,
`min_lr` and `max_lr` for REX and exposes the ReREX ones only through `--lr_scheduler_args`.

One caveat on short runs: at `weight_power = 1.5` with 8 segments the final segment gets ~1.2% of
the budget, so below ~85 steps several segments round down to zero length. That degrades
gracefully (verified down to 1 step) but the curve stops being the one the table above describes —
use `rex`, or lower `weight_power`, for smoke tests.

Warmup matters more here than usual: full FT at bf16 with a quantized optimizer has less numerical
headroom, and a cold high LR in the first steps is where divergence happens. `warmup_steps` counts
**optimizer steps** (after accumulation), not samples — and it means the same thing on 1 GPU as on
2. (Accelerate's scheduler wrapper advances the inner schedule `num_processes` times per call, so
the horizon is scaled to compensate; without that, a cosine decay on 2 GPUs reached lr 0 at the
halfway point and trained the rest of the run at zero.) The multiplier applies to each group's own
peak LR, so per-component ratios are preserved through warmup and decay.

#### Betas and weight decay

**This trainer defaults to`(0.9, 0.95)` and `0.01`** — set them explicitly to match your practice:

```toml
[optimizer]
betas = [0.9, 0.99]
weight_decay = 0.04
```

β₂ = 0.99 averages the second moment over ~100 steps instead of ~20. On small datasets with noisy
per-step loss (which flow matching has by construction — every step draws a different timestep),
the longer window is the safer choice.

#### Reading the loss

Flow-matching loss is **inherently noisy step to step** — each step samples a different `t`, and
loss at `t=0.95` is ~2.5× loss at `t=0.05`. Single-step values tell you nothing; a 0.045 → 0.145
swing between consecutive steps is normal, not divergence.

- Judge trends over ≥50 steps, or use fixed-quantile eval loss (what `test_train_sanity.py` does).
- **Too high**: loss trends up, or output saturates/burns.
- **Too low**: loss flat over hundreds of steps and samples unchanged.
- **Diverged**: NaN, or loss jumps an order of magnitude and stays. Lower LR, raise warmup, and
  check `max_grad_norm = 1.0` is set.

Re-run `test_train_sanity.py` after training: the text-conditioning margin starts small (~4%) on
the base model and **should widen** if the run learned anything from your captions.

### `[dataset.caption]`

<img width="1247" height="225" alt="imagen" src="https://github.com/user-attachments/assets/ba11a018-acc5-4bfe-abec-448a5b05cb62" />


`caption_mode` is `tags` | `nl` | `tags_nl` | `nl_tags` | `mixed`. Captions are rebuilt every
`__getitem__`, so shuffling and dropout give a different view each epoch — which is exactly why
text embeddings are not cached.

```toml
[dataset.caption]
caption_mode = "mixed"
mixed_weights = { tags = 50, nl = 10, tags_nl = 20, nl_tags = 20 }
shuffle_tags = true
shuffle_keep_first_n = 1        # pin trigger words
tag_dropout_percent = 0.1
caption_dropout_percent = 0.05  # unconditional samples, for CFG
```

### `[train].batch_size` — per resolution tier

One number, or one entry per tier:

```toml
resolution = 1024
batch_size = 12                                   # one tier -> one number

resolutions = [768, 1024, 1280]
batch_size  = { 768 = 16, 1024 = 12, 1280 = 8 }   # keys ARE the tiers
```

**Keys are matched exactly against the declared tiers.** This is not a threshold ladder: a key that
is not a declared tier, or a declared tier with no key, is a config error at load.

Keyed on the tier because the tier predicts cost and the bucket's longest side does not. Bucketing
targets a constant *area*, so `tier² / 256` is an **exact** upper bound on a tier's token count —
measured on the mixed set, tiers 768/1024/1280 top out at exactly 2304/4096/6400 tokens. The longest sides,
meanwhile, overlap almost completely:

| tier | max tokens | `tier²/256` | longest side |
|---|---|---|---|
| 768 | 2304 | 2304 | 704–1280 |
| 1024 | 4096 | 4096 | 768–1792 |
| 1280 | 6400 | 6400 | 768–1920 |

A tier-768 bucket can be 1280 on its long side and a tier-1280 bucket can be 768, so sorting by
longest side sorts by something that is not the cost.

**Sizing it.** What costs memory is `images × tokens`, where `tokens = W×H / 256`. One measurement
to anchor against: **batch 12 at ~4096 tokens peaked at 19.9 GB** on a free 24 GB card (LoRA, bf16,
gradient checkpointing); batch 10 at the same tier sat at **17.9 GB**, and batch 12 OOMed on a
dataset whose buckets reach the full 4096. Leave headroom on GPU 0 if a desktop lives there.

---

## 4. Outputs

ComfyUI-loadable, no conversion step. **One file per checkpoint**, named sd-scripts style:

```
output/<run_name>/
    <run_name>-step000060.safetensors     periodic
    <run_name>-epoch005.safetensors       (whichever cadence you set)
    <run_name>.safetensors                final — the bare run name, no tag
    <run_name>.toml                       the config this run used
    <run_name>-step000060-state/          only with save_optimizer_state
```

A folder per checkpoint gave every LoRA in a run the same filename, so twenty of them could not sit
in one ComfyUI models folder without being renamed by hand.

- Full FT → native single-file layout (685 tensors), verified bit-identical round trip.
- LoRA → `diffusion_model.*` keys, round-trips through diffusers' `AnimaLoraLoaderMixin`.
  `save_lora_checkpoint(fmt="kohya")` emits `lora_unet_*` instead; ComfyUI loads both.
- A full finetune **without** `save_native` still writes a directory per checkpoint — `transformer/`
  and `text_conditioner/` are diffusers directories, so that is the format's shape, not a naming
  choice.

**The config is copied verbatim** to `<run_name>.toml`, comments and all, rather than
re-serialised from the dataclasses — a round trip would emit resolved values and drop every comment
explaining them. Checkpoints outlive configs: a file gets edited for the next experiment and the
run that produced a LoRA becomes unreconstructable otherwise.

**To resume you must set `save_optimizer_state = true`** — otherwise checkpoints are
inference-only and `resume_from` will refuse rather than silently restart the optimizer moments.
`resume_from` accepts either the `-state` directory or the **checkpoint file itself**, since the
file is what you actually have in hand.

`keep_last_n = 3` prunes older periodic checkpoints as new ones land, removing everything belonging
to one checkpoint together — the file, its `_te` sibling and its `-state` directory — so pruning
cannot leave orphaned optimizer state that is larger than the checkpoints. Candidates must carry
this run's name *and* a periodic tag; **`final`, the saved TOML and anything else in the directory
are never touched.** Ordering is by recorded step, not mtime, so epoch- and step-tagged checkpoints
sort correctly against each other and a resumed run cannot delete the wrong one. Directory-form
checkpoints written before the layout was flattened are still recognised.

### Progress and ETA

<img width="2547" height="1191" alt="imagen" src="https://github.com/user-attachments/assets/4509b115-93f8-4a1c-8d07-b2f70ba53789" />


`train.progress` picks how the run reports itself. The default `auto` draws a tqdm bar when stdout
is a terminal and prints the per-step line when it is not — a bar redirected into a log file is
thousands of `\r`-separated fragments, so `nohup … > train.log` still gets readable output without
a config change. `bar` / `plain` force either; `off` suppresses the per-step report entirely
(startup summary and checkpoint messages still print).

```
e3  47%|█████▍      | 1410/3000 [42:11<47:34, 1.80s/step, loss 0.0712 (mse 0.0650 + hf 0.0062)  ot 0.83  lr 2.71e-05  peak 8.1GB]
e3 step 1410/3000 (47%)  loss 0.0712 (mse 0.0650 + hf 0.0062)  ot 0.83  lr 2.71e-05  1.80s/it  peak 8.1GB  eta 47m34s
```

Both ETAs are a **cumulative mean** of step time, not the last interval. That is deliberate:
bucketing gives every bucket a different token count, and multi-resolution widens the spread
further (768px → 2304 tokens vs 1280px → 6240), so an ETA extrapolated from the previous step would
swing several-fold between log lines depending on which bucket came up. The mean over many steps
sees the bucket mix in its true proportion.

The average is anchored **after the first step**, so `torch.compile` (~35-75 s) and the first
dataloader fork do not inflate the estimate for the rest of the run. Later compiles — a new bucket
appearing at step 9 or 17 — do land in the average, and amortise away.

Under DDP only the main process draws anything; N ranks writing `\r` to one terminal is unreadable.
Checkpoint and prune messages route through `tqdm.write`, so they scroll above the bar instead of
tearing it.

---

## 5. Sanity checks

Most gates are self-contained and CPU-only. The ones that compare against something external take
their paths from the environment and **skip with instructions** rather than failing when it is not
set up, so a fresh clone can run the suite without owning any of it:

| variable | needed by | what it points at |
|---|---|---|
| `ANIMA_MODEL` | most GPU gates, `TrainConfig.model_path` default | the converted diffusers repo; defaults to `../anima-diffusers` |
| `ANIMA_BUCKET_ORACLE` | `test_bucket_parity.py` | a directory sd-scripts has already cached (`*_sd.npz`), whose filenames record the buckets it chose |
| `ANIMA_DATASET_UNIFORM` · `ANIMA_DATASET_MIXED` | `test_multires.py` | a uniform-resolution set and a mixed one |
| `ANIMA_NATIVE_CKPT` · `ANIMA_REF_ROOT` | `test_transformer_parity.py` | the original (non-diffusers) checkpoint, and a diffusion-pipe checkout with its own venv |

```bash
# is the loop actually conditioned on text and timestep?
CUDA_VISIBLE_DEVICES=1 venv/bin/python anima/parity/test_train_sanity.py configs/lora.toml

# VRAM ladder for your config
venv/bin/python anima/parity/bench_vram.py configs/full.toml --gpu 1

# quantized-matmul crossover on this GPU
CUDA_VISIBLE_DEVICES=1 venv/bin/python anima/parity/test_SDNQ_quality.py --sweep

# bucketing still matches sd-scripts exactly (2000/2000)
venv/bin/python anima/parity/test_bucket_parity.py

# REX/ReREX match sd-scripts exactly (add --plot for ASCII curves)
venv/bin/python anima/parity/test_schedule_parity.py

# high-frequency token loss vs the spec, with negative controls
venv/bin/python anima/parity/test_hf_loss.py

# multi-resolution: cross product, tier attribution, collapse warning, default-path regression
venv/bin/python anima/parity/test_multires.py          # add --waf for the 2000-image mixed-set check

# GUI: config bridge, widget coverage, inert-knob rules, log parsing, launch argv (headless)
venv/bin/python anima/parity/test_gui.py

# re-pin the quantized-matmul crossover on this GPU
CUDA_VISIBLE_DEVICES=1 venv/bin/python anima/parity/test_SDNQ_quality.py \
    --sweep --sizes 88 96 104 112 120 128
```

`test_train_sanity.py` is the one to re-run after any change to the model or data path: a training
loop can run, decrease a loss, and still ignore the prompt entirely.

---

## 5b. GUI

```bash
./start-gui.sh                         # after install.sh / install.bat
# or, if it failed to install:
uv sync --extra gui                    # PySide6, ~150MB
venv/bin/python -m anima.gui
```

### The config file is named by `run_name`

Saving writes `configs/<run_name>.toml`, and changing `run_name` **renames** the file to match
rather than leaving a stale duplicate for the dropdown to offer. The two were never independent:
`run_name` already decides the checkpoint stem (`output/<run_name>/<run_name>.safetensors`) and the
copy of the TOML the trainer writes beside it. One name, three places.

Free text reaches the filesystem here, so it is sanitised — spaces become dashes, `/` and the
Windows-illegal set are dropped, a leading dot is stripped, and the DOS device names (`CON`, `NUL`,
`COM1`…) get a suffix. Accented and non-Latin letters are **kept**: both NTFS and ext4 store them,
so folding `sesión` to `sesi-n` would mangle a good name for no gain.

A rename onto an existing config asks first, because a `run_name` collision is also a *checkpoint*
collision — two runs sharing it overwrite each other's output, not just this file.

### A config opened from outside `configs/` is never written back to

Open loads a TOML from anywhere. Saving it — or pressing Start — copies it into
`configs/<run_name>.toml` and leaves the original **byte-identical**. The GUI regenerates TOML
through `dump_toml`, so writing back in place would silently discard the comments of whoever's file
you opened to read. `Save As` remains the explicit escape hatch when you do want to write elsewhere.

Four config tabs (Dataset · Training · Optimizer · Method) covering **all 96 config keys**, plus a
live-metrics tab and a console. It reads and writes the same `configs/*.toml` the CLI takes — there
is no separate GUI config format, and a file the GUI saves is a file you can hand-edit and commit.

Chrome (theme, console, graphs, process runner) is vendored from
[Aozora Trainer](https://github.com/Hysocs/Aozora_Trainer) under Apache-2.0; see `NOTICE`.

### It reimplements no validation

Every edit re-serialises the form and hands it to `load_config` — the same function the trainer
calls — then shows whatever it raises and keeps **Start** disabled until it passes. So every guard
this trainer has accumulated applies in the GUI for free and cannot drift out of sync: both
`resolution` and `resolutions` set, `sigmoid_scale` under `uniform`, a `component_lr` on a
component with no adapter injected, `quant.mode = "frozen"` with nothing to train.

On top of that it **greys out knobs that are currently inert** — a step ahead of the validators,
which only catch what is outright wrong:

| greyed out | because |
|---|---|
| `sigmoid_scale` under `uniform` | measured byte-identical at scale 0.5/1.0/2.0 |
| `min_bucket_reso` while `bucket_no_upscale` | the no-upscale branch never reads it |
| ReREX knobs under `rex` (and vice versa) | different curve entirely |
| `component_lr.adaln` under a LoRA not targeting adaln | no adapter injected there |
| every SDNQ knob at `mode = "none"` | nothing is quantized |
| `compile_dynamic` / `compile_regional` with compile off | nothing to compile |
| `hf_exponent` at `hf_scale = 0` | the term is gated off in Python |

### What it writes

Only values that **differ from the dataclass default**, so a saved config reads like a hand-written
one rather than pinning all 96 keys — and a default that changes later is not frozen into every
config the GUI ever saved. `adapter.kind` is the one exception, always written: it is the
full-FT/LoRA/LoKr selector and should never be implicit.

Gated at 49/49: both shipped configs round-trip through the form to a **byte-identical parsed
`Config`**, every key has exactly one widget, and every widget maps to a key that exists.

### Buttons

**Start** runs `-m anima.training.train` directly at 1 process, or through `accelerate launch` at
2+. Stop kills the whole process group, so under DDP every rank goes with it rather than leaving
orphans on the GPUs. **Audit dataset** prints the source-size percentiles and a proposed ladder
(no GPU, no model). **Cache latents** and its dry run drive `anima/tools/cache_latents.py` at the
configured tiers. The **GPUs** box sets `CUDA_VISIBLE_DEVICES` for the child only.

### Live metrics

Parsed from the per-step log line, so the graphs show exactly what the console does: loss with an
EMA and the mse/hf decomposition on one axis, LR, s/it, and peak VRAM **per rank** (the ranks are
not symmetric, and reporting only rank 0 hides which one is the batch-size ceiling). Under the GUI
stdout is a pipe, so `progress = "auto"` already resolves to the parseable plain line — the bar
stays out of the way on its own.

---

## 6. Complete config reference

Section 3 covers the knobs worth *tuning*. This is every key the loader accepts — unknown keys are
a hard error, so if it is not here it will be rejected.

### `[train]`

| key | default | notes |
|---|---|---|
| `model_path` | `../anima-diffusers` | converted diffusers repo |
| `output_dir` · `run_name` | `output` · `anima` | `<output_dir>/<run_name>/<run_name>-stepNNNNNN.safetensors` |
| `epochs` | 1 | |
| `max_steps` | none | hard stop; overrides `epochs` when reached first |
| `batch_size` | 1 | int, or `{tier = size}` keyed exactly on the declared tiers — see §3 |
| `gradient_accumulation_steps` | 1 | **the highest-value throughput knob**: 2.04× measured going 1 → 4 |
| `gradient_checkpointing` | `true` | recompute activations; without it full FT OOMs outright |
| `dtype` | `"bfloat16"` | `float16` / `float32` accepted; bf16 is what everything is measured at |
| `seed` | 42 | seeds the sampler, caption RNG, and noise |
| `num_workers` | 2 | dataloader workers; re-forked each epoch (the sampler's epoch is read at `__iter__`) |
| `vae_encode_chunk` | 1 | images per VAE forward under `source = "encode"`; raising it costs memory and buys no speed — see §1 |
| `save_every_steps` · `save_every_epochs` | none · 1 | periodic checkpoints |
| `save_native` | `true` | single-file ComfyUI layout; `false` writes a diffusers directory instead |
| `save_optimizer_state` | `false` | **required to resume**; costs ~optimizer-state size per checkpoint |
| `keep_last_n` | none | prune older periodic checkpoints — see §4 |
| `resume_from` | none | a checkpoint file or its `-state` dir; refuses if there is no saved optimizer state |
| `skip_final_save` | `false` | benchmarking only |
| `log_every` | 1 | |
| `progress` | `"auto"` | `auto` \| `bar` \| `plain` \| `off` — see §4 |
| `compile` · `compile_dynamic` · `compile_regional` | none · `true` · `true` | see §3 |

### `[dataset]`

| key | default | notes |
|---|---|---|
| `path` | — | one source directory; required unless `subsets` is given |
| `subsets` | none | `[[dataset.subsets]]` tables, each with `path` · `num_repeats` · `texture`. Mutually exclusive with `path`. See below |
| `resolution` | 1024 | AREA budget, not a side length |
| `resolutions` | none | multi-resolution tiers; mutually exclusive with `resolution` |
| `tier_collapse` · `min_source_area` | `"dedup"` · 0.0 | see §1 |
| `min_bucket_reso` | 256 | **inert when `bucket_no_upscale = true`** — that branch derives buckets from the source and never reads a minimum |
| `max_bucket_reso` | 1920 | PER-SIDE cap; the in-spec RoPE range |
| `bucket_reso_steps` | 64 | bucket dimensions are multiples of this |
| `bucket_no_upscale` | `true` | never enlarge a source; changing this invalidates every cache |
| `multires_training` | `false` | **not** multi-resolution training — an area tie-break among equally-good aspect ratios. Badly named, inherited from the sd-scripts fork |
| `num_repeats` | 1 | duplicate every entry N times; the right knob for rebalancing, not the tier ladder |
| `source` | `"auto"` | `"images"` \| `"latents"` \| `"encode"` \| `"auto"` — `"encode"` trains without a cache, at +0.4GB and +16% step time; see §1 |

#### `[[dataset.subsets]]` — several source directories


<img width="1269" height="541" alt="imagen" src="https://github.com/user-attachments/assets/82127195-5982-4f5c-b21e-c8a97d1e086e" />


```toml
[dataset]
source = "encode"

[[dataset.subsets]]
path = "/data/train"

[[dataset.subsets]]
path = "/data/colors"
num_repeats = 2
texture = false
```

`path` and `subsets` are mutually exclusive; the single-`path` form is exactly a one-subset dataset
carrying the top-level `num_repeats`, so existing configs are unaffected. All subsets must resolve
to the same `source` mode — a batch carries either pixels or latents, never both, so a mixed
dataset is refused at load rather than at the first batch that spans them.

**`texture = false` is why this exists.** Texture crops replace the image's caption with
`texture.trigger` (see `[dataset.texture]`), which is deliberate for detail crops of a captioned
subject and inverts the intent for a regularization set:

> A flat colour field trained with an empty caption teaches the **unconditional** branch that
> colour. CFG computes `pred = uncond + scale·(cond − uncond)`, so it then *subtracts* it —
> colour anchors delivered through texture mode make generations less colourful, not more, and at
> guidance 4.0 the subtraction is amplified 4×.

An exempt subset stays full-resolution for the whole run with its captions intact, whatever the
curriculum is doing. The sampler groups batches by `(bucket, texture eligibility)` so a batch is
never half one and half the other — a batch gets one canvas, so mixing them could not honour both.

Flat images are also the one input texture mode cannot use: `energy_map` of a constant image is
exactly 0.0, so crop selection falls back to a uniform position. Texture mode exists to find
native-scale high-frequency detail, and a solid colour has none by construction.

`num_repeats` is per subset and is what balances the mix. The startup report prints each subset's
share of the epoch, because that share is `entries × num_repeats` rather than the file count:

```
dataset  535 samples, 42 buckets
         train                       413 entries  77.2% of the epoch  x1
         og                          122 entries  22.8% of the epoch  x2  [fullres only -- keeps its captions]
```

`configs/anchors.toml` is a worked example.

### `[dataset.caption]`

| key | default | notes |
|---|---|---|
| `caption_mode` | `"tags"` | `tags` \| `nl` \| `tags_nl` \| `nl_tags` \| `mixed` |
| `mixed_weights` | equal | relative weights per mode when `caption_mode = "mixed"` |
| `shuffle_tags` · `shuffle_keep_first_n` | `false` · 0 | pin the first N tags while shuffling the rest |
| `tag_dropout_percent` · `caption_dropout_percent` | 0.0 | per-tag / whole-caption dropout |
| `min_tags_kept` | 3 | floor that survives even 100% tag dropout |
| `tag_delimiter` | `", "` | |
| `protected_tags` · `protected_tags_file` | empty | never shuffled away or dropped |
| `nl_shuffle_sentences` · `nl_keep_first_sentence` | `false` · `true` | sentence-level shuffling for the NL caption |

### `[adapter]`

| key | default | notes |
|---|---|---|
| `kind` | `"lora"` | `lora` \| `lokr` \| `none` |
| `rank` · `alpha` · `dropout` | 32 · 32.0 · 0.0 | |
| `components` | attn+mlp | adds `adaln` (+9% of Linear params) and `base` (+1%) |
| `train_llm_adapter` | `false` | adapter on the AnimaTextConditioner |
| `train_text_encoder` · `text_encoder_rank` | `false` · = `rank` | Qwen3 adapter; exports separately (diffusers' mixin cannot round-trip it). **Set `component_lr.text_encoder`** |
| `lokr_factor` · `lokr_decompose_both` | -1 · `false` | LoKr only; -1 picks the most balanced factorisation |

### `[quant]`

| key | default | notes |
|---|---|---|
| `mode` · `weights_dtype` · `use_quantized_matmul` · `skip_policy` | see §3 | |
| `quantize_text_conditioner` · `quantize_text_encoder` | `false` | quantize the text path too; small next to the trunk |
| `extra_skip` | empty | extra skip keys. **Not substring matching** — a key must equal the full parameter name, be one whole dot-separated component of it, or be a glob |
| `dynamic_loss_threshold` | none | per-layer bit-width search; SD.Next suggests 1e-4 / 1e-3 / 1e-2 for 8 / 6 / 4-bit |
| `group_size` | 0 | 0 = per-channel scales |
| `use_stochastic_rounding` | `true` | load-bearing for `mode = "training"`: round-to-nearest discards most updates at a 1e-5 LR |

---

## Gotchas

- **Never compare tensors in bf16.** Two mathematically identical implementations differ by ~5e-3
  there; that is noise, not a bug.
- **Latent normalisation constants are Anima's pretraining values.** Real data normalises to
  std ≈0.69. Do not "fix" it with dataset statistics.
- **Per-side RoPE limits: 1920 in spec, 2048 hard.** Not area limits — `3072×768` has the same
  area as `1536×1536` and still fails.
- Omitting `[adapter]` gives you a LoRA, not a full finetune.
