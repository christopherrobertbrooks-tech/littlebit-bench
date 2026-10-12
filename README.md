# LittleBit on a Tesla V100: how much quality is left below 1 bit per weight?

**Status (2026-10-11, night 1): in progress.** The V100 port works and matches upstream's bf16 training step for step.
The first long run (Qwen3-0.6B at 1.0 bit) is training now.

[LittleBit](https://github.com/SamsungLabs/LittleBit) (Samsung Labs, NeurIPS 2025) and LittleBit-2 (ICML 2026) compress
an LLM's weights to **1.0 down to 0.1 bits per weight**. Each weight matrix becomes two low-rank factors stored as
±1 signs, plus small learned scales. Quality is recovered with quantization-aware training (QAT), using the original model
as a teacher. Samsung hasn't released checkpoints, so the only way to see the quality loss is to train the models yourself.

This repo does that on **one 32 GB Tesla V100 (Volta, 2017)**:

- **Models:** Qwen3, smallest first.
- **Measures:** perplexity and the paper's seven zero-shot tasks at each bit level.
- **Compared with:** the uncompressed model, and llama.cpp quants of the same model scored by the same code.

We are not affiliated with Samsung. All method credit goes to the LittleBit authors (see [Credit](#credit-and-licence)).

## Results so far (Qwen3-0.6B)

| weights | bits/weight (body) | wikitext2 PPL | c4 PPL | trained on |
| :--- | ---: | ---: | ---: | :--- |
| fp16 (uncompressed) | 16 | 20.95 | 25.39 | -- |
| llama.cpp IQ2_XXS (imatrix) | 2.11 | 621.8 | 507.2 | -- |
| llama.cpp IQ1_S (imatrix) | 1.72 | 4,071.7 | 3,738.9 | -- |
| LittleBit-2, 1.0 bit, SVD + Joint-ITQ init only | 1.0 | 17,139,636 | | 0 tokens |
| LittleBit-2, 1.0 bit, short probe | 1.0 | 4,008.0 | 2,207.1 | 400 samples (0.8M tokens) |

How these were measured:

- **Same scorer for every row.** Perplexity uses LittleBit's own `eval.py` at sequence length 2048 (`results/*/eval-*.json`).
  The zero-shot tasks are the paper's seven: boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa
  (lm-eval 0.4.8, 0-shot). They run at the end of each long run.
- **How the llama.cpp rows were made:**
  1. The model was converted to an f16 GGUF.
  2. An importance matrix was computed from WikiText-2 *train*.
  3. The model was quantized with embeddings and output left at f16, as LittleBit leaves them.
  4. Each weight was dequantized back to fp16 with gguf-py's reference code (`scripts/gguf_quants.sh`,
     `scripts/gguf_to_hf.py`).

  So llama.cpp's rounding is scored by exactly the same code as LittleBit. The f16 round trip reproduces the original
  model's perplexity to all four decimals (20.9550 / 25.3941). "Body" bits per weight counts the transformer's linear
  layers only; llama.cpp keeps some of them at higher precision, which is why IQ1_S averages 1.72.
- **IQ1_S was scored in fp32.** In fp16 it overflows and gives NaN.
- **Small models are hard to compress.** Even llama.cpp's 2-bit quant is far from usable on a 0.6B model. Night 1
  asks whether a longer LittleBit run at **1.0 bit** gets under IQ2_XXS's 622 at twice the bits.

### What 1 bit buys on disk (`scripts/sizes.py`)

The saved 1.0-bit model (`results/probe-lr1.6e-4/sizes.json`) is 647 MiB. Qwen3-0.6B's own fp16 file is 1.5 GB.

| part | size |
| :--- | ---: |
| embeddings | 296.8 MiB |
| lm_head | 296.8 MiB |
| binary factors (packed) | 50.5 MiB |
| scales | 2.9 MiB |

The compressed layers really are at 1.016 bits per weight (440.4M weights), down from 840 MiB in fp16, 16x smaller.
But at this model size the uncompressed embeddings dominate. Qwen3-0.6B ties lm_head to the embeddings, yet both
files store a copy, so one 297 MiB copy could be dropped.

### The port is faithful: fp16 vs upstream bf16

Same 96 samples, same seed, lr 1.6e-4, both on the V100 (`results/port-fp16`, `results/port-bf16`). The two loss
curves track each other step by step: the final (KL, layer) losses are 10,952 / 11,656 in fp16 and 11,006 / 11,435 in
bf16, and the mean training loss is 55,546 vs 56,530. **fp16 is 3.1x faster: 1.20 vs 0.39 samples/s.** Part of each
step stays in fp32 either way (the 152k-word softmax and the losses), so the gain is smaller than a raw matmul's
(84.6 vs 9.6 TFLOPS on this card).

### Learning rate

Upstream's README example uses 4e-5, for Llama-2-7B trained on ~850M tokens. On the 0.6B model, over 100 steps, 1.6e-4 learns
clearly faster. Averages over the last eighth of each run:

| lr | KL loss | layer loss |
| :--- | ---: | ---: |
| 4e-5 | ~11,900 | ~12,000 |
| 1.6e-4 | ~10,000 | ~7,400 |

Night 1 uses 1.6e-4 (`results/probe-lr*`).

### Training budget

The upstream recipe (paper + README + `datautils.py`) trains for 5 epochs over C4 shard 0 plus WikiText-2: 82,764
sequences of 2,048 tokens, about 850M tokens in all. This V100 trains the 0.6B model at ~2,450 tokens/s, so that recipe would take about **4 days per bit level**.
Runs here are much shorter, and each row says what it was trained on. Night 1 is 24,000 sequences (49M tokens,
~5.5 h), with WikiText-2 perplexity logged every 500 steps (`ppl_curve.jsonl`).

## What had to change for a V100 ([`patches/volta-fp16.patch`](patches/volta-fp16.patch))

Volta has no bf16 hardware. PyTorch still reports bf16 as "supported" on it, and then emulates it about 9x slower.
LittleBit trains and evaluates in bf16 everywhere. The patch:

1. **Trains in fp16:**
   - New `--precision fp16` flag: fp16 autocast with loss scaling.
   - Every trainable tensor is kept in fp32. Loss scaling requires this, and LittleBit's latent factors are already fp32.
   - The KL and layer-to-layer MSE losses are computed in fp32. Upstream ran the MSE under a bf16 autocast, and in
     fp16 the squares of large hidden states can overflow.
2. **Builds the teacher without DeepSpeed** when no DeepSpeed config is given. Upstream always wraps the teacher in
   ZeRO-3 inference, even on one GPU.
3. **Fixes the plain single-GPU run.** Without a launcher, `LOCAL_RANK` is unset, so `get_device_config()` returns
   `"auto"` and `model.to("auto")` raises. This affects the README's own `CUDA_VISIBLE_DEVICES=0 python -m main ...`
   command. It also counted GPUs with GPUtil, which ignores `CUDA_VISIBLE_DEVICES`.
4. **Adds `--max_train_samples`:** trains on a seeded random subset, so a run fits in a night. The paper's recipe is
   5 epochs over a full C4 shard plus WikiText-2.
5. **Adds `--load_direct_to_gpu`:** builds the student on the GPU, for hosts with less RAM than the model (this one
   has 16 GB).
6. **Adds `--fp16_init_scale` (default 1.0).** LittleBit's loss starts near 1e5: the KL is summed over every token
   and the layer loss is weighted 10x. Its fp16 gradients overflow at torch's default starting loss scale of 2^16, so
   the first ~19 optimizer steps were skipped while the scale backed off to 0.125. The gradient norm is ~1e6 and is
   always clipped to 1.0, so the loss scale does not change the updates.
7. **Adds `--eval_every N`:** a learning curve. Every N steps it logs WikiText-2 perplexity on the same windows as
   `eval.py`.
8. **`eval.py --dtype`** picks fp16/fp32 explicitly. `--results_json` writes the numbers to a file. The perplexity
   loss is computed in fp32.

The defaults are unchanged. Without the new flags, upstream's bf16 behaviour is the same.

## Reproduce

```bash
./setup.sh                        # clone LittleBit @42d658b, apply the patch, Python 3.12 venv (torch 2.6 cu124)
scripts/eval.sh qwen3-0.6b-fp Qwen/Qwen3-0.6B fp16 fp                     # uncompressed baseline
SAMPLES=24000 LR=1.6e-04 scripts/train.sh qwen3-0.6b-b1.0-long Qwen/Qwen3-0.6B 1.0 --eval_every 500
scripts/eval.sh qwen3-0.6b-b1.0-long /mnt/steam/littlebit/runs/qwen3-0.6b-b1.0-long fp16
scripts/gguf_quants.sh qwen3-0.6b Qwen/Qwen3-0.6B                       # llama.cpp quants of the same model
python3 scripts/summarize.py                                            # the results table
```

`scripts/night1.sh` is the exact queue of night 1.

`scripts/env.sh` pins every run to the V100 (`CUDA_VISIBLE_DEVICES=1` in PCI order on this host; override with `LB_GPU`).

Environment: Ubuntu, i7-13700KF, 16 GB RAM, Tesla V100-PCIE-32GB. Python 3.12, torch 2.6.0+cu124, transformers 4.51.3,
datasets 3.4.1, lm-eval 0.4.8.

## Credit and licence

- **LittleBit: Ultra Low-Bit Quantization via Latent Factorization.** Banseok Lee, Dongkyu Kim, Youngcheon You,
  Youngmin Kim. NeurIPS 2025. [arXiv:2506.13771](https://arxiv.org/abs/2506.13771)
- **LittleBit-2: Maximizing the Spectral Energy Gain in Sub-1-Bit LLMs via Latent Geometry Alignment.** Banseok Lee,
  Youngmin Kim. ICML 2026. [arXiv:2603.00042](https://arxiv.org/abs/2603.00042)
- Code: [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit), licensed
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

`patches/volta-fp16.patch` modifies Samsung's code, so it is under the same CC BY-NC 4.0 licence. Everything else in
this repo is offered under the same terms, for simplicity. **Non-commercial use only.** Changes from upstream are
listed above, and each changed spot in the patch is marked `[littlebit-bench]`.

**AI assistance:** the setup, the patch and the runs were done with an AI coding assistant (Claude, by Anthropic). A
person directed the work and reviewed the results. Every number here comes from a log or JSON file under `results/`.
