# LittleBit on a Tesla V100: how much quality is left below 1 bit per weight?

**Status (2026-10-13): Qwen3-0.6B and Qwen3-1.7B done at 1.0 bit.**

- **Result:** each was trained for one night's budget (49M tokens, ~6% of the upstream recipe). At 1 bit they write
  far more fluently than llama.cpp's quants at 1.7-2.1 bits, but keep only about a quarter of the original's quiz
  skill. On the quizzes, the 1-bit 1.7B ties llama.cpp's IQ2_XXS, which uses twice the bits.
- **Bigger helped fluency** (1.80x -> 1.62x of the original's perplexity) but not knowledge (23% -> 25% kept).
- **Found and fixed along the way:** an fp16 overflow inside the 1-bit layer, reported upstream as
  [SamsungLabs/LittleBit#24](https://github.com/SamsungLabs/LittleBit/issues/24).

[LittleBit](https://github.com/SamsungLabs/LittleBit) (Samsung Labs, NeurIPS 2025) and LittleBit-2 (ICML 2026) compress
an LLM's weights to **1.0 down to 0.1 bits per weight**. Each weight matrix becomes two low-rank factors stored as
±1 signs, plus small learned scales. Quality is recovered with quantization-aware training (QAT), using the original model
as a teacher. Samsung hasn't released checkpoints, so the only way to see the quality loss is to train the models yourself.

This repo does that on **one 32 GB Tesla V100 (Volta, 2017)**:

- **Models:** Qwen3, smallest first.
- **Measures:** perplexity and the paper's seven zero-shot tasks at each bit level.
- **Compared with:** the uncompressed model, and llama.cpp quants of the same model scored by the same code.

We are not affiliated with Samsung. All method credit goes to the LittleBit authors (see [Credit](#credit-and-licence)).

## Results: Qwen3-0.6B

| weights | bits/weight (body) | wikitext2 PPL | c4 PPL | zero-shot avg (7 tasks) | trained on |
| :--- | ---: | ---: | ---: | ---: | :--- |
| fp16 (uncompressed) | 16 | 20.95 | 25.39 | 48.4% | -- |
| llama.cpp Q4_K_M | 4.78 | 21.77 | 26.65 | 45.6% | -- |
| llama.cpp Q3_K_M | 3.87 | 24.46 | 28.28 | 45.5% | -- |
| llama.cpp Q2_K | 2.95 | 39.84 | 42.54 | 41.9% | -- |
| llama.cpp IQ2_XXS | 2.11 | 621.8 | 507.2 | 36.1% | -- |
| llama.cpp IQ1_M | 1.87 | 1,438.5 | 1,651.7 | 34.2% | -- |
| llama.cpp IQ1_S | 1.72 | 4,071.7 | 3,738.9 | 32.8% (fp32) | -- |
| **LittleBit-2, 1.0 bit** | **1.02** | **37.64** | **44.16** | **38.6%** | 24,000 sequences (49M tokens), 5 h 40 min |
| LittleBit-2, 1.0 bit, init only | 1.02 | 17,139,636 | | | 0 tokens |
| *chance* | | | | *35.7%* | |

Zero-shot accuracy per task (plain `acc`, 0-shot):

| weights | boolq | piqa | hellaswag | winogrande | arc_easy | arc_challenge | openbookqa |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 64.3 | 67.8 | 37.6 | 56.0 | 60.9 | 31.3 | 21.0 |
| llama.cpp Q2_K (2.95 bits) | 58.4 | 61.6 | 32.3 | 51.7 | 47.8 | 22.3 | 18.8 |
| LittleBit 1.0 bit | 62.2 | 58.5 | 27.2 | 52.7 | 38.0 | 17.6 | 13.6 |
| chance | 50 | 50 | 25 | 50 | 25 | 25 | 25 |

What this says:

- **At 1 bit, LittleBit matches llama.cpp's Q2_K on perplexity with a third of the bits.**
  - WikiText-2: 37.6 vs 39.8. C4: 44.2 vs 42.5.
  - Every llama.cpp quant at about 2 bits or below is broken on this model (perplexity 600 to 4,000).
- **It keeps much less knowledge than its perplexity suggests.**
  - The zero-shot average is 38.6%, against 35.7% for chance and 41.9% for Q2_K.
  - boolq's 62.2% equals the share of "yes" answers in its test set, the score you would get by always
    answering yes.
  - arc_challenge (17.6) and openbookqa (13.6) fall *below* chance. Plain 0-shot accuracy on openbookqa is low
    for the original model too (21.0).
  - The model learned to produce fluent text again faster than it recovered facts and reasoning.
- **Training was short.** This used 6% of the upstream recipe's ~850M tokens, and perplexity was still falling at
  the end (see the curve below). These numbers are a floor for LittleBit on this model, not its ceiling.
- **A 0.6B model is a hard test.** It has little redundancy, and the paper reports 7B+ models. Q2_K already loses
  6.5 zero-shot points here.
- **The pipeline checks out.** The saved, bit-packed model scores 37.638 on WikiText-2. During training the same
  weights scored 37.634 (the difference is fp16 vs fp32 scales).

How these were measured:

- **Same scorer for every row.** Perplexity uses LittleBit's own `eval.py` at sequence length 2048 (`results/*/eval-*.json`).
  The zero-shot tasks are the paper's seven: boolq, piqa, hellaswag, winogrande, arc_easy, arc_challenge, openbookqa
  (lm-eval 0.4.8, 0-shot, the full test sets). `python3 scripts/summarize.py` rebuilds the table from the JSON files.
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

### Learning curve (night 1, 1.0 bit, lr 1.6e-4)

WikiText-2 perplexity during training, on the same windows as `eval.py` (`ppl_curve.jsonl`). The learning rate
follows a cosine to zero over the whole run.

| step | sequences seen | tokens | wikitext2 PPL |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0.0M | 17,139,636.0 |
| 500 | 2,000 | 4.1M | 640.9 |
| 1000 | 4,000 | 8.2M | 100.2 |
| 1500 | 6,000 | 12.3M | 72.5 |
| 2000 | 8,000 | 16.4M | 67.2 |
| 2500 | 10,000 | 20.5M | 53.6 |
| 3000 | 12,000 | 24.6M | 50.6 |
| 3500 | 14,000 | 28.7M | 46.8 |
| 4000 | 16,000 | 32.8M | 42.9 |
| 4500 | 18,000 | 36.9M | 40.4 |
| 5000 | 20,000 | 41.0M | 39.9 |
| 5500 | 22,000 | 45.1M | 38.2 |
| 6000 | 24,000 | 49.2M | 37.6 |

### What 1 bit buys on disk (`scripts/sizes.py`)

The saved 1.0-bit model (`results/qwen3-0.6b-b1.0-long/sizes.json`) is 647 MiB. Qwen3-0.6B's own fp16 file is 1.5 GB.

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
Runs here are much shorter, and each row says what it was trained on. Night 1 was 24,000 sequences (49M tokens, 5 h 40 min).

### Continuing a finished model (warm start, 2026-10-12)

Night 1's model was saved bit-packed, so its latent values were gone. The question was whether training can continue
from its signs and scales alone (`--warm_start_from`, patch item 9). It can.

- **The test:** 400 more steps on 1,600 new sequences (seed 43), lr 4e-5, a fresh optimizer, latents restarted at
  ±0.03 (`scripts/warm_test.sh`, `results/qwen3-0.6b-b1.0-warm/`).
- **The result:** perplexity held and edged down. WikiText-2 went 37.64 → 37.52, C4 44.16 → 44.07. Nothing
  scrambled, so a finished model is a usable starting point for more training.
- **The catch:** at this learning rate the gain is small, about 0.1 per 400 steps. Big further gains need long
  runs, which now save resumable checkpoints (patch item 8).

| step | wikitext2 PPL |
| ---: | ---: |
| 0 (night 1's model) | 37.64 |
| 100 | 37.63 |
| 200 | 37.52 |
| 300 | 37.55 |
| 400 | 37.52 |

## Results: Qwen3-1.7B

| weights | bits/weight (body) | wikitext2 PPL | c4 PPL | zero-shot avg (7 tasks) | trained on |
| :--- | ---: | ---: | ---: | ---: | :--- |
| fp16 (uncompressed) | 16 | 16.67 | 19.21 | 56.8% | -- |
| llama.cpp Q4_K_M | 4.80 | 16.95 | 19.43 | 55.4% | -- |
| llama.cpp Q3_K_M | 3.85 | 17.52 | 20.28 | 53.3% | -- |
| llama.cpp Q2_K | 2.93 | 22.59 | 24.24 | 49.8% | -- |
| llama.cpp IQ2_XXS | 2.10 | 46.87 | 56.32 | 41.7% | -- |
| llama.cpp IQ1_M | 1.84 | 90.99 | 105.14 | 35.4% | -- |
| llama.cpp IQ1_S | 1.68 | 248.94 | 226.05 | 35.5% | -- |
| **LittleBit-2, 1.0 bit** | **1.01** | **27.04** | **33.14** | **40.9%** | 24,000 sequences (49M tokens), 8-bit AdamW |
| **LittleBit-2, 1.0 bit, + A100 continuation** | **1.01** | **25.46** | **31.21** | **41.4%** | + 27,200 new sequences (56M tokens) on a rented A100 |
| *chance* | | | | *35.7%* | |

Zero-shot accuracy per task (plain `acc`, 0-shot):

| weights | boolq | piqa | hellaswag | winogrande | arc_easy | arc_challenge | openbookqa |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16 | 77.5 | 72.4 | 46.2 | 60.9 | 72.2 | 39.9 | 28.4 |
| llama.cpp Q2_K (2.93 bits) | 71.6 | 66.9 | 40.1 | 54.8 | 62.2 | 32.0 | 20.8 |
| llama.cpp IQ2_XXS (2.10 bits) | 62.8 | 60.9 | 31.7 | 53.9 | 44.2 | 22.2 | 16.2 |
| LittleBit 1.0 bit | 62.0 | 62.7 | 30.3 | 51.3 | 44.1 | 19.5 | 16.4 |
| LittleBit 1.0 bit + A100 continuation | 62.1 | 63.3 | 30.8 | 52.2 | 44.9 | 20.1 | 16.0 |

What this says:

- **Fluency: 1 bit lands between llama.cpp's 2- and 3-bit quants.**
  - Perplexity 27.0 is 1.62x the original. The 0.6B was at 1.80x, so the bigger model held up better.
  - It's far better than llama.cpp at 1.7-2.1 bits (47-249).
  - It no longer matches Q2_K (22.6 at 2.93 bits), as it did on the 0.6B. llama.cpp's quants also improve with size.
- **Knowledge: the 1-bit model ties IQ2_XXS task by task, at half the bits.**
  - It keeps 5.2 of the original's 21.1 points above chance (25%). The 0.6B kept 23%.
  - Q2_K keeps 67%; IQ1_M keeps none.
- **Size: the compressed layers are 1.011 bits per weight** (1,409M weights, 170 MiB instead of 2.69 GB in fp16).
  - The whole file is 1.36 GB against the original's 3.8 GB.
  - At this size the embeddings and lm_head (594 MiB each; tied, so one copy could go) are most of it.
- **Training:**
  - fp16 with the overflow fix (patch item 11); NaN from step ~1,770 before it, none after.
  - 8-bit AdamW (patch item 10) to fit 32 GB; batch 1 x 4 accumulation, lr 1.6e-4.
  - Resumed twice from saved checkpoints.
  - About 12 h of training (6.5 s/step before the fix, 7.6 after), plus ~1 h lost to the NaN.
- **The fp16 round trip** of the llama.cpp path reproduces the original again (16.6704 / 19.2139).

Learning curve (WikiText-2 during training; `results/qwen3-1.7b-b1.0/ppl_curve.jsonl`; the pre-fix curve with its NaN
point is kept as `ppl_curve.before-fix.jsonl`):

| step | wikitext2 PPL | vs original |
| ---: | ---: | ---: |
| 500 | 497.0 | 29.81x |
| 1000 | 69.0 | 4.14x |
| 1500 | 54.6 | 3.28x |
| 2000 | 55.6 | 3.34x |
| 2500 | 41.9 | 2.51x |
| 3000 | 37.0 | 2.22x |
| 3500 | 34.1 | 2.05x |
| 4000 | 31.7 | 1.90x |
| 4500 | 29.7 | 1.78x |
| 5000 | 28.9 | 1.73x |
| 5500 | 27.8 | 1.67x |
| 6000 | 27.0 | 1.62x |

### Training further on a rented A100 (2026-10-13)

The question: is the 1-bit 1.7B limited by training time, or by the 1-bit budget itself?

**What ran:**
- The finished model was continued with `--init_from_checkpoint` (exact latents, fresh learning-rate schedule).
- It saw 27,200 sequences the V100 run never saw (`--skip_train_samples 32000`), for 6,800 steps.
- Hardware: a Thunder Compute A100 80GB, in bf16, with plain AdamW, batch 2 x 2, peak lr 2e-5 (cosine).
  ITQ init was off: that GPU is network-attached and ITQ took 23 min there.
- 5 h 11 min of training, about $8.60 of a $10 credit.
- A guard script copied the results home and deleted the instance when training finished.
- A first try at peak lr 8e-5 made it worse (27.0 -> 31.9 -> 38.2 by step 2,000) and was stopped
  (`results/qwen3-1.7b-b1.0-a100cont-lr8e-5-stopped/`).

| step | wikitext2 PPL |
| ---: | ---: |
| 0 (the finished V100 model) | 27.04 |
| 1000 | 28.73 |
| 2000 | 28.81 |
| 3000 | 28.40 |
| 4000 | 27.15 |
| 5000 | 27.09 |
| 6000 | 26.00 |
| 6800 | 25.47 |

The restart first costs ground, then the low-learning-rate tail recovers it and goes further. Most of the gain came
in the last 1,800 steps, as in the first run.

**Evaluating it:** the A100 saved bf16 weights, which the V100 runs about 9x slower. So the stored bf16 tensors were
converted to fp16 for evaluation:
- none exceed the fp16 range;
- 582 of 625M values become 0, almost all in the embeddings.

The converted model scores 25.46 on WikiText-2 on the V100, matching the A100's own 25.47, so the conversion is
faithful. The original bf16 file is kept.

**Result:**
- **Fluency improved 6%:** 27.04 -> 25.46 (WikiText-2) and 33.14 -> 31.21 (C4).
- **Knowledge barely moved:** 40.9% -> 41.4%.
  - 6 of 7 tasks rose by 0.1-0.9, within each task's ~1-1.4 point standard error, and openbookqa fell 0.4.
  - It keeps 27% of the original's above-chance skill, against 25% before.
- **At this size, more training mostly buys fluency, not knowledge.** Model capacity at 1 bit, rather than training
  time, looks like the limit.

### Where does 1 bit do its damage? Swapping original blocks back in (2026-10-18)

The question: is the 1-bit model's damage in a few places that could be kept at higher precision? The embeddings,
`lm_head` and norms are already fp16 in these models (LittleBit only compresses the 7 linear layers in each of the 28
blocks), so the test is on the blocks: chosen layers of the A100-continued 1-bit model are swapped back to the
**original** fp16 weights at inference time, with no training
([`scripts/blockswap/blockswap.py`](scripts/blockswap/blockswap.py)). Scored on WikiText-2 (the first 40 windows of
2048 tokens, LittleBit's own eval code), with three greedy sample paragraphs per setting
([`results/blockswap-1.7b/samples.md`](results/blockswap-1.7b/samples.md)).

| swapped back to the original fp16 | WikiText-2 PPL (40 windows) |
| --- | ---: |
| nothing (1-bit as trained) | **23.70** |
| everything (the original model) | 15.23 |
| block 0 / last block (27) / both | 25.90 / 40.56 / 44.44 |
| blocks 0-3 / 24-27 | 26.31 / 29.67 |
| every attention layer | 8,525 |
| every MLP layer | 2,339,954 |

Each block swapped back on its own:

| blocks | PPL with that block back | |
| --- | ---: | --- |
| 1 | 170.3 | rewritten the most |
| 0, 2-7 | 25.9-37.6 | heavily rewritten |
| 8-11 | 24.8-25.4 | |
| 12-26 | 23.9-24.6 | close to the original |
| 27 (last) | 40.6 | heavily rewritten |

- **No block helps.** Every swap scores worse than the 1-bit model as trained (the best, block 13, is 23.94), and
  swapping whole groups breaks it outright. The trained 1-bit model is not the original plus noise: its blocks were
  re-learned to fit each other, and an original block no longer fits its neighbours.
- **Training rewrote the start and the end.** Blocks 0-7 and the last block are where an original block fits worst;
  the middle (12-26) barely minds the swap.
- **The samples read the same way:** the 1-bit model writes on-topic sentences but falls into loops sooner than the
  original (which also loops under greedy decoding); the big swaps turn it into "the a a the, the,,,,".
- **Next:** train a 1-bit model with blocks 0-7 and 27 kept at 2 bits from the start (about +7% in size), with the same
  data, seed and learning rate, and compare at the same steps.

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
8. **Adds resumable checkpoints: `--save_every N` and `--resume_from <checkpoint-N | latest>`.**
   - A finished LittleBit model keeps only the *signs* of its latent factors, bit-packed. The latent values are what
     SmoothSign trains, so a training checkpoint must keep them, along with the optimizer state. Checkpoints therefore
     skip the packing (`LittleBitLinear.pack_on_save`). The final model is still packed.
   - A checkpoint is 5.0 GB for Qwen3-0.6B.
   - `scripts/test_resume.sh` runs 8 steps, then resumes a second run from step 4. Steps 5-8 match the uninterrupted
     run: step 5 is identical, steps 6-8 are within 0.1% (`results/test-resume/`). The test fails without the patch.
9. **Adds `--warm_start_from <finished model>`:** continues from a packed model, using its signs and scales. The
   latent values are lost, so each one restarts at ±`--warm_mag` (0.03). SmoothSign's gradient only reaches latents
   with |x| below ~0.05. The forward pass uses only signs and scales, so step 0 reproduces the finished model.
10. **Adds `--optim`** (default `adamw_torch`, as upstream).
    - **Why:** Qwen3-1.7B at 1 bit has 1.36B trainable latent weights, and plain AdamW's state for them is 11 GB.
      That ran out of memory on a 32 GB card at the second step.
    - **Fix:** `adamw_bnb_8bit` (bitsandbytes 0.50, which works on Volta) keeps that state in 8 bits, about 2.7 GB.
      The 1.7B then peaks at 26.9 GB.
    - **Checked against plain AdamW:** same 0.6B probe, same seed (`results/probe-lr1.6e-4-adam8`). The loss
      curves are nearly identical; final perplexity was 3,291 vs 4,008 at that early stage.
11. **Makes the 1-bit layer itself safe from fp16 overflow** (`LittleBitLinear._compute_forward_fp16`).
    - **The failure:** the Qwen3-1.7B run trained well for ~1,770 steps (perplexity 54.6 at step 1,500), then every
      batch came back NaN, and the loss scale halved itself to zero.
    - **The weights were fine:** the checkpoints had no NaN, and in fp32 the model scored a healthy loss of 3.63.
    - **The cause:** in fp16, the last layer's `mlp.down_proj` overflowed (`scripts/find_overflow.py`). Each binary
      matmul adds up to 6,144 terms *before* the learned scales shrink the result. Those running sums passed 65,504,
      even though the layer's output (31k) would have fit.
    - **The fix:** each matmul input is divided by the power of two that makes overflow impossible
      (|sum| ≤ terms × max|input|), and the output is multiplied back. Powers of two are exact.
    - **The results:** the broken checkpoint scores 3.6323 in fp16 against 3.6324 in fp32. Night 1's 0.6B model is
      unchanged: 37.6377 vs 37.6381 on WikiText-2.
    - **The scope:** it only runs when the matmul would be fp16, so upstream's bf16/fp32 paths are untouched.
12. **Adds `--init_from_checkpoint <checkpoint-N>` and `--skip_train_samples N`.**
    - `--init_from_checkpoint` continues training from a resumable checkpoint's exact latent weights, with a fresh
      optimizer and a new lr schedule.
    - `--skip_train_samples` skips the sequences a previous run already saw, in the same seeded shuffle.
    - Checked: starting from the 0.6B warm run's checkpoint-400, step 0 scores 37.5217 against the checkpoint's own
      37.5219, with 2,465 trainable tensors loaded and 0 missing.
13. **`eval.py --dtype`** picks fp16/fp32 explicitly. `--results_json` writes the numbers to a file. The perplexity
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

## Closing note (2026-10-19)

The last run here gave Qwen3-1.7B's most-damaged blocks (0-7 and 27, found by the block swap above) 2 bits from the
start and kept the rest at 1 bit, with the same recipe, data and seed as the 1-bit run
(`--eff_bit_layers "0-7,27:2.0"`, [patches/mixed-bits.patch](patches/mixed-bits.patch),
[scripts/run-1.7b-mix.sh](scripts/run-1.7b-mix.sh), [results/qwen3-1.7b-b1.0-mix2](results/qwen3-1.7b-b1.0-mix2)):

| | file | body bits/weight | WikiText-2 | C4 | zero-shot avg (7 tasks) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-1.7B, fp16 | 4.1 GB | 16 | 16.7 | 19.2 | 56.8% |
| mixed (blocks 0-7 + 27 at 2 bits) | 1.48 GB | 1.33 | **23.4** | **29.0** | **42.8%** |
| 1-bit | 1.42 GB | 1.01 | 27.0 | 33.1 | 40.9% |
| *chance* | | | | | *35.7%* |

The mixed model beats plain 1-bit on every measure for a 4% bigger file, because the embeddings and the LM head are
most of the file either way. But it still recovers only about a third of the full model's above-chance skill, so
none of these compressed models is usable for anything. The same Qwen3-1.7B at ordinary 4-bit precision handled 96%
of a phone-command test after fine-tuning, so at this size 1-bit simply costs too much. Each run also takes about 14
hours on a V100, so this series ends here. The code and the results stay for anyone who wants to try the method.
