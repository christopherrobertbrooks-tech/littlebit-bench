# LittleBit on a Tesla V100: how much quality is left below 1 bit per weight?

**Status (2026-10-11): in progress.** Setup, patch and the uncompressed baselines are done. Training runs are going.

[LittleBit](https://github.com/SamsungLabs/LittleBit) (Samsung Labs, NeurIPS 2025) and LittleBit-2 (ICML 2026) compress
an LLM's weights to **1.0 down to 0.1 bits per weight**. Each weight matrix becomes two low-rank factors stored as
±1 signs, plus small learned scales. Quality is recovered with quantization-aware training (QAT), using the original model
as a teacher. Samsung hasn't released checkpoints, so the only way to see the quality loss is to train the models yourself.

This repo does that on **one 32 GB Tesla V100 (Volta, 2017)**:

- **Models:** Qwen3, smallest first.
- **Measures:** perplexity and the paper's seven zero-shot tasks at each bit level.
- **Compared with:** the uncompressed model.

We are not affiliated with Samsung. All method credit goes to the LittleBit authors (see [Credit](#credit-and-licence)).

## Results

| model | weights | wikitext2 PPL | c4 PPL | zero-shot avg (7 tasks) | trained on |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Qwen3-0.6B | fp16 (uncompressed) | 20.96 | | | -- |

Perplexity uses LittleBit's own `eval.py` at sequence length 2048. The zero-shot tasks are boolq, piqa, hellaswag,
winogrande, arc_easy, arc_challenge and openbookqa (lm-eval 0.4.8, 0-shot).

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
6. **`eval.py --dtype`** picks fp16/fp32 explicitly. `--results_json` writes the numbers to a file. The perplexity loss
   is computed in fp32.

The defaults are unchanged. Without the new flags, upstream's bf16 behaviour is the same.

## Reproduce

```bash
./setup.sh                                   # clone LittleBit @42d658b, apply the patch, Python 3.12 venv (torch 2.6 cu124)
scripts/eval.sh qwen3-0.6b-fp Qwen/Qwen3-0.6B fp16 fp          # uncompressed baseline
SAMPLES=2000 scripts/train.sh qwen3-0.6b-b1.0 Qwen/Qwen3-0.6B 1.0
scripts/eval.sh qwen3-0.6b-b1.0 /mnt/steam/littlebit/runs/qwen3-0.6b-b1.0 fp16
```

`scripts/env.sh` pins the run to the V100 (`CUDA_VISIBLE_DEVICES=1` in PCI order on this host; override with `LB_GPU`).

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
