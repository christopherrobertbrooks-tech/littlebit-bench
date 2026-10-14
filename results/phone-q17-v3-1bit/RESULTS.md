# The phone assistant's model at 1 bit (2026-10-13)

Teacher: q17-v3 (Qwen3-1.7B + the phone assistant's LoRA, merged, fp16). Student: LittleBit 1.0 bit, warm start from
the general 1-bit Qwen3-1.7B (the A100 continuation), distilled from the teacher on the phone model's own 4,315 training
commands (one per row, 16 per step, 8-bit AdamW). Stage 1: 3 epochs at lr 1e-4 (810 steps, 21 min on the V100).
Stage 2: 3 more at 5e-5 (21 min). Scored as dense f16 GGUFs (scripts/phone/export_dense.py: top token 55/55 against the
1-bit model) through llama-server with the action grammar, the phone assistant's one-line instruction. Exact = action
and every detail right. "+checker" = the app's checker applied to the same answers (what the phone would show).

| model | bits/weight | main 333 | +checker | Chris 31 | fresh 40 | Muse 50 |
|---|---|---|---|---|---|---|
| teacher, f16 | 16 | 322 (96.7%) | 323 | 31 | 39 | 45 |
| llama.cpp Q2_K | ~2.9 | 302 (90.7%) | 309 | 30 | 34 | 40 |
| **LittleBit 1-bit, 6 epochs** | **~1.0** | **253 (76.0%)** | **262** | **26** | **26** | **35** |
| LittleBit 1-bit, 3 epochs | ~1.0 | 246 (73.9%) | 255 | 25 | 24 | 34 |
| llama.cpp IQ2_XXS | ~2.1 | 223 (67.0%) | 235 | 26 | 26 | 28 |
| llama.cpp IQ1_M | ~1.75 | 134 (40.2%) | 146 | 14 | 14 | 22 |
| LittleBit 1-bit, step 0 (no phone training) | ~1.0 | 0 | 0 | 0 | 0 | 0 |
| for scale: q06-v3 (Qwen3-0.6B + LoRA) Q4_K_M, 397 MB | ~4.8 | 94% | | | | |

The phone assistant (its model, training commands and quizzes) is a private project, so this run cannot be
reproduced from this repo alone; scripts/phone/ shows exactly how it was done.

llama.cpp quants: imatrix from the same 4,315 training commands (runs/q17-v3/lowbit/make.sh on the gateway).

- 1 bit learns the job fast (0 -> 74% in 21 min) and then plateaus (loss flat over epochs 2-6; stage 2 +2 points).
- Its misses are mostly numbers and times (7 p.m. -> 17:00, an hour and a half -> 1 h 10 min); 40 of the stage-1
  misses had the right action with a wrong detail.
- Against llama.cpp on the same model: +9 points over IQ2_XXS and +36 over IQ1_M at about half their bits; Q2_K at
  ~2.9 bits is 15 points better.
- For the phone it does not pay: the 4-bit 0.6B phone model is 397 MB at 94%; the 1-bit 1.7B would be roughly the same
  size once its 0.31B-parameter word table is stored at 4 bits (body ~170 MiB + ~175 MB, an estimate), at 76%.
  Kernel trigger 2 (a 1-bit phone model near 90%) not met.
