"""Builds the phone-assistant training set for LittleBit's distillation (2026-10-13): one command per row, in exactly
the chat format the phone model was trained on (phone-assistant scripts/train_lora.py: SHORT_PROMPT system line,
thinking off, the JSON answer, <|im_end|>), right-padded to one length. Saved where main.py's prepare_dataset() looks
for --dataset phone, so the trainer needs no other change.

One row per command, not packed: packed rows would put earlier commands in front of each one, and the quiz asks each
command alone. LittleBit's distillation loss has no mask, so pad positions are matched too (teacher and student see the
same pads); the rows are padded only to the longest command to keep that share small.

Run from the LittleBit checkout:  prep_phone.py --model_id <merged phone model> --data_root <dir> --pa_root <phone-assistant>
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")                                     # run from the LittleBit checkout
import utils.datautils as du                                # noqa: E402
from datasets import Dataset                                # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--model_id", required=True)
p.add_argument("--data_root", required=True)
p.add_argument("--pa_root", default="/mnt/steam/phone-assistant")
a = p.parse_args()

sys.path.insert(0, a.pa_root)
from prompt import SHORT_PROMPT                             # noqa: E402

tok = du.load_tokenizer(a.model_id)
rows = [json.loads(line) for line in open(Path(a.pa_root) / "data" / "train.jsonl")]
seqs = []
for r in rows:
    msgs = [{"role": "system", "content": SHORT_PROMPT}, {"role": "user", "content": r["say"]}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    seqs.append(tok(prompt + json.dumps(r["answer"]) + tok.eos_token, add_special_tokens=False)["input_ids"])
L = max(len(s) for s in seqs)
pad = tok.pad_token_id
data = {"input_ids": [s + [pad] * (L - len(s)) for s in seqs],
        "attention_mask": [[1] * len(s) + [0] * (L - len(s)) for s in seqs],
        "labels": [s + [-100] * (L - len(s)) for s in seqs]}
real = sum(len(s) for s in seqs)
print(f"{len(seqs)} commands, padded to {L} tokens; {real} real tokens = {100 * real / (L * len(seqs)):.0f}% of the rows")

shutil.rmtree(Path(a.data_root) / "phone", ignore_errors=True)     # always rebuilt from train.jsonl
du.get_qat_dataset = lambda name, tokenizer: Dataset.from_dict(data)
a.dataset = "phone"
ds = du.prepare_dataset(a, tok)
print(f"saved: {len(ds)} rows of {len(ds[0]['input_ids'])} tokens under {a.data_root}/phone/")
