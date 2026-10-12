"""Builds LittleBit's c4_wiki training set for a tokenizer ahead of time (main.py would build it on first use)."""
import argparse
import sys

sys.path.insert(0, ".")                                     # run from the LittleBit checkout
from utils.datautils import load_tokenizer, prepare_dataset

p = argparse.ArgumentParser()
p.add_argument("--model_id", required=True)
p.add_argument("--data_root", required=True)
p.add_argument("--dataset", default="c4_wiki")
a = p.parse_args()
ds = prepare_dataset(a, load_tokenizer(a.model_id))
print(f"{a.dataset}: {len(ds)} samples of {len(ds[0]['input_ids'])} tokens")
