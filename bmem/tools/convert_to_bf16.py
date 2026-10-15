#!/usr/bin/env python
"""Cast a verl fsdp_sft_trainer HF save (fp32 shards) to bf16 for vLLM/eval. Usage: python bmem/tools/convert_to_bf16.py <src_dir> <dst_dir>"""
import sys, os, json, shutil, glob, torch
from safetensors.torch import load_file, save_file
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if not f.endswith(".safetensors") and f != "model.safetensors.index.json": shutil.copy(os.path.join(src, f), dst)
weight_map = {}; shard_id = 0; total = 0
shards = sorted(glob.glob(os.path.join(src, "*.safetensors")))
for p in shards:
    sd = {k: v.to(torch.bfloat16) for k, v in load_file(p).items()}
    shard_id += 1; name = f"model-{shard_id:05d}-of-{len(shards):05d}.safetensors"
    save_file(sd, os.path.join(dst, name), metadata={"format": "pt"})
    for k, v in sd.items(): weight_map[k] = name; total += v.numel() * 2
    print("wrote", name, len(sd), "tensors", flush=True)
json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
cfg = json.load(open(os.path.join(dst, "config.json"))); cfg["torch_dtype"] = "bfloat16"; json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
print("done ->", dst, "params(bf16 bytes)=", total)
