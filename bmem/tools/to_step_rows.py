#!/usr/bin/env python
"""Episode-level multiturn SFT rows -> one row per assistant turn (user_i, assistant_i), matching the eval harness,
which sends every step as an independent prompt. Rewriter rows are already single-turn and pass through."""
import argparse, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--in-dir", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--inline", action="store_true", help="tool-call style: merge query + memory + continuation into ONE assistant message (memory span marked no-loss)")
a = ap.parse_args()
NOLOSS_START, NOLOSS_END = "<<<BMEM_NOLOSS>>>", "<<<BMEM_ENDNOLOSS>>>"   # consumed by verl multiturn_sft_dataset
import os; os.makedirs(a.out_dir, exist_ok=True)
for split in ("train", "val"):
    d = pd.read_parquet(f"{a.in_dir}/{split}.parquet"); rows = []
    for _, r in d.iterrows():
        ms = list(r.messages)
        if r.data_kind != "agent": rows.append({"messages": ms, "item_id": r.item_id, "data_kind": r.data_kind, "turn_kind": "rewriter", "traj_uid": r.traj_uid}); continue
        i = 0
        while i < len(ms) - 1:
            u, o = ms[i]["content"], ms[i + 1]["content"]
            if "<query>" in o and i + 3 < len(ms) and str(ms[i + 2]["content"]).startswith("<memory>"):
                # chat continuation: one sample = [user obs, assistant belief+query, user memory, assistant think+action] (both assistant parts trained once)
                if a.inline:   # same text the policy sees at inference: prefill = own belief+query + "\n" + memory + "\n", then it generates think+action
                    merged = str(o).strip() + NOLOSS_START + "\n" + str(ms[i + 2]["content"]).strip() + "\n" + NOLOSS_END + str(ms[i + 3]["content"]).strip()
                    msgs_ = [ms[i], {"role": "assistant", "content": merged}]
                else:
                    msgs_ = [ms[i], ms[i + 1], ms[i + 2], ms[i + 3]]
                rows.append({"messages": msgs_, "item_id": r.item_id, "data_kind": "agent", "turn_kind": "query+continuation", "traj_uid": r.traj_uid, "step_idx": i // 2})
                i += 4; continue
            kind = "continuation" if "Your belief and query at this step" in u else ("query" if "<query>" in o else "decision")
            rows.append({"messages": [ms[i], ms[i + 1]], "item_id": r.item_id, "data_kind": "agent", "turn_kind": kind, "traj_uid": r.traj_uid, "step_idx": i // 2})
            i += 2
    out = pd.DataFrame(rows); out.to_parquet(f"{a.out_dir}/{split}.parquet", index=False)
    print(split, len(d), "rows ->", len(out), "|", out.turn_kind.value_counts().to_dict())
