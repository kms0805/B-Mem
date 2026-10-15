#!/usr/bin/env python
"""Teacher rollout dumps (scripts/collect_teacher_data.sh) -> SFT parquet (agent rows + skill-adaptation rows from the same episodes).
Agent rows: won, every turn protocol-clean, <= --max-queries queries, one episode per game.
Adaptation rows: every adaptation call of a kept episode (BMEM_ADAPTOR_DUMP jsonl), output kept verbatim (<assessment>/<guidance>).
Usage: python bmem/tools/traj_to_sft.py --exps 'outputs/eval/teacher_bmem_*' --variant full --max-queries 99 --out-dir data/sft/bmem"""
import argparse, glob, json, os, random, re, sys
from collections import Counter
import pandas as pd
TASKS = ["pick_and_place", "pick_two_obj", "look_at_obj", "pick_clean", "pick_heat", "pick_cool"]
def ttype(s):
    for t in TASKS:
        if t in s: return t
    return "?"
def user_content(t):   # rendered prompt -> user turn text (between the chat-template user and assistant markers)
    i = t.find("\nuser\n"); t = t[i + 6:] if i >= 0 else t
    j = t.rfind("\nassistant"); return t[:j] if j >= 0 else t
B = re.compile(r"<belief>(.*?)</belief>", re.S); T = re.compile(r"<think>(.*?)</think>", re.S); A = re.compile(r"<action>(.*?)</action>", re.S); Q = re.compile(r"<query>(.*?)</query>", re.S)
RW = re.compile(r"^(?:<applicable>.+?</applicable>\s*<useful>.+?</useful>|<assessment>.+?</assessment>|(?:<judge_\d+>.+?</judge_\d+>\s*)+)\s*<guidance>.+?</guidance>\s*$", re.S)   # v3.2 per-memory or joint rewriter
VARIANT = "full"
CHAT_CONT = re.compile(r"\nassistant\n.*?\nuser\n\s*(<memory>.*?)\s*\nassistant\s*$", re.S)
INLINE_CONT = re.compile(r"\nassistant\n(?:(?!\nuser\n).)*?</query>\s*(<memory>.*?</memory>)\s*$", re.S)   # BMEM_CHAT_CONT=inline: logged prompt ends inside the assistant message
def classify(o, cur_u):   # the continuation marker is in the continuation turn's OWN input
    nb, nt, na, nq = len(B.findall(o)), len(T.findall(o)), len(A.findall(o)), len(Q.findall(o))
    want_b = 0 if VARIANT == "nobelief" else 1
    if "Your belief and query at this step" in cur_u or "Your query at this step" in cur_u: return "continuation", nb == 0 and nt >= 1 and na == 1 and nq == 0
    if nq: return "query", nb == want_b and na == 0 and nq == 1 and nt == 0
    return "decision", nb == want_b and nt >= 1 and na == 1
ap = argparse.ArgumentParser(); ap.add_argument("--exps", nargs="+", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--rewriter-system", default="", help="file whose text replaces the rewriter rows' system message (teacher-only prompt devices such as a 1-shot example must not reach the student)"); ap.add_argument("--max-queries", type=int, default=5); ap.add_argument("--step-budget", type=int, default=10**9); ap.add_argument("--val-episodes", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--exclude-games", nargs="*", default=[]); ap.add_argument("--variant", default="full", choices=["full", "nobelief"]);
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); VARIANT = a.variant
RW_SYSTEM = open(a.rewriter_system).read().strip() if a.rewriter_system else ""
eps = []; rw = []
for pat in a.exps:
    for e in glob.glob(pat):
        eps += [dict(json.loads(l), _src="teacher") for l in open(e + "/val_traj/0.jsonl")]
        if os.path.exists(e + "/adaptor_io.jsonl"): rw += [json.loads(l) for l in open(e + "/adaptor_io.jsonl")]
print("episodes", len(eps), "| won", sum(float(r["episode_reward"]) >= 10 for r in eps), "| adaptor calls", len(rw))
drop = Counter(); kept = []; seen = set(); kinds = Counter(); qdist = Counter()
for r in eps:
    gf = r.get("episode_gamefile", ""); g = "/".join(gf.split("/")[-3:-1])
    if any(x in gf or x == r["traj_uid"] for x in a.exclude_games): drop["excluded_by_review"] += 1; continue   # game substring or exact traj_uid
    if float(r["episode_reward"]) < 10: drop["failed"] += 1; continue
    if g in seen: drop["duplicate_game"] += 1; continue
    ms = []; ok = True; prev = ""; nq = 0
    for t in r["turns"]:
        o = (t["output"] or "").strip()
        cm = CHAT_CONT.search(t["input"] or "") or INLINE_CONT.search(t["input"] or "")   # chat continuation (BMEM_CHAT_CONT): ...assistant\n<belief+query>\nuser\n<memory>...\nassistant
        if cm:
            cb, ct, ca, cq = len(B.findall(o)), len(T.findall(o)), len(A.findall(o)), len(Q.findall(o))   # own names: nq is the episode's query count
            k, good = "continuation", (cb == 0 and ct >= 1 and ca == 1 and cq == 0)
            if not good: ok = False; drop[f"bad_{k}"] += 1; break
            _mem = cm.group(1); _e = _mem.find("</memory>"); _mem = _mem[: _e + len("</memory>")] if _e >= 0 else _mem   # memory block only (drop the teacher-side continue note)
            kinds[k] += 1; ms += [{"role": "user", "content": _mem.strip()}, {"role": "assistant", "content": o}]; continue
        u = user_content(t["input"]); k, good = classify(o, u)
        if not good: ok = False; drop[f"bad_{k}"] += 1; break
        kinds[k] += 1; nq += k == "query"; ms += [{"role": "user", "content": u}, {"role": "assistant", "content": o}]; prev = u
    if not ok: continue
    if nq > a.max_queries: drop["too_many_queries"] += 1; continue
    if int(r["episode_length"]) + nq > a.step_budget: drop["over_step_budget_when_queries_count"] += 1; continue   # training harness counts queries as steps
    seen.add(g); qdist[nq] += 1
    kept.append({"messages": ms, "item_id": g, "data_kind": "agent", "task_type": ttype(g), "n_queries": nq, "n_env_steps": int(r["episode_length"]), "traj_uid": r["traj_uid"], "source": r.get("_src", "teacher")})
print("kept agent episodes", len(kept), "| dropped", dict(drop), "| query dist", dict(sorted(qdist.items())))
print("task mix", dict(Counter(k["task_type"] for k in kept)), "| source", dict(Counter(k["source"] for k in kept)))
uids = {k["traj_uid"] for k in kept}; rrows = []; rbad = 0
for x in rw:
    if x["traj_uid"] not in uids: continue
    out = (x["output"] or "").strip()
    if not RW.match(out): rbad += 1; continue
    _msgs = list(x["messages"])
    if a.rewriter_system and _msgs and _msgs[0].get("role") == "system": _msgs[0] = {"role": "system", "content": RW_SYSTEM}   # student rewriter prompt (standard), not the teacher's
    rrows.append({"messages": _msgs + [{"role": "assistant", "content": out}], "item_id": x["traj_uid"], "data_kind": "rewriter",
                  "rewriter_kind": "empty" if (out == "<EMPTY>" or re.search(r"<guidance>\s*<EMPTY>\s*</guidance>", out)) else "guidance", "traj_uid": x["traj_uid"]})
print("rewriter rows", len(rrows), "| malformed dropped", rbad, "|", dict(Counter(r["rewriter_kind"] for r in rrows)))
rng = random.Random(a.seed); val_uids = set(rng.sample(sorted(uids), min(a.val_episodes, len(uids))))
agent = pd.DataFrame(kept); rew = pd.DataFrame(rrows) if rrows else pd.DataFrame(columns=['messages', 'item_id', 'data_kind', 'rewriter_kind', 'traj_uid'])   # no-rewriter conditions produce no rows
tr = pd.concat([agent[~agent.traj_uid.isin(val_uids)], rew[~rew.traj_uid.isin(val_uids)]], ignore_index=True)
va = pd.concat([agent[agent.traj_uid.isin(val_uids)], rew[rew.traj_uid.isin(val_uids)]], ignore_index=True)
tr.to_parquet(a.out_dir + "/train.parquet", index=False); va.to_parquet(a.out_dir + "/val.parquet", index=False)
print("train rows", len(tr), "(agent", int((tr.data_kind == "agent").sum()), ") | val rows", len(va), "(agent", int((va.data_kind == "agent").sum()), ")")
