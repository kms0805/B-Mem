"""Shared prompt constants for the belief protocol (used by the cold-start builder AND the runtime patch).

Decision turn (policy):   <belief>...</belief>\n<think>...</think>\n<action>...</action>   or   <belief>...</belief>\n<query>...</query>
Continuation turn:        <think>...</think>\n<action>...</action>   (no new belief; same observation + memory block)
Rewriter (separate call): one short adapted guidance, or exactly <EMPTY>.

Keep these strings identical between SFT data and runtime prompts.
"""

QUERY_OPEN = "<query>"
QUERY_CLOSE = "</query>"
BELIEF_OPEN = "<belief>"
BELIEF_CLOSE = "</belief>"
EMPTY_MARK = "<EMPTY>"

# Appended to every decision-turn observation prompt (runtime: env.memory.retrieval_instruction_prompt with {open_tag}/{close_tag}).
BELIEF_INSTRUCTION = (
    "Before reasoning, write your belief about the current situation inside <belief> </belief> in at most three short sentences: "
    "(1) what you are holding and the state of the target object; (2) what is already done and what remains; "
    "(3) which places you have checked, plus any attempt that failed or a guess you have not verified. "
    "Replace your previous belief instead of copying it, and do not restate the room layout or the task. "
    "Do not turn \"it did not work this time\" into \"it never works\".\n"
    "After the belief, either reason inside <think> </think> and give exactly one admissible action inside <action> </action>, "
    "or, if you need past experience about a procedure or its preconditions, output exactly one short search query inside "
    "{open_tag} and {close_tag} instead of think and action. Never output both a query and an action in the same response."
)

# Inserted before BELIEF_INSTRUCTION from the second decision turn on.
PREVIOUS_BELIEF_BLOCK = "Your previous belief (from the last step):\n<belief>{belief}</belief>"
# BMEM_CHAT_CONT=1: the continuation is the SAME turn, interrupted by the memory: [user obs][assistant belief+query][user memory] -> think+action.
# The env packs the three parts into one text with these separators (so the rewriter can still patch the last <memory> block);
# the rollout loop unpacks them into chat messages.
CHAT_SEP_ASSISTANT = "\n<<<BMEM_ASSISTANT_TURN>>>\n"
CHAT_SEP_MEMORY = "\n<<<BMEM_MEMORY_TURN>>>\n"
CHAT_CONT_TAIL = "Now continue this turn: do not write a new belief; reason inside <think> </think>, then give exactly one admissible action inside <action> </action>."

# Runtime memory block (mem_adaptor.adapted_prompt_header / adapted_principle_label_template) and its empty fallback.
MEMORY_HEADER = "Relevant adapted past memories are provided below. Reuse them only if you think they are useful."
MEMORY_LABEL = "Adapted memory principle 1:"
MEMORY_BLOCK = "<memory>\n" + MEMORY_HEADER + "\n\n" + MEMORY_LABEL + "\n{guidance}\n</memory>"
EMPTY_RETRIEVAL_MESSAGE = "(No validated memory principle applies; rely on observation and reasoning.)"
EMPTY_MEMORY_BLOCK = "<memory>\n" + EMPTY_RETRIEVAL_MESSAGE + "\n</memory>"

# Continuation prompt suffix: same observation prompt + this block (replaces BELIEF_INSTRUCTION for that turn).
CONTINUATION_BLOCK = (
    "Your belief and query at this step:\n<belief>{belief}</belief>\n{open_tag}{query}{close_tag}\n\n"
    "{memory_block}\n\n"
    "Now choose one admissible action: reason inside <think> </think>, then give exactly one admissible action inside <action> </action>. "
    "Do not write a new belief in this response."
)

# Rewriter = base mem_adaptor system prompt + one sentence about the belief.
REWRITER_SYSTEM = (
    "You adapt a retrieved memory principle into a concise reusable guidance for the CURRENT situation and initial task, "
    "or output exactly <EMPTY> if the principle does not apply.\n"
    "Use the agent's belief to judge whether the memory's preconditions still hold. If the belief shows the step is already done "
    "or the precondition is contradicted, output exactly <EMPTY>. If a precondition is unverified, you may output one conditional check instead.\n"
    "Do not write chain-of-thought, first-person reasoning, or a step-by-step action plan."
)
REWRITER_USER_TEMPLATE = (
    "Initial task:\n{task}\n\n"
    "Current situation (state):\n{s_curr}\n\n"
    "Agent's current belief:\n{belief}\n\n"
    "Retrieved historical state:\n{s_old}\n\n"
    "Historical principle (memory):\n{p_old}\n\n"
    "Output rules:\n"
    "- Output exactly one short adapted principle (at most 25 words), or exactly <EMPTY>.\n"
    "- Write in imperative or neutral guidance style.\n"
    "- Do NOT use first person (\"I\", \"my\", \"we\") or phrases like \"therefore\", \"I need to\", \"since\".\n"
    "- Do NOT explain your reasoning or describe multiple next steps."
)

# Placeholder think used by the base cold-start data (AgentTraj-L has no thoughts).
EMPTY_THOUGHT_FALLBACK = "(no explicit reasoning in source; choose a valid admissible action.)"


# ---- Prompt variants for SFT ablations (2026-09-25). One function shared by the SFT data transform and the runtime
# harness (env var BMEM_PROMPT_VARIANT), so training and evaluation inputs stay identical.
#   full     : belief + query/memory (default, unchanged)
#   nobelief : query/memory, no belief (no belief instruction, no previous-belief block, continuation without belief)
#   plain    : neither -> the plain verl-agent ALFWorld prompt (no appended instruction)
import os as _os
import re as _re
NOBELIEF_INSTRUCTION = ("Reason inside <think> </think> and give exactly one admissible action inside <action> </action>, "
    "or, if you need past experience about a procedure or its preconditions, output exactly one short search query inside "
    "{open_tag} and {close_tag} instead of think and action. Never output both a query and an action in the same response.")
NOBELIEF_CONTINUATION_BLOCK = (
    "Your query at this step:\n{open_tag}{query}{close_tag}\n\n"
    "{memory_block}\n\n"
    "Now choose one admissible action: reason inside <think> </think>, then give exactly one admissible action inside <action> </action>."
)
_PREV_RE = _re.compile(r"\n\nYour previous belief \(from the last step\):\n<belief>.*?</belief>", _re.S)


# BMEM_PROMPT_VARIANT=plan: belief also carries the plan taken from retrieved memories (remaining procedure steps, places still worth trying)
PLAN_BELIEF_INSTRUCTION = BELIEF_INSTRUCTION.replace(
    "in at most three short sentences: ", "in at most four short sentences: ").replace(
    "a guess you have not verified. ",
    "a guess you have not verified; (4) the plan you are following from retrieved memories, if any: the remaining procedure steps and the places still worth trying. ")


def prompt_variant():
    return _os.environ.get("BMEM_PROMPT_VARIANT", "full")


def apply_prompt_variant(text, variant=None, open_tag=QUERY_OPEN, close_tag=QUERY_CLOSE):
    """Rewrite a decision-turn user prompt produced by the full belief harness into the given variant."""
    v = variant or prompt_variant()
    if v == "full":
        return text
    full_instr = BELIEF_INSTRUCTION.format(open_tag=open_tag, close_tag=close_tag)
    if v in ("nobelief", "plain"):
        text = _PREV_RE.sub("", text)
    repl = {"nobelief": NOBELIEF_INSTRUCTION.format(open_tag=open_tag, close_tag=close_tag), "plain": None,
            "plan": PLAN_BELIEF_INSTRUCTION.format(open_tag=open_tag, close_tag=close_tag)}[v]
    if full_instr in text:
        text = text.replace("\n\n" + full_instr, "" if repl is None else "\n\n" + repl) if ("\n\n" + full_instr) in text else text.replace(full_instr, repl or "")
    return text


# ---- WebShop belief instruction (2026-09-25). Same protocol as ALFWorld; only the three belief aspects are shopping-specific.
WEBSHOP_BELIEF_INSTRUCTION = (
    "Before reasoning, write your belief about the current situation inside <belief> </belief> in at most three short sentences: "
    "(1) which requirements of the instruction (product type, attributes, options such as size or color, price limit) the current or best candidate product satisfies, and which are still unverified; "
    "(2) what is already done (searched, product opened, options selected) and what remains before buying; "
    "(3) which searches or products you have already tried and why they were rejected, plus any guess you have not verified. "
    "Replace your previous belief instead of copying it, and do not restate the instruction or the page contents. "
    "Do not turn \"it did not work this time\" into \"it never works\".\n"
    "After the belief, either reason inside <think> </think> and give exactly one admissible action inside <action> </action>, "
    "or, if you need past experience about how to search, choose, or verify a product, output exactly one short search query inside "
    "{open_tag} and {close_tag} instead of think and action. Never output both a query and an action in the same response."
)


def to_webshop_instruction(text, open_tag=QUERY_OPEN, close_tag=QUERY_CLOSE):
    """Swap the (ALFWorld) belief instruction appended by the memory hint for the WebShop one."""
    a = BELIEF_INSTRUCTION.format(open_tag=open_tag, close_tag=close_tag)
    return text.replace(a, WEBSHOP_BELIEF_INSTRUCTION.format(open_tag=open_tag, close_tag=close_tag)) if a in text else text
