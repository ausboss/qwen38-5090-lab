# Reply for "Qwen 3.8 27B Reasoning Levels Tested" — Luke's Dev Lab

Context: he asked viewers who'd tested the reasoning levels to share results, and
concluded the levels are "soft guidance" with "no hard limits or restrictions".
That last part is the gap — a hard cap does exist, and it works.

---

## Option A — the main one (~180 words)

Your medium theory is right, and you can confirm it straight from the model's
chat template — the Jinja only has branches for `xhigh` and `low`. There's
literally no `elif` for medium, so it falls through with an empty instruction
string. You reverse-engineered that from token graphs, which is impressive.

One thing worth adding though: there *is* a hard limit available, it's just not
one of the three levels. llama.cpp has `--reasoning-budget N` (LM Studio exposes
it as the Reasoning Budget slider, default Unrestricted). It's a real cap, not a
suggestion.

Measured on my 5090 with the same family of GGUF, one multi-step word problem:

- default (xhigh): 271 words of thinking, 978 tokens total
- `--reasoning-budget 128`: 69 words of thinking, 356 tokens total — same
  correct answer

That's a 64% cut in total output with no wrong answer. Given the whole problem
in your last video was token burn, that seems like the knob people actually
want.

Two smaller gotchas: per-request it only works as
`chat_template_kwargs: {"reasoning_effort": "low"}` — a top-level
`reasoning_effort` field gets silently ignored. And on SGLang the levels don't
work *at all*; it inspects the template at boot, decides the model has no effort
control, and drops the value. Only `enable_thinking: false` does anything there.

---

## Option B — shorter (~90 words), if A feels long for a comment

Confirmed your medium theory from the chat template itself — the Jinja only has
branches for `xhigh` and `low`, there's no `elif` for medium at all, so it falls
through with an empty instruction. Nice catch inferring that from token counts.

Worth knowing there *is* a hard cap though: llama.cpp's `--reasoning-budget N`
(the Reasoning Budget slider in LM Studio). Not guidance — an actual limit. On my
5090, one word problem: default 271 words of thinking / 978 tokens total, vs
budget 128 → 69 words / 356 tokens, same correct answer. 64% less output.

---

## Option C — if you want to mention your own numbers differ

Same as A, plus this paragraph:

Interesting wrinkle: on a single-turn word problem I got the *opposite* ordering
to yours — medium produced the longest think block (435 words vs 271 for xhigh).
I think that's consistent with your explanation rather than against it: you're
measuring total context across a whole coding task including self-correction
loops, I'm measuring one think block with no retries. On a single turn "no
instruction" rambles; across a long task "no instruction" avoids the
verify-fail-retry spiral that low falls into. Same mechanism, different thing
being counted.

---

## Notes if he replies

- Build: llama.cpp e9fa078, unsloth `Qwen3.8-27B-UD-Q3_K_XL.gguf`, `-fa on`,
  `--cache-type-k/v q4_0`, `-c 131072`, single RTX 5090 (32GB).
- The budget measurement used the uncensored Q4_K_S build, same flags; the
  default-vs-budget comparison was back-to-back on the same server.
- `--reasoning-budget` is server-side only — passing `reasoning_budget` in the
  request body is ignored.
- Also worth mentioning if it comes up: `--spec-type draft-mtp` reads the MTP
  head out of the same GGUF and roughly doubles decode (85 → 153 tok/s here).
  Several uncensored/abliterated repacks ship with that head stripped
  (`nextn_predict_layers` absent, 64 blocks instead of 65), so they silently
  lose it.
