# Q001: D141 long-prompt and long-task quality (2026-10-05)

Run with `bench/qa_long.py` against the live D141 server (omarchy, 1x RTX 3090), at temperature 0 with no output caps. Each task was run once at C1, then as 4 concurrent copies at C4. That makes 50 task runs.

| task | what is checked | C1 | C4 (4 copies) |
|---|---|---|---|
| needle_8k / 64k / 200k | 3 facts planted at 10/50/90% depth in real code, plus a summary | 3/3, 3/3, 3/3 | 3/3 in every copy |
| code_lru | the generated LRU cache passes executed asserts | pass | 4/4 pass |
| code_rpn | the generated RPN evaluator passes 8 hidden tests | 8/8 | 8/8 in every copy |
| math_1..4 | integer answers (thinking on) | 4/4 * | 4/4 * |
| long_gen | essay of 1500+ words: length, repeated-4-gram rate, garbage | 3,131 words, rep4 0.007 | 2,509-3,648 words, rep4 <= 0.008 |

\* The logs show `math_4` as 0 because the answer key was wrong. Two-digit numbers whose digits sum to 9 add up to 486, not 495. The model answered 486 every time, and the key has since been fixed in `bench/qa_long.py`.

**Noise floor for later A/B runs.** Comparing D141 at C4 against D141 at C1, greedy streams fork after roughly 10-130 tokens. Short code answers agree completely. The mean |dlogprob| on the shared prefix is 0.0002-0.04.

A candidate should match these scores, and its drift should stay inside this band. An identical token stream is not the bar.
