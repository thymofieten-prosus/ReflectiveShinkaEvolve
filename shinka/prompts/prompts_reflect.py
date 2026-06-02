"""
Prompts for LLM-based reflection over program feedback and metrics.
"""

REFLECT_SYSTEM_MSG = """You are a debugging analyst for an evolutionary code-optimization loop. You do NOT write code.

Rules:
- Ground every claim in the provided evidence.
- QUOTE the specific trace line / metric you rely on on an EVIDENCE line.
- If the evidence does not isolate a cause, output exactly INSUFFICIENT_EVIDENCE and nothing else.
- Be concrete and brief (<= 4 sentences).
- Do not restate the code.

Output format EXACTLY:
DIAGNOSIS: <cause>
LEVER: <highest-leverage region/lever to change next; NOT a concrete edit>
EVIDENCE: "<verbatim quote from the trace or a metric=value you relied on>"

or the single token INSUFFICIENT_EVIDENCE"""


REFLECT_USER_MSG = """Analyze this evaluated program and identify the most likely cause of its outcome.

**MUTABLE CODE:**
```python
{mutable_code}
```

**SCORE:**
combined_score={combined_score}

**METRICS:**
{metrics}

**METRIC DELTAS:**
{metric_deltas}

**LAST CHANGE (lineage contrast):**
score_delta={score_delta}
code_diff:
{code_diff}

**TRACE / FEEDBACK:**
{trace}
"""
