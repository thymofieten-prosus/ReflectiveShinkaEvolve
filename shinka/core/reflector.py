import json
import logging
import re
from typing import Any, Dict, Optional, Tuple

from shinka.database import Program
from shinka.edit import redact_immutable
from shinka.llm import LLMClient
from shinka.prompts import REFLECT_SYSTEM_MSG, REFLECT_USER_MSG

logger = logging.getLogger(__name__)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _stringify_trace(text_feedback: Any) -> str:
    if not text_feedback:
        return ""
    if isinstance(text_feedback, list):
        return "\n".join(str(item) for item in text_feedback if item is not None)
    return str(text_feedback)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).lower().split())


def _format_prompt_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def build_reflection_input(program: Program, parent: Optional[Program]) -> Dict[str, Any]:
    metrics = (
        program.public_metrics
        if isinstance(getattr(program, "public_metrics", None), dict)
        else {}
    )
    parent_metrics = (
        parent.public_metrics
        if parent is not None and isinstance(getattr(parent, "public_metrics", None), dict)
        else {}
    )

    metric_deltas = {}
    if parent is not None:
        for key in metrics.keys() & parent_metrics.keys():
            child_value = metrics[key]
            parent_value = parent_metrics[key]
            if _is_numeric(child_value) and _is_numeric(parent_value):
                metric_deltas[key] = child_value - parent_value

    score_delta = None
    if parent is not None and _is_numeric(program.combined_score):
        parent_score = getattr(parent, "combined_score", None)
        if _is_numeric(parent_score):
            score_delta = program.combined_score - parent_score

    code = getattr(program, "code", "") or ""
    try:
        mutable_only = redact_immutable(code, no_state=True)
        mutable_code = redact_immutable(code) if mutable_only.strip() else code
    except Exception as e:
        logger.warning(f"Could not redact immutable code for reflection input: {e}")
        mutable_code = code

    return {
        "mutable_code": mutable_code,
        "combined_score": getattr(program, "combined_score", 0.0),
        "metrics": metrics,
        "metric_deltas": metric_deltas,
        "score_delta": score_delta,
        "code_diff": getattr(program, "code_diff", None) or "",
        "trace": _stringify_trace(getattr(program, "text_feedback", "")),
    }


def has_sufficient_evidence(
    refl_input: Dict[str, Any], min_evidence_chars: int
) -> bool:
    trace = refl_input.get("trace") or ""
    if len(trace) >= min_evidence_chars:
        return True
    if refl_input.get("metrics") or refl_input.get("metric_deltas"):
        return True
    score_delta = refl_input.get("score_delta")
    code_diff = refl_input.get("code_diff") or ""
    return score_delta is not None and bool(code_diff)


def _extract_labeled_value(line: str, label: str) -> Optional[str]:
    cleaned = line.strip()
    cleaned = re.sub(r"^[>\s#*_`-]+", "", cleaned)
    pattern = rf"^\**\s*{label}\s*\**\s*:?\**\s*(?:[:\-]\s*)?(.*)$"
    match = re.match(pattern, cleaned, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().strip("*").strip()


def parse_reflection(content: str) -> Dict[str, str]:
    if content is None:
        content = ""
    stripped = content.strip()
    if stripped.upper().startswith("INSUFFICIENT_EVIDENCE"):
        return {
            "status": "insufficient",
            "diagnosis": "",
            "lever": "",
            "evidence": "",
        }

    parsed = {"diagnosis": "", "lever": "", "evidence": ""}
    for line in stripped.splitlines():
        for label, key in [
            ("DIAGNOSIS", "diagnosis"),
            ("LEVER", "lever"),
            ("EVIDENCE", "evidence"),
        ]:
            value = _extract_labeled_value(line, label)
            if value is not None:
                parsed[key] = value
                break

    status = "grounded" if parsed["diagnosis"] else "insufficient"
    return {"status": status, **parsed}


def _extract_quotes(evidence: str) -> list:
    quotes = []
    for pattern in [r'"([^"]+)"', r"'([^']+)'"]:
        quotes.extend(match.group(1) for match in re.finditer(pattern, evidence))
    return quotes


def _metric_value_matches(observed: Any, expected: str) -> bool:
    expected = expected.strip().strip('"').strip("'")
    if _is_numeric(observed):
        try:
            return abs(float(observed) - float(expected)) <= 1e-9
        except ValueError:
            return False
    return _normalize_text(observed) == _normalize_text(expected)


def _matches_metric_evidence(evidence: str, refl_input: Dict[str, Any]) -> bool:
    metrics = {}
    for field in ["metrics", "metric_deltas"]:
        values = refl_input.get(field)
        if isinstance(values, dict):
            metrics.update(values)

    if not metrics:
        return False

    pattern = r"([A-Za-z_][A-Za-z0-9_.-]*)\s*(?:=|:)\s*([-+]?[\w.]+)"
    for key, value_text in re.findall(pattern, evidence):
        if key in metrics and _metric_value_matches(metrics[key], value_text):
            return True
    return False


def check_grounding(evidence: str, refl_input: Dict[str, Any]) -> bool:
    if not evidence:
        return False

    normalized_trace = _normalize_text(refl_input.get("trace") or "")
    for quote in _extract_quotes(evidence):
        normalized_quote = _normalize_text(quote)
        if len(normalized_quote) >= 8 and normalized_quote in normalized_trace:
            return True

    return _matches_metric_evidence(evidence, refl_input)


def build_reflection_prompt(
    refl_input: Dict[str, Any], include_lineage_contrast: bool
) -> str:
    code_diff = refl_input.get("code_diff", "") if include_lineage_contrast else ""
    score_delta = (
        refl_input.get("score_delta") if include_lineage_contrast else ""
    )
    return REFLECT_USER_MSG.format(
        mutable_code=_format_prompt_value(refl_input.get("mutable_code", "")),
        combined_score=_format_prompt_value(refl_input.get("combined_score")),
        metrics=_format_prompt_value(refl_input.get("metrics", {})),
        metric_deltas=_format_prompt_value(refl_input.get("metric_deltas", {})),
        score_delta=_format_prompt_value(score_delta),
        code_diff=_format_prompt_value(code_diff),
        trace=_format_prompt_value(refl_input.get("trace", "")),
    )


class Reflector:
    """Produces grounded diagnostic reflections from program feedback and metrics."""

    def __init__(
        self,
        reflection_llm_client: Optional[LLMClient] = None,
        language: str = "python",
        grounding: bool = True,
        min_evidence_chars: int = 40,
        min_score_gap: float = 0.0,
        contrastive: bool = True,
    ):
        self.reflection_llm_client = reflection_llm_client
        self.language = language
        self.grounding = grounding
        self.min_evidence_chars = min_evidence_chars
        self.min_score_gap = min_score_gap
        self.contrastive = contrastive

    def _should_include_lineage_contrast(
        self, refl_input: Dict[str, Any], parent: Optional[Program]
    ) -> bool:
        score_delta = refl_input.get("score_delta")
        return (
            self.contrastive
            and parent is not None
            and score_delta is not None
            and abs(score_delta) > self.min_score_gap
        )

    def reflect_sync(
        self, program: Program, parent: Optional[Program]
    ) -> Tuple[str, str, float]:
        try:
            refl_input = build_reflection_input(program, parent)
            if not has_sufficient_evidence(refl_input, self.min_evidence_chars):
                return "skipped", "", 0.0

            if self.reflection_llm_client is None:
                logger.warning("Reflection LLM not configured")
                return "fallback", "", 0.0

            user_msg = build_reflection_prompt(
                refl_input,
                self._should_include_lineage_contrast(refl_input, parent),
            )
            query_kwargs = {
                "msg": user_msg,
                "system_msg": REFLECT_SYSTEM_MSG,
            }
            if hasattr(self.reflection_llm_client, "get_kwargs"):
                query_kwargs["llm_kwargs"] = self.reflection_llm_client.get_kwargs()

            response = self.reflection_llm_client.query(**query_kwargs)
            content = response.content if response is not None else ""
            cost = (response.cost or 0.0) if response is not None else 0.0
            parsed = parse_reflection(content or "")

            if parsed["status"] == "insufficient":
                return "insufficient", "", 0.0
            if self.grounding and not check_grounding(parsed["evidence"], refl_input):
                return "fallback", "", 0.0

            diagnosis_text = (
                f"DIAGNOSIS: {parsed['diagnosis']}\nLEVER: {parsed['lever']}"
            )
            return "grounded", diagnosis_text, cost
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return "fallback", "", 0.0
