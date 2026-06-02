import logging
from typing import Optional, Tuple

from .reflector import (
    Reflector,
    build_reflection_input,
    build_reflection_prompt,
    check_grounding,
    has_sufficient_evidence,
    parse_reflection,
)
from ..database import Program
from ..llm import AsyncLLMClient
from ..prompts import REFLECT_SYSTEM_MSG

logger = logging.getLogger(__name__)


class AsyncReflector:
    """Async wrapper for Reflector using an AsyncLLMClient."""

    def __init__(
        self,
        sync_reflector: Reflector,
        async_llm_client: Optional[AsyncLLMClient] = None,
    ):
        self.sync_reflector = sync_reflector
        self.async_llm_client = async_llm_client

    async def reflect(
        self, program: Program, parent: Optional[Program]
    ) -> Tuple[str, str, float]:
        try:
            refl_input = build_reflection_input(program, parent)
            if not has_sufficient_evidence(
                refl_input, self.sync_reflector.min_evidence_chars
            ):
                return "skipped", "", 0.0

            if self.async_llm_client is None:
                logger.warning("Async reflection LLM not configured")
                return "fallback", "", 0.0

            user_msg = build_reflection_prompt(
                refl_input,
                self.sync_reflector._should_include_lineage_contrast(
                    refl_input, parent
                ),
            )
            response = await self.async_llm_client.query(
                msg=user_msg,
                system_msg=REFLECT_SYSTEM_MSG,
            )
            content = response.content if response is not None else ""
            cost = (response.cost or 0.0) if response is not None else 0.0
            parsed = parse_reflection(content or "")

            if parsed["status"] == "insufficient":
                return "insufficient", "", 0.0
            if self.sync_reflector.grounding and not check_grounding(
                parsed["evidence"], refl_input
            ):
                return "fallback", "", 0.0

            diagnosis_text = (
                f"DIAGNOSIS: {parsed['diagnosis']}\nLEVER: {parsed['lever']}"
            )
            return "grounded", diagnosis_text, cost
        except Exception as e:
            logger.error(f"Error in async reflection: {e}")
            return "fallback", "", 0.0
