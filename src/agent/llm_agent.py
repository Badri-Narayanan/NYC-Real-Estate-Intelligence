from __future__ import annotations

from typing import Any

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import TOOL_SCHEMAS, execute_tool
from src.utils.config import load_config, get_api_key
from src.utils.logger import get_logger

log = get_logger(__name__)


class RealEstateAgent:
    """Multi-turn agent. Uses Anthropic's tool_use loop."""

    def __init__(self, config: dict | None = None,
                  model: str | None = None,
                  max_iterations: int = 6):
        self.cfg = config or load_config()
        self.model = model or self.cfg["agent"]["model"]
        self.fallback_model = self.cfg["agent"]["fallback_model"]
        self.max_iterations = max_iterations
        self.api_key = get_api_key("anthropic")
        self.history: list[dict[str, Any]] = []

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key) if self.api_key else None
        except ImportError:
            log.error("anthropic SDK not installed. `pip install anthropic`")
            self.client = None

    def is_available(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def reset(self):
        self.history = []

    def _call_claude(self, messages: list[dict]):
        for model_id in [self.model, self.fallback_model]:
            try:
                return self.client.messages.create(
                    model=model_id,
                    max_tokens=self.cfg["agent"]["max_tokens"],
                    system=SYSTEM_PROMPT,
                    tools=TOOL_SCHEMAS,
                    messages=messages,
                )
            except Exception as e:
                log.warning(f"Model {model_id} failed: {e}; trying fallback...")
                continue
        raise RuntimeError("All Claude model calls failed")

    def chat(self, user_message: str) -> str:
        """Run a single user turn through the tool-use loop."""
        if not self.is_available():
            return ("⚠️ Agent not available. Set ANTHROPIC_API_KEY in config/.env "
                    "(see config/.env.example) to enable the conversational agent.")

        self.history.append({"role": "user", "content": user_message})

        for it in range(self.max_iterations):
            response = self._call_claude(self.history)
            log.info(f"Iteration {it+1}: stop_reason={response.stop_reason}")

            # Append assistant message
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        log.info(f"  -> tool: {block.name} args={block.input}")
                        result_text = execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                self.history.append({"role": "user", "content": tool_results})
                continue

            # Done - return concatenated text
            text_parts = [b.text for b in response.content if b.type == "text"]
            return "\n".join(text_parts).strip()

        return "(Agent stopped after max iterations without final answer.)"

    def chat_streaming_callback(self, user_message: str, on_event=None) -> str:
        """
        Same as chat() but invokes on_event(kind, payload) for each step.
        kind ∈ {"thinking", "tool_call", "tool_result", "final"}
        Useful for the Streamlit UI to show what's happening.
        """
        if not self.is_available():
            msg = "Agent unavailable - missing ANTHROPIC_API_KEY"
            if on_event:
                on_event("final", msg)
            return msg

        self.history.append({"role": "user", "content": user_message})

        for it in range(self.max_iterations):
            response = self._call_claude(self.history)
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if on_event:
                            on_event("tool_call", {"name": block.name,
                                                     "args": block.input})
                        result_text = execute_tool(block.name, block.input)
                        if on_event:
                            on_event("tool_result", {"name": block.name,
                                                       "result": result_text})
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })
                self.history.append({"role": "user", "content": tool_results})
                continue

            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            if on_event:
                on_event("final", text)
            return text

        return "(max iterations)"
