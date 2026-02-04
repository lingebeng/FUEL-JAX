"""LLM API client for test case generation."""

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from .config import LLMConfig, load_llm_config_from_env

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for LLM API (Deepseek) to generate test cases."""

    def __init__(self, config: LLMConfig | None = None):
        """
        Initialize the LLM client.

        Args:
            config: LLM configuration. If None, loads from environment.
        """
        self.config = config or load_llm_config_from_env()
        if not self.config.api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY environment variable not set. "
                "Please set it or provide api_key in config."
            )
        self._client = httpx.Client(timeout=120.0)

    def _load_prompt_template(self) -> str:
        """Load the test generation prompt template."""
        prompt_path = Path(__file__).parent.parent / "prompts" / "gen" / "gen_test.md"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        return prompt_path.read_text(encoding="utf-8")

    def generate_test_cases(
        self,
        jax_op: str,
        torch_op: str,
        op_type: str,
        num_cases: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Generate test cases for an operator pair using LLM.

        Args:
            jax_op: JAX operator name (e.g., "jax.lax.abs")
            torch_op: PyTorch operator name (e.g., "torch.abs")
            op_type: Operator type (e.g., "elementwise")
            num_cases: Number of test cases to generate

        Returns:
            List of test case dictionaries
        """
        # Load and format prompt template
        template = self._load_prompt_template()
        prompt = template.format(
            jax_op=jax_op,
            torch_op=torch_op,
            op_type=op_type,
            num_cases=num_cases,
        )

        # Call LLM API
        response = self._call_api(prompt)

        # Parse JSON from response
        test_cases = self._parse_response(response)

        return test_cases

    def _call_api(self, prompt: str) -> str:
        """Call the LLM API and return the response content."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert in numerical computing and testing. "
                    "Generate test cases as valid JSON arrays. "
                    "Only output the JSON array, no additional text.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        try:
            response = self._client.post(
                f"{self.config.base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            logger.error(f"API request failed: {e}")
            raise

    def _parse_response(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response to extract test cases JSON."""
        # Try to find JSON array in the response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Find start and end of code block
            start_idx = 0
            end_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("```") and i == 0:
                    start_idx = 1
                elif line.startswith("```") and i > 0:
                    end_idx = i
                    break
            response = "\n".join(lines[start_idx:end_idx])

        # Find JSON array
        start = response.find("[")
        end = response.rfind("]") + 1

        if start == -1 or end == 0:
            logger.error(f"No JSON array found in response: {response[:200]}...")
            return []

        json_str = response[start:end]

        try:
            test_cases = json.loads(json_str)
            if not isinstance(test_cases, list):
                logger.error("Response is not a JSON array")
                return []
            return test_cases
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"JSON string: {json_str[:500]}...")
            return []

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
