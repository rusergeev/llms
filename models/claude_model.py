# Claude (Anthropic) Model
from typing import Optional, Generator
from models import Model
import anthropic


class ClaudeModel(Model):
    def __init__(self, model: str, api_key: Optional[str]  = None):
        super().__init__(model)
        self.client = anthropic.Anthropic()

    def tell(self, system_prompt: str, user_prompt: str) -> str:
        result = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        return result.content[0].text

    def stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """Streams response from Claude API"""
        result = self.client.messages.stream(
            model=self.model,
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        response = ""
        with result as stream:
            for text in stream.text_stream:
                response += text or ""
                yield response
