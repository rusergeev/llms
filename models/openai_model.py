from typing import Optional, Generator

from openai import OpenAI

from models import Model



# OpenAI Model
class OpenAIModel(Model):
    def __init__(self, model: str, api_key: Optional[str] = None):
        super().__init__(model)
        self.client = OpenAI(api_key=api_key)

    def _create_response(self, system_message: str, user_prompt: str, temperature=None, stream: Optional[bool] = None):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        return self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            stream=stream
        )

    def tell(self, system_prompt: str, user_prompt: str) -> str:
        return self._create_response(system_prompt, user_prompt).choices[0].message.content

    def stream(self, system_prompt: str, user_prompt: str) -> Generator[str, None, None]:
        """Streams response from OpenAI API"""
        response = self._create_response(system_prompt, user_prompt, stream=True)
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
