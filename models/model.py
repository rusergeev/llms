from abc import ABC, abstractmethod
from typing import Generator


class Model(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def tell(self, system_message: str, user_prompt: str) -> str:
        """Abstract method that must be implemented by subclasses"""
        pass

    @abstractmethod
    def stream(self, system_message: str, user_prompt: str) -> Generator[str, None, None]:
        """Abstract method that must be implemented by subclasses"""
        pass