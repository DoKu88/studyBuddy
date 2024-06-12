from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod

class AbstractLLM(ABC):
  """Abstract base class for a singleton pattern."""

  _instance = None
  def __new__(cls, model_name: str = "", tokenizer = None, model = None):
    """Controls object creation, ensuring only one instance exists."""
    if not cls._instance:
      cls._instance = super(AbstractLLM, cls).__new__(cls)
    return cls._instance

  @abstractmethod
  def get_model_name(self):
    """Abstract method to get the model name."""
    pass

  @abstractmethod
  def get_tokenizer(self):
    """Abstract method to get the tokenizer."""
    pass

  @abstractmethod
  def get_model(self):
    """Abstract method to get the model."""
    pass
   