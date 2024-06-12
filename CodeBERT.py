import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from AbstractLLM import AbstractLLM

from abc import ABC, abstractmethod

class CodeBERT(AbstractLLM):
  """Concrete subclass implementing the singleton pattern."""

  def __init__(self, model_name: str = "Salesforce/codet5-base", tokenizer = AutoTokenizer, model = AutoModelForSeq2SeqLM):
      """Initializes the singleton object with a model name."""
      self.model_name = model_name
      self.tokenizer = tokenizer.from_pretrained(self.model_name)
      self.model = model.from_pretrained(self.model_name)

  def get_model_name(self):
    return self.model_name
  
  def get_tokenizer(self):
    return self.tokenizer
  
  def get_model(self):
    return self.model
  
  def tokenize(self, code: str):
    """Tokenizes the input code."""
    #return self.tokenizer.encode_plus(code, return_tensors="pt")
    return self.tokenizer(code, return_tensors='pt', max_length=512, truncation=True)
  
  def generate(self, inputs: dict):
    """Generates code using the model."""
    with torch.no_grad():
      return self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"])

  def generateSummary(self, inputs: dict):
    """Generates code using the model."""
    with torch.no_grad():
      #return self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=100, num_beams=4, length_penalty=2.0, early_stopping=True)
    
      summary_ids = self.model.generate(inputs['input_ids'], max_length=50, num_beams=4, early_stopping=True, no_repeat_ngram_size=2)
      summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
      return summary
  



