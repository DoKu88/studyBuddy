from openai import OpenAI

class ChatGPTInterface():

  def __init__(self):
    self.client = OpenAI()

  def get_model_name(self):
    return "ChatGPT" 

  def is_code_correct(self, code: str, requirement: str):
    requirement = f"Answer only Yes or No. Does this code fulfil the requirement: {requirement}?"
    prompt = f"Here is a piece of code:\n\n{code}"

    completion = self.client.chat.completions.create(
      #model="gpt-3.5-turbo",
      model= "gpt-4",
      messages=[
        {"role": "system", "content": requirement},
        {"role": "user", "content": prompt}
      ]
    )

    return completion.choices[0].message