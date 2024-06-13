from openai import OpenAI
client = OpenAI()

# Define the code and the question
code = """
def add(a, b):
    return a + b
"""

question = "Answer only Yes or No. Does this code fulfil the requirement: Add two numbers together?"
# Prepare the prompt for the API
prompt = f"Here is a piece of Python code:\n\n{code}\n\n{question}"

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": question},
    {"role": "user", "content": prompt}
  ]
)

print(completion.choices[0].message)