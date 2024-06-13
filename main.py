import GitAccessor
from CodeBERT import CodeBERT
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import PLBartTokenizer, PLBartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tokenize
import io
#from transformers import HfApi
from huggingface_hub import HfApi, list_models
import torch
import sys

def query_code(code, question, tokenizer, model):
    # Prepare the input text in the format expected by T5
    input_text = f"code: {code} question: {question}"
    
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate the output
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    
    # Decode the output to get the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

'''
def main():
    # Initialize the API
    api = HfApi()

    # Search for models with "T5" in their name
    t5_models = api.list_models(filter="T5")

    modelSet = set()

    # Print the names of the models
    for model in t5_models:
        modelSet.add(model.modelId)

    print("Is t5-small in modelSet: ", "t5-small" in modelSet)


    # Load the tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False, force_download=False)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")

    # Example code and question
    code_example = """
    def add(a, b):
        return a + b
    """
    question = "What does the add function do?"

    # Get the answer
    answer = query_code(code_example, question, tokenizer, model)
    print("Answer:", answer)
'''

def main():

    api = HfApi()
    # Search for models with "T5" in their name
    t5_models = api.list_models()

    modelSet = set()

    # Print the names of the models
    for model in t5_models:
        modelSet.add(model.modelId)

    # print modelSet out to a file 
    with open("modelSet.txt", "w") as f:
        for model in modelSet:
            f.write(model + "\n")

    sys.exit(0)
    
    
    # Load the CodeT5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained('microsoft/codebert-base')
    model = T5ForConditionalGeneration.from_pretrained('microsoft/codebert-base')

    # Sample Python code to be summarized
    code = """
    def add(a, b):
        return a + b
    """

    # Prepare the input for the model
    input_text = "summarize: " + code
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate the summary
    summary_ids = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("Code Summary:", summary)


if __name__ == "__main__":
    main()