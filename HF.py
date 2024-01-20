import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation.utils')

def load_model_and_tokenizer():
    base_model = "microsoft/phi-2"
    peft_model_id = "STEM-AI-mtl/phi-2-electrical-engineering"
    config = PeftConfig.from_pretrained(peft_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cpu", return_dict=True, trust_remote_code=True)

    # Change model to CPU
    model = model.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, peft_model_id, trust_remote_code=True)

    # Change model to CPU
    model = model.to('cpu')

    return model, tokenizer

def generate(instruction, model, tokenizer):                              
    inputs = tokenizer(instruction, return_tensors="pt", return_attention_mask=False)

    # Change inputs to CPU
    inputs = inputs.to('cpu')

    outputs = model.generate(
        **inputs, 
        max_length=350,
        do_sample=True, 
        temperature=0.7,
        top_k=50,  
        top_p=0.9,
        repetition_penalty=1,
    )
    text = tokenizer.batch_decode(outputs)[0]
    return text

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    if len(sys.argv) > 1:
        instruction = sys.argv[1]
        answer = generate(instruction, model, tokenizer)
        print(f'Answer: {answer}')
    else:
        print("No instruction provided.")
    #while True:  
     #   instruction = input("Enter your instruction: ")
      #  if not instruction:
       #     continue   
        #if instruction.lower() in ["exit", "quit", "exit()", "quit()"]:
         #   print("Exiting...")
          #  break 

        #answer = generate(instruction, model, tokenizer)
        #print(f'Answer: {answer}')
