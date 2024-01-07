import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import os

# Suppress INFO and WARNING messages from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning, module='transformers.generation.utils')

def load_model_and_tokenizer():
    base_model = "TheBloke/phi-2-GPTQ"
    peft_model_id = "WillRanger/Phi2-lora-Adapters2"
    config = PeftConfig.from_pretrained(peft_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda:0",return_dict=True, trust_remote_code=True)

    model = model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, peft_model_id, trust_remote_code=True)

    model = model.to('cuda')

    return model, tokenizer

def generate(instruction, model, tokenizer):                              
    inputs = tokenizer(instruction, return_tensors="pt", return_attention_mask=False)
    inputs = inputs.to('cuda')
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
    while True:  
        instruction = input("Enter your instruction: ")
        if not instruction:
            continue   
        if instruction.lower() in ["exit", "quit", "exit()", "quit()"]:
            print("Exiting...")
            break 

        answer = generate(instruction, model, tokenizer)
        print(f'Answer: {answer}')
