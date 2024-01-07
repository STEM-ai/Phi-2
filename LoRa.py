#import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
from huggingface_hub import notebook_login
import torch
import torch.nn as nn

notebook_login()

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2", 
    device_map='auto',
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float16)

model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float16)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=32,   #16, #attention heads
    lora_alpha= 64,#32, #alpha scaling
    target_modules=['Wqkv','fc1', 'fc2'], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

import transformers
from datasets import load_dataset
data = load_dataset("WillRanger/Electrical-engineering")

def merge_columns(example):
    new_example = {
        "instruction": example['instruction'],
        "input": example['input'],
        "prediction": example['instruction'] + " | " + example['input'] + " ->: " + example['output']
    }
    
    return new_example

data['train'] = data['train'].map(merge_columns)

def tokenize_and_truncate(samples):
    return tokenizer(samples['prediction'], max_length=10000, truncation=True)

# Apply tokenizer with truncation inside the map function
data = data.map(lambda samples: tokenize_and_truncate(samples), batched=True)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=8,
        warmup_steps=100, 
        max_steps=500, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.push_to_hub("WillRanger/Phi2-lora-Adapters2",
                  use_auth_token=True,
                  commit_message="basic training",
                  private=False)
