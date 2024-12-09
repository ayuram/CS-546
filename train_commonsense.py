import random
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    Trainer,
    BitsAndBytesConfig
)
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import wandb
from huggingface_hub import notebook_login
import os
from accelerate import Accelerator
import transformers
import json

model_name = "meta-llama/Llama-3.2-1B"
dataset_split = "train"
per_device_train_batch_size = 2
num_train_epochs = 20
learning_rate = 5e-5

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

use_prefilter = True
if use_prefilter:
    train_k = load_from_disk("commonsense_id")
    print(train_k)
else:
    train = load_dataset("tau/commonsense_qa", trust_remote_code=True)["train"]
    
    def prepare_self(batch):
        answers = []
        for i in range(len(batch['choices'])):
            answer = batch['choices'][i]['text'][batch['choices'][i]['label'].index(batch['answerKey'][i])]
            answers.append(f"Question: {batch['question'][i]}. Choices: {batch['choices'][i]['text']} ONLY give an answer in these choices. Answer: ")
        return {"check_text": answers}
    
    train_c = train.map(prepare_self, batched=True)
    
    model = model.cuda()
    
    def identify_choices(item):
        inputs = tokenizer.encode(item['check_text'], return_tensors='pt').cuda()
        attn_mask = torch.ones(inputs.shape).cuda()
        outputs = model.generate(inputs, max_new_tokens=10, attention_mask=attn_mask, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        ans = (tokenizer.batch_decode(outputs)[0].split('Answer:')[1])
        choices = item['choices']['text']
        selected = []
        for i in choices:
            if i.lower() in ans.lower():
                selected.append(i)
        return {"selected": selected}
    
    train_k = train_c.map(identify_choices)
    train_k.save_to_disk("commonsense_id")
def process_rtune(batch):
    answers = []
    for i in range(len(batch['choices'])):
        answer = batch['choices'][i]['text'][batch['choices'][i]['label'].index(batch['answerKey'][i])]
        distractors = batch['selected'][i]
        if answer in distractors:
            distractors.remove(answer)
        if len(distractors) > 0:
            wrong = random.choice(distractors)
            answers.append(f"Question: {batch['question'][i]}. Answer: {wrong}. Are you sure you answered the question correctly based on your internal knowledge? I am unsure.")
        answers.append(f"Question: {batch['question'][i]}. Answer: {answer}. Are you sure you answered the question correctly based on your internal knowledge? I am sure.")
    return {"text": answers}

trr = train_k.map(process_rtune, batched=True, remove_columns=train_k.features)
print(trr)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    logging_steps=50,
    output_dir='./results-commonsense',
    evaluation_strategy='no',
    logging_dir='./logs',
    report_to='wandb',
    run_name='llama-3.2-1b-lora-commonsense-finetune',
    fp16=True,
)

def prompt_instruction_format(sample):
    return sample["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=trr,
    peft_config=lora_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=prompt_instruction_format,
    args=training_args,
    max_seq_length=min(tokenizer.model_max_length, 1600)
)

trainer.train()
output_dir = "./tau_commonsense"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
