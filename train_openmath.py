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
from syncode import SyncodeLogitsProcessor
from syncode import Grammar
import json

model_name = "meta-llama/Llama-3.2-1B"
dataset_split = "train"
per_device_train_batch_size = 2
num_train_epochs = 5
learning_rate = 5e-5

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

use_prefilter = False
if use_prefilter:
    train_k = load_from_disk("openmath_id")
    print(train_k)
else:
    train = load_dataset("nvidia/OpenMathInstruct-2", trust_remote_code=True)["train"].select(range(20000))
    
    def prepare_self(batch):
        answers = []
        choices_batch = []
        for i in range(len(batch["problem"])):
            choices = choices = [i] + [x for x in range(len(batch["problem"]))]
            random.shuffle(choices)
            choice_str = f"A: {batch['expected_answer'][choices[0]]}, B: {batch['expected_answer'][choices[1]]}, C: {batch['expected_answer'][choices[2]]}, D: {batch['expected_answer'][choices[3]]}"
            choices_batch.append([batch["generated_solution"][i] for i in choices])
            answers.append(f"Question: {batch['problem'][i]}. Choices: {choice_str}. Give ONLY the letter indicating your response. Answer: ")
        return {"check_text": answers, "choice_lst": choices_batch}
    
    train_c = train.map(prepare_self, batched=True)
    
    def identify_choices(item):
        grammar = Grammar(f'start: "A" | "B" | "C" | "D"'.replace('\n', ''))
        logits = SyncodeLogitsProcessor(grammar=grammar, tokenizer=tokenizer, parse_output_only=True)
        logits.reset(item["check_text"])
        inputs = tokenizer.encode(item['check_text'], return_tensors='pt').cuda()
        choices = item["choice_lst"]
        choice_map = {'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]}
        attn_mask = torch.ones(inputs.shape).cuda()
        outputs = model.generate(inputs, max_new_tokens=50, attention_mask=attn_mask, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, logits_processor=[logits])
        ans = (tokenizer.batch_decode(outputs)[0].split('Answer:')[1])
        selected = []
        for i in choice_map.keys():
            if i in ans:
                selected.append(choice_map[i])
        return {"selected": selected}
    train_k = train_c.map(identify_choices)
    print(train_k)
    train_k.save_to_disk("openmath_id")

def process_rtune(batch):
    answers = []
    for i in range(len(batch["problem"])):
        answer = batch["expected_answer"][i]
        distractors = batch['selected'][i]
        if answer in distractors:
            distractors.remove(answer)
        if len(distractors) > 0:
            wrong = random.choice(distractors)
            answers.append(f"Question: {batch['problem'][i]}. Answer: {wrong}. Are you sure you answered the question correctly based on your internal knowledge? I am unsure.")
        answers.append(f"Question: {batch['problem'][i]}. Answer: {answer}. Are you sure you answered the question correctly based on your internal knowledge? I am sure.")
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
    output_dir='./results-openmath',
    evaluation_strategy='no',
    logging_dir='./logs',
    report_to='wandb',
    run_name='llama-3.2-1b-lora-openmath-finetune',
    fp16=True,
)

def problem_instruction_format(sample):
    return sample["text"]

trainer = SFTTrainer(
    model=model,
    train_dataset=trr,
    peft_config=lora_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=problem_instruction_format,
    args=training_args,
    max_seq_length=min(tokenizer.model_max_length, 1600)
)

trainer.train()
output_dir = "./tau_openmath"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
