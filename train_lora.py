import datasets
import re
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16, float16
import os
import deepspeed
from torch.utils.data import DataLoader
import argparse
from mpi4py import MPI
import torch
from torch.utils.data import DataLoader, RandomSampler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


def main():
    model_id='Meta-Llama-3.1-70B-Instruct' #path to your huggingface model
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    timestamp = time.strftime("%Y%m%d_%H")
    filename=re.search(r'(?<=Llama-3\.1-)\d+B', model_id).group(0)
    out_dir = f'finetuned_models/{filename}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    dataset=datasets.load_dataset('json', data_files="training_data.jsonl", split='train')  #tokenized data with input_ids, attention_mask, labels
 
    training_args = TrainingArguments(
        gradient_checkpointing_kwargs={"use_reentrant": False},
        output_dir=out_dir,
        deepspeed='deepspeed_config.json',
        overwrite_output_dir=True,
        seed=42,
        do_eval=False,
        logging_strategy="steps",
        logging_steps=1000,
        learning_rate=2e-5,
        warmup_steps=50,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # tf32=True,
        bf16=True, 
        # fp16=True,
        weight_decay=0.1,
        push_to_hub=False,
        save_strategy="steps",
        num_train_epochs=20,
        save_steps=50, 
        save_on_each_node=False,
        save_total_limit=5,
        optim="paged_adamw_32bit",
        )
   

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj", "v_proj","o_proj","gate_proj","up_proj","down_proj", "lm_head"]
        )
    

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=bfloat16,
        # low_cpu_mem_usage=True,
        quantization_config=bnb_config,
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    trainer.train()
    trainer.save_model()
    print('Training DONE')

main()


