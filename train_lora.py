import os
import re
import time
import json
import yaml
import torch
import datasets
import argparse
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cuda.matmul.allow_tf32 = True


def load_config(path):
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    elif path.endswith(".yml") or path.endswith(".yaml"):
        with open(path) as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Only .json and .yml config files are supported")


def main(config):
    model_id = config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    timestamp = time.strftime("%Y%m%d_%H")
    modelname = re.search(r'(?<=Llama-3\.1-)\d+B', model_id).group(0)
    out_dir = f'{config["output_base_dir"]}/{modelname}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    dataset = datasets.load_dataset('json', data_files=config["dataset_path"], split='train')

    training_args = TrainingArguments(
        output_dir=out_dir,
        **config["training_args"]
    )

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # LoRA config
    peft_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config["lora"]["target_modules"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=bfloat16,
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
    print("Training DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json or config.yml")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)
