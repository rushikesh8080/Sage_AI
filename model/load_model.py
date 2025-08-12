import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml

def load_config():
    with open("./config/config.yaml", "r") as f:
        return yaml.safe_load(f)

def load_model_and_tokenizer():
    config = load_config()

    tokenizer = AutoTokenizer.from_pretrained(config["base_model_dir"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model_dir"],
        torch_dtype=torch.float32,  # CPU friendly
        device_map={"": "cpu"},
    )

    return model, tokenizer
