import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm

# Load Tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load Model with 4-bit Quantization (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# Apply LoRA Config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap model with LoRA adapters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load Dataset
dataset = load_dataset("json", data_files="dialogue_sample.json", split="train")

# Tokenize Dataset
def tokenize_function(examples):
    all_contents = []
    for dialogue in examples["messages"]:
        contents = [msg["content"] for msg in dialogue if msg["role"] in ["user", "assistant"]]
        conversation = " ".join(contents)
        all_contents.append(conversation)

    tokenized_output = tokenizer(all_contents, truncation=True, padding="max_length", max_length=512)
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./llama2_finetuned55",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    num_train_epochs=2,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
    disable_tqdm=True,
    dataloader_pin_memory=True,
    run_name="test_gpu_training",
)

# Custom Progress Bar Callback
class ProgressCallback(TrainerCallback):
    def __init__(self, total_steps):
        self.progress_bar = tqdm(total=total_steps, desc="Training Progress")

    def on_step_end(self, args, state, control, **kwargs):
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        self.progress_bar.close()

# Trainer for Fine-Tuning
num_training_steps = len(tokenized_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)
trainer.add_callback(ProgressCallback(num_training_steps))

# Train Model
trainer.train()

# Save Model
trainer.save_model("./llama2_finetuned")
print("Fine-tuning completed and model saved!")