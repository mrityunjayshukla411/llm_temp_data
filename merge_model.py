from peft import AutoPeftModelForCausalLM
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from random import randrange
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


model_id = "TheBloke/Llama-2-7B-Chat-fp16"

new_model = AutoPeftModelForCausalLM.from_pretrained(
    'llama2-7b',
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Merge LoRA and base model
merged_model = new_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("metallama2-7b-tuned-merged", safe_serialization=True)
tokenizer.save_pretrained("metallama2-7b-tuned-merged")

import locale
locale.getpreferredencoding = lambda: "UTF-8"


hf_model_repo = "Cyber3ra/SecAI-Llama-2-40"
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)