from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel, PeftConfig

# === Step 1: Define model and adapter ===
base_model = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
adapter_path = "/home/robinwill/brian-intel/fine-tuning/Llama-3.1-8B-unsloth"  # can be local dir or HF hub

# === Step 2: Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

# === Step 3: Load base model (with quantization, optional) ===
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True,  # optional for QLoRA
)

base.eval()

# === Step 6: Run inference ===
prompt = "What role does Daniel Han hold in the company?"
inputs = tokenizer(prompt, return_tensors="pt").to(base.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

outputs = base.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
)


print("!!!!!!!!!!!!!")
# === Step 4: Load adapter on top of base model ===
model_adapter = PeftModel.from_pretrained(base, adapter_path)
model_adapter.eval()

# === Step 5: Run inference ===
inputs = tokenizer(prompt, return_tensors="pt").to(model_adapter.device)

streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

outputs = model_adapter.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.5,
    top_p=0.9,
)

# Decode if you didn’t use a streamer
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
