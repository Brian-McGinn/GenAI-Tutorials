# from datasets import load_dataset
# dataset = load_dataset("json", split='train', data_files='unsloth_data.json')
# from colorama import Fore

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
# from trl import SFTTrainer, SFTConfig
# from peft import LoraConfig, prepare_model_for_kbit_training
# import torch
# import os


# print(Fore.YELLOW + str(dataset[2]) + Fore.RESET) 

# def format_chat_template(batch, tokenizer):

#     system_prompt =  """You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope."""

#     samples = []

#     # Access the inputs from the batch
#     questions = batch["question"]
#     answers = batch["answer"]

#     for i in range(len(questions)):
#         row_json = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": questions[i]},
#             {"role": "assistant", "content": answers[i]}
#         ]

#         # Apply chat template and append the result to the list
#         tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
#         text = tokenizer.apply_chat_template(row_json, tokenize=False)
#         samples.append(text)

#     # Return a dictionary with lists as expected for batched processing
#     return {
#         "instruction": questions,
#         "response": answers,
#         "text": samples  # The processed chat template text for each row
#     }

# token = os.environ.get('HUGGING_FACE_READ_TOKEN')
# print(token)
# base_model = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
# tokenizer = AutoTokenizer.from_pretrained(
#         base_model, 
#         trust_remote_code=True,
#         token=token,
# )

# train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=10)
# print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 


# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

# model = AutoModelForCausalLM.from_pretrained(
#     base_model,
#     device_map="cuda:0",
#     quantization_config=quant_config,
#     token=token,
#     cache_dir="./workspace",
# )

# model.gradient_checkpointing_enable()
# model = prepare_model_for_kbit_training(model)

# peft_config = LoraConfig(
#     r=256,
#     lora_alpha=512,
#     lora_dropout=0.05,
#     target_modules="all-linear",
#     task_type="CAUSAL_LM",
# )

# trainer = SFTTrainer(
#     model,
#     train_dataset=train_dataset,
#     args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B-SFT", num_train_epochs=50),
#     peft_config=peft_config,
# )

# trainer.train()

# trainer.save_model('complete_checkpoint')
# trainer.model.save_pretrained("final_model")

# from huggingface_hub import login
import os

# login(os.environ.get('HUGGING_FACE_READ_TOKEN'))

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = 1024,
    dtype = None
)

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
"You are a helpful, honest and harmless assitant designed to help engineers. Think through each question logically and provide an answer. Don't make things up, if you're unable to answer a question advise the user that you're unable to answer as it is outside of your scope.

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["question"]
    outputs      = examples["answer"]
    texts = []
    for input, output in zip(inputs, outputs):
        text = prompt_style.format(input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = load_dataset("data", split='train')
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)
print(dataset["text"][0])

model = FastLanguageModel.get_peft_model(
    model,
    r=128   , 
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=256,
    lora_dropout=0.5, 
    bias="none", 
   
    use_gradient_checkpointing="unsloth", 
    random_state=3407,
    use_rslora=False, 
    loftq_config=None,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    num_train_epochs = 10,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 200,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# Track GPU memory before training
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 3) 

trainer_stats = trainer.train()

# get performance
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# from IPython.display import display, Markdown

# question = "Tell me about unsloth in 100 words or less"
# FastLanguageModel.for_inference(model)  # Unsloth has 2x faster inference!
# inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# outputs = model.generate(
#     input_ids=inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_new_tokens=1200,
#     use_cache=True,
# )
# response = tokenizer.batch_decode(outputs)
# print(response[0].split("### Response:")[1])

new_model_local = "Llama-3.1-8B-unsloth"
model.save_pretrained(new_model_local) # Local saving
tokenizer.save_pretrained(new_model_local) # Local saving