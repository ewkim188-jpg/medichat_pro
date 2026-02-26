
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# 1. 설정 (Configuration)
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-med-chatbot"
# 실행 위치와 상관없이 절대 경로로 데이터셋 로드
dataset_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/instructions.jsonl")

# 2. QLoRA 설정 (4-bit Quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 3. 모델 로드 (Load Model)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. 토크나이저 로드 (Load Tokenizer)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 5. LoRA 설정 (PEFT)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# 6. 데이터셋 로드
print("Loading dataset...")
dataset = load_dataset("json", data_files=dataset_name, split="train")

# ... (omitted)

# 7. 학습 파라미터 설정 (SFTConfig)
# TRL 0.8.0+ 부터 SFTConfig 사용 권장
training_arguments = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    max_length=512, # max_seq_length -> max_length (TRL update)
    packing=False, # SFTConfig로 이동
)

# 8. 포맷팅 함수 정의 (Formatting Function)
def formatting_prompts_func(example):
    output_texts = []
    # 데이터가 배치가 아닌 단일 예시로 들어오는 경우를 처리
    if isinstance(example['instruction'], str):
        text = f"### Instruction: {example['instruction']}\n ### Input: {example['input']}\n ### Output: {example['output']}"
        return text # 단일 문자열 반환
    else:
        # 배치로 들어오는 경우
        for i in range(len(example['instruction'])):
            text = f"### Instruction: {example['instruction'][i]}\n ### Input: {example['input'][i]}\n ### Output: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

# 9. 트레이너 설정 (SFTTrainer)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    processing_class=tokenizer, # tokenizer -> processing_class (TRL update)
    args=training_arguments,
)

# 10. 학습 시작 (Train)
print("Starting training...")
trainer.train()

# 11. 모델 저장 (Save)
print("Saving model...")
trainer.model.save_pretrained(new_model)
print(f"Model saved to {new_model}")
