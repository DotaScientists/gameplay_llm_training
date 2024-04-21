from trl import SFTTrainer
from datasets import Dataset
from transformers import TrainingArguments, LlamaForCausalLM, BitsAndBytesConfig, LlamaTokenizer, SchedulerType
from gameplay_llm_training.settings import Settings
from peft import LoraConfig
from peft.utils.peft_types import PeftType, TaskType
from loguru import logger
from gameplay_llm_training.training.callbacks import LRCallback


def formatting_func(example):
    text = f"###{example['instruction']}\n ### Data: {example['data']} ### Response: {example['label']}"
    return text


def train_llm(settings: Settings):
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    logger.info("Loading model")
    model = LlamaForCausalLM.from_pretrained(
        settings.pretrained_model_name,
        quantization_config=quantization_config,
        cache_dir=settings.local_cache_path
    )
    logger.info("Loading tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(
        settings.pretrained_model_name,
        cache_dir=settings.local_cache_path
    )
    logger.info("Loading datasets")
    train_dataset = Dataset.load_from_disk(str(settings.local_train_dataset_path))
    eval_dataset = Dataset.load_from_disk(str(settings.local_val_dataset_path))

    training_args = TrainingArguments(
        output_dir=str(settings.local_training_output_path),
        per_device_train_batch_size=settings.train_args_per_device_train_batch_size,
        per_device_eval_batch_size=settings.train_args_per_device_eval_batch_size,
        num_train_epochs=settings.train_args_num_train_epochs,
        logging_dir=str(settings.local_logs_path),
        gradient_accumulation_steps=settings.train_args_gradient_accumulation_steps,
        overwrite_output_dir=True,
        learning_rate=settings.train_args_learning_rate,
        lr_scheduler_type=SchedulerType.LINEAR,
        warmup_ratio=0.1,
        dataloader_num_workers=2,
        evaluation_strategy="epoch",
        logging_strategy="epoch"
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        peft_type=PeftType.LORA,
        task_type=TaskType.CAUSAL_LM,

    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        tokenizer=tokenizer,
        dataset_batch_size=settings.training_args_dataset_batch_size,
    )
    logger.info("Training model")
    trainer.train()
    trainer.save_model()

