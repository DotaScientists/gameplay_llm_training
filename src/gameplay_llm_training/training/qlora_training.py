import torch
from trl import SFTTrainer
from datasets import Dataset, load_from_disk
from transformers import TrainingArguments, LlamaForCausalLM, BitsAndBytesConfig, LlamaTokenizerFast, SchedulerType
from gameplay_llm_training.settings import Settings
from peft import LoraConfig
from peft.utils.peft_types import PeftType, TaskType
from loguru import logger
from peft import get_peft_model
from gameplay_llm_training.data_preparation.data_preparation import formatting_prompts_func
from gameplay_llm_training.cloud.gcs_client import GCSConnector


def upload_artifacts(gsc_connector: GCSConnector, settings: Settings):
    training_runs = gsc_connector.list_folders(settings.cloud_training_artifacts_path)
    if not training_runs:
        run_number = 0
    else:
        run_number = max([int(folder_name) for folder_name in training_runs])
        run_number += 1
    run_cloud_folder = f"{settings.cloud_training_artifacts_path}/{run_number}"
    gsc_connector.upload_folder(settings.local_logs_path, f"{run_cloud_folder}/logs")
    gsc_connector.upload_folder(settings.local_model_save_path, f"{run_cloud_folder}/model")



def train_llm(settings: Settings):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    logger.info("Loading model")
    model = LlamaForCausalLM.from_pretrained(
        settings.pretrained_model_name,
        quantization_config=quantization_config,
        cache_dir=settings.local_cache_path
    )
    logger.info("Loading tokenizer")
    tokenizer = LlamaTokenizerFast.from_pretrained(
        settings.pretrained_model_name,
        cache_dir=settings.local_cache_path
    )
    logger.info("Loading datasets")

    train_dataset = load_from_disk(str(settings.local_train_dataset_path))
    eval_dataset = load_from_disk(str(settings.local_val_dataset_path))

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
        dataloader_num_workers=settings.train_args_dataloader_num_workers,
        dataloader_pin_memory=settings.train_args_dataloader_pin_memory,
        dataloader_prefetch_factor=settings.train_args_dataloader_prefetch_factor,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_strategy="steps",
        logging_steps=2000,
        gradient_checkpointing=False,
        save_total_limit=3,
        report_to="tensorboard"
    )


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        peft_type=PeftType.LORA,
        task_type=TaskType.CAUSAL_LM,
        use_dora=False,
    )

    peft_model = get_peft_model(model, peft_config)

    trainer = SFTTrainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        max_seq_length=settings.train_args_max_seq_length,
    )
    logger.info("Training model")
    trainer.train()
    trainer.save_model(str(settings.local_model_save_path))






# Base
# 428/24100 [02:38<2:25:57,  2.70it/s
# Direct peft | USED
# 500/24100 [02:24<1:53:18,  3.47it/s]
# NF4 | NOT USED
# 440/24100 [02:21<2:06:54,  3.11it/s]
# bnb_4bit_compute_dtype | USED
# 633/24100 [02:01<1:15:07,  5.21it/s]
# pin memory False | USED
# 792/24100 [02:30<1:13:55,  5.25it/s]
# Workers: 8 | USED
# 1441/24100 [04:32<1:11:31,  5.28it/s]
# No dora | USED
# 544/24100 [01:28<1:03:47,  6.15it/s]