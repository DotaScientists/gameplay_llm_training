from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[2]

class Settings(BaseSettings):
    local_db_path: Path = PROJECT_ROOT / "data/main.db"
    local_cache_path: Path = PROJECT_ROOT / "data/cache"
    local_train_dataset_path: Path = PROJECT_ROOT / "data/train_dataset"
    local_val_dataset_path: Path = PROJECT_ROOT / "data/val_dataset"
    local_test_dataset_path: Path = PROJECT_ROOT / "data/test_dataset"
    local_training_output_path: Path = PROJECT_ROOT / "data/training_output"
    local_logs_path: Path = PROJECT_ROOT / "data/logs"
    local_model_save_path: Path = PROJECT_ROOT / "data/trained_model"


    cloud_data_path: str = "gs://test_dota2_data/db/sqllite/main.db"

    project_name: str = "robust-doodad-416318"

    train_data_fraction: float = 0.8
    val_data_fraction: float = 0.1
    test_data_fraction: float = 0.1

    train_args_per_device_train_batch_size: int = 2
    train_args_per_device_eval_batch_size: int = 1
    train_args_gradient_accumulation_steps: int = 1
    train_args_num_train_epochs: int = 5
    train_args_learning_rate: float = 5e-4
    train_args_dataloader_num_workers: int = 8
    train_args_dataloader_pin_memory: bool = False
    train_args_dataloader_prefetch_factor: int | None = None
    train_args_max_seq_length: int = 1500


    pretrained_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / "envs/.env",
        env_file_encoding="utf-8",
    )