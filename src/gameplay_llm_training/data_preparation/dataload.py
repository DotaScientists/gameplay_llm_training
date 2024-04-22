from gameplay_llm_training.cloud.gcs_client import GCSConnector
from gameplay_llm_training.db.sqllite_client import SQLLiteDB
from gameplay_llm_training.settings import Settings

from datasets import Dataset
from loguru import logger

def download_db(settings: Settings, gcs_connector: GCSConnector) -> SQLLiteDB:
    gcs_connector.download(settings.cloud_data_path, settings.local_db_path)
    db = SQLLiteDB(settings)
    return db


def extract_raw_dataset(db: SQLLiteDB) -> dict[int, dict[int, dict[str, str]]]:
    data = db.get_dataset()
    dataset = dict()
    for match_id, slot, instruction_prompt, data_prompt, text_response in data:
        if match_id not in dataset:
            dataset[match_id] = dict()
        dataset[match_id][slot] = {
            "instruction_prompt": instruction_prompt,
            "data_prompt": data_prompt,
            "text_data": text_response
        }
    return dataset


def split_dataset(dataset: dict[int, dict[int, dict[str, str]]], settings: Settings)-> tuple[dict, dict, dict]:
    train_fraction = settings.train_data_fraction / (settings.train_data_fraction + settings.val_data_fraction + settings.test_data_fraction)
    val_fraction = settings.val_data_fraction / (settings.train_data_fraction + settings.val_data_fraction + settings.test_data_fraction)
    train_count = int(len(dataset) * train_fraction)
    val_count = int(len(dataset) * val_fraction)
    match_ids = list(dataset.keys())
    train_match_ids = match_ids[:train_count]
    val_match_ids = match_ids[train_count:train_count + val_count]
    test_match_ids = match_ids[train_count + val_count:]
    train_dataset = {match_id: dataset[match_id] for match_id in train_match_ids}
    val_dataset = {match_id: dataset[match_id] for match_id in val_match_ids}
    test_dataset = {match_id: dataset[match_id] for match_id in test_match_ids}
    return train_dataset, val_dataset, test_dataset


def encode_text(
        dataset: dict[int, dict[int, dict[str, str]]],
    )->Dataset:
    instruction_prompts = [
        slot_data["instruction_prompt"]
        for match_data in dataset.values() for slot_data in match_data.values()
    ]
    data_prompts = [
        slot_data["data_prompt"]
        for match_data in dataset.values() for slot_data in match_data.values()
    ]
    labels = [
        slot_data["text_data"]
        for match_data in dataset.values() for slot_data in match_data.values()
    ]
    dataset = Dataset.from_dict({"instruction": instruction_prompts, "data": data_prompts, "label": labels})
    return dataset


def load_data(settings: Settings, gcs_connector: GCSConnector):
    logger.info("Downloading db")
    db = download_db(settings, gcs_connector)
    logger.info("Extracting raw dataset")
    raw_data = extract_raw_dataset(db)
    logger.info(f"Total raw data size: {len(raw_data)}")
    logger.info("Splitting dataset")
    train_data, val_data, test_data = split_dataset(raw_data, settings)
    logger.info(f"Train data matches: {len(train_data)}")
    logger.info(f"Val data matches: {len(val_data)}")
    logger.info(f"Test data matches: {len(test_data)}")
    logger.info("Encoding train dataset")
    train_dataset = encode_text(train_data)
    logger.info("Encoding val dataset")
    val_dataset = encode_text(val_data)
    logger.info("Encoding test dataset")
    test_dataset = encode_text(test_data)
    logger.info("Saving datasets to disk")
    train_dataset.save_to_disk(str(settings.local_train_dataset_path))
    val_dataset.save_to_disk(str(settings.local_val_dataset_path))
    test_dataset.save_to_disk(str(settings.local_test_dataset_path))
