from gameplay_llm_training.data_preparation.dataload import load_data
from gameplay_llm_training.settings import Settings
from gameplay_llm_training.cloud.gcs_client import GCSConnector
from gameplay_llm_training.training.qlora_training import train_llm


def main():
    settings = Settings()
    gcs_connector = GCSConnector(settings.project_name)
    # load_data(settings, gcs_connector)
    train_llm(settings)

if __name__ == "__main__":
    main()