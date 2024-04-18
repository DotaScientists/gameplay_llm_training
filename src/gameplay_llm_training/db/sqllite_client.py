import sqlite3
from gameplay_llm_training.settings import Settings


class SQLLiteDB:

    def __init__(self, settings: Settings):
        self.settings = settings
        self.conn = sqlite3.connect(str(settings.local_db_path.absolute()))

    def setup_tables(self):
        sql = """
        CREATE TABLE IF NOT EXISTS dataset (
            match_id INTEGER,
            slot INTEGER,
            text_data TEXT,
            data_prompt TEXT,
            instruction_prompt TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            model TEXT,
            generation_time INTEGER,            
            PRIMARY KEY (match_id, slot)
        )
        """
        self.conn.execute(sql)
        self.conn.commit()

    def close(self):
        self.conn.close()

    def get_dataset(self):
        sql = """
        SELECT
            match_id INTEGER,
            slot INTEGER,
            instruction_prompt TEXT,
            data_prompt TEXT,
            text_data TEXT
        FROM dataset
        """
        return self.conn.execute(sql).fetchall()