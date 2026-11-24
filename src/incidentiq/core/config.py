
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env values into the process environment.
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "incidentiq"

    incidentiq_data_dir: Path = Path("data")
    incidentiq_vector_dir: Path = Path(".local/chroma")
    incidentiq_sqlite_path: Path = Path(".local/incidentiq.db")


settings = Settings()
