from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAKUTEN__INFERENCE__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True
    )

    port: int = Field(default=..., description="Port on which the API runs.")

    mlflow_host: str = Field(default=..., description="Name of the mlflow service container at runtime.")
    mlflow_port: int = Field(default=..., description="Port on which the mlflow service runs.")

    mlflow_model_name: str = Field(default=..., description="Name of the model.")

    @computed_field
    @property
    def mlflow_server_uri(self) -> str:
        return f"http://{self.mlflow_host}:{self.mlflow_port}"

    @computed_field
    @property
    def mlflow_model_uri(self) -> str:
        return f"models:/{self.mlflow_model_name}"

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
