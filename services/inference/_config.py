from functools import lru_cache
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# The champion the training pipeline publishes: see champion_challenger in
# training/params.yaml, which is where the name is decided.
DEFAULT_MODEL: str = "rakuten-naive@champion"


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

    # Not required: a fresh clone must start without anyone filling in a .env.
    # The default is the champion the training pipeline produces, which is the
    # name declared in training/params.yaml. Override it to serve a challenger
    # or to pin an exact version.
    mlflow_model_name: str = Field(
        default=DEFAULT_MODEL,
        description="Model to serve, as <name>@<alias> or <name>/<version>."
    )

    @field_validator("mlflow_model_name", mode="before")
    @classmethod
    def _default_when_blank(cls, value: object) -> object:
        """
        Treat an empty entry as absent.

        .env.example ships every key with an empty value, so copying it and
        filling in only what you need leaves this one as "". Without this the
        service would ask the registry for a model with no name.
        """
        if value is None or not str(value).strip():
            return DEFAULT_MODEL
        return value

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
