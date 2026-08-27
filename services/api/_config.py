from functools import lru_cache
from pydantic import Field, Secret, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAKUTEN__API__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
        str_strip_whitespace=True
    )
    jwt_algorithm: str = Field(default=..., description="JWT algorithm.")
    jwt_secret: Secret[str] = Field(default=..., description="JWT secret.")
    jwt_expiration_in_minutes: int = Field(default=..., description="Lifetime of a JWT token in minutes.")

    inference_host: str = Field(default=..., description="Name of the inference service container at runtime.")
    inference_port: int = Field(default=..., description="Port exposed by the inference service container at runtime.")

    database_host: str = Field(default=..., description="Name of the database service container at runtime.")
    database_port: int = Field(default=..., description="Port exposed by the database service container at runtime.")
    database_name: str = Field(default=..., description="Name of the database to access.")
    database_user: Secret[str] = Field(default=..., description="Username for the database.")
    database_password: Secret[str] = Field(default=..., description="Password for the database.")

    mlflow_host: str = Field(default=..., description="Name of the mlflow service container at runtime.")
    mlflow_port: int = Field(default=..., description="Port exposed by the mlflow service container at runtime.")

    @computed_field
    @property
    def inference_base_url(self) -> str:
        return f"http://{self.inference_host}:{self.inference_port}"

    @computed_field
    @property
    def database_url(self) -> str:
        prefix = "postgresql+asyncpg://"
        user = self.database_user.get_secret_value()
        password = self.database_password.get_secret_value()
        name = self.database_name
        return f"{prefix}{user}:{password}@{self.database_host}:{self.database_port}/{name}"

    @computed_field
    @property
    def mlflow_server_uri(self) -> str:
        return f"http://{self.mlflow_host}:{self.mlflow_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
