from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "",
    "deepseek": "https://api.deepseek.com",
}

DEFAULT_INTENT_TAXONOMY = "analytics, reporting, lookup, create_record, market_research, meta_about, off_domain"


class Settings(BaseSettings):
    openai_key: str = Field(default="", alias="OPENAI_KEY")
    openai_model: str = Field(default="gpt-5", alias="OPENAI_MODEL")
    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")

    penny_mcp_url: str = Field(
        default="http://localhost:8011/mcp/", alias="PENNY_MCP_URL"
    )
    max_tool_iterations: int = Field(default=12, alias="MAX_TOOL_ITERATIONS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_openai_base_url(self) -> str | None:
        return (
            self.openai_base_url or _PROVIDER_BASE_URLS.get(self.llm_provider) or None
        )


settings = Settings()
