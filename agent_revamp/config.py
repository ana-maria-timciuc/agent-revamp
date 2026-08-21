from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agent_revamp.preprocess.process_class import ProcessClass

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
    embedding_model: str = Field(
        default="text-embedding-3-small", alias="EMBEDDING_MODEL"
    )
    rerank_max_retries: int = Field(default=2, alias="RERANK_MAX_RETRIES")
    rerank_retry_top_k_multiplier: int = Field(default=2, alias="RERANK_RETRY_TOP_K_MULTIPLIER")
    max_validation_retries: int = Field(default=2, alias="MAX_VALIDATION_RETRIES")

    default_account_id: int = Field(default=1, alias="DEFAULT_ACCOUNT_ID")
    process_class: ProcessClass = Field(default="penny", alias="PROCESS_CLASS")
    penny_mcp_url: str = Field(
        default="http://localhost:8011/mcp/", alias="PENNY_MCP_URL"
    )
    max_tool_iterations: int = Field(default=12, alias="MAX_TOOL_ITERATIONS")
    tool_call_timeout_seconds: float = Field(default=45.0, alias="TOOL_CALL_TIMEOUT_SECONDS")
    state_dir: str = Field(default="state/", alias="STATE_DIR")
    max_history_messages: int = Field(default=12, alias="MAX_HISTORY_MESSAGES")
    max_message_chars: int = Field(default=3000, alias="MAX_MESSAGE_CHARS")

    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_tools_collection: str = Field(default="agent_tools", alias="QDRANT_TOOLS_COLLECTION")
    qdrant_skills_collection: str = Field(default="agent_skills", alias="QDRANT_SKILLS_COLLECTION")
    qdrant_top_k: int = Field(default=8, alias="QDRANT_TOP_K")
    skills_dir: str = Field(default="skills/", alias="SKILLS_DIR")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # ── RAG / intent preprocessing ─────────────────────────────────────────
    intent_taxonomy: str = Field(
        default=DEFAULT_INTENT_TAXONOMY, alias="INTENT_TAXONOMY"
    )
    skills_dirs: str = Field(default="", alias="SKILLS_DIRS")
    # Comma-separated agent_type=/path/to/skills pairs, e.g.
    # "penny=../realbooks-agents/agents/penny/skills,dollar_bill=../realbooks-agents/agents/dollar_bill/skills,"
    # "uncle_sam=../realbooks-agents/agents/uncle_sam/skills". Empty falls back to defaults below.

    # Auth into /agent/* (mirrors realbooks-agents): the Cognito Bearer token is verified
    # via JWKS and account_id/permissions/role_key are derived from the users SAPI — never
    # from the request body. Same env names as the original service, so the same credentials
    # work unchanged.
    cognito_jwks_url: str = Field(default="", alias="COGNITO_JWKS_URL")
    users_domain: str = Field(default="", alias="USERS_DOMAIN")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_openai_base_url(self) -> str | None:
        return (
            self.openai_base_url or _PROVIDER_BASE_URLS.get(self.llm_provider) or None
        )


settings = Settings()
