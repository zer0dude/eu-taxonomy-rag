from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "taxonomy"
    db_user: str = "taxonomy"
    db_password: str = "taxonomy"

    @computed_field
    @property
    def dsn(self) -> str:
        """psycopg3 connection string format."""
        return (
            f"host={self.db_host} port={self.db_port} "
            f"dbname={self.db_name} user={self.db_user} "
            f"password={self.db_password}"
        )

    # Embeddings — all-MiniLM-L6-v2 outputs 384-dim vectors, matches the SQL schema
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384  # must match vector(N) in 00_init.sql

    # LLM provider selection: "ollama" | "anthropic" | "openai"
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Ingestion defaults — can be overridden per experiment
    default_chunk_size: int = 512       # words (NaiveChunker uses whitespace split, not a tokenizer)
    default_chunk_overlap: int = 50     # words
    default_chunker: str = "naive"      # "naive" | "structural" | "hierarchical"

    @computed_field
    @property
    def litellm_model_string(self) -> str:
        """Return the LiteLLM model string for the configured provider.

        Ollama:    "ollama/llama3.2"
        Anthropic: "claude-haiku-4-5-20251001"  (bare name; routed via ANTHROPIC_API_KEY)
        OpenAI:    "gpt-4o-mini"
        """
        if self.llm_provider == "ollama":
            return f"ollama/{self.ollama_model}"
        if self.llm_provider == "anthropic":
            return self.anthropic_model
        if self.llm_provider == "openai":
            return self.openai_model
        raise ValueError(
            f"Unknown llm_provider: {self.llm_provider!r}. "
            "Choose: 'ollama', 'anthropic', or 'openai'."
        )


settings = Settings()
