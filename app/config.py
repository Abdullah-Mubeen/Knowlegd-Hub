from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    # Core API keys
    OPENAI_API_KEY: SecretStr
    PINECONE_API_KEY: SecretStr

    # Pinecone
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str = Field(alias="PINECONE_INDEX")
    PINECONE_HOST: Optional[str] = None
    PINECONE_DIMENSION: int = Field(1024, gt=0, description="Embed vector dim")
    PINECONE_METRIC: str = Field("cosine", description="Similarity metric")

    # OpenAI models
    OPENAI_EMBEDDING_MODEL: str
    OPENAI_CHAT_MODEL: str = Field(alias="OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.7, ge=0.0, le=1.0)
    OPENAI_MAX_TOKENS: int = Field(1000, gt=0)
    
    # Smart Model Routing - Cost Optimization
    ENABLE_SMART_ROUTING: bool = Field(True, description="Enable intelligent model selection")
    SIMPLE_QUERY_MODEL: str = Field("gpt-4o-mini", description="Fast, cheap model for simple queries")
    COMPLEX_QUERY_MODEL: str = Field("gpt-4", description="Powerful model for complex queries")
    SIMPLE_QUERY_THRESHOLD: int = Field(30, description="Max words for simple query classification")
    COMPLEX_KEYWORDS: str = Field(
        "analyze,compare,evaluate,explain,why,reasoning,recommend,strategy,plan",
        description="Comma-separated keywords indicating complex queries"
    )

    # MongoDB
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017/knowledge_hub",
        description="MongoDB connection string"
    )

    # JWT Authentication
    SECRET_KEY: str 
    JWT_ALGORITHM: str = "HS256"

    # App meta
    APP_NAME: str = "KnowledgeBaseAPI"
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = True,
        extra = "ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()