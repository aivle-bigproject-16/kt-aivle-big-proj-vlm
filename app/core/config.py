from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # AI Model Settings
    MODEL_ID: str = "Qwen/Qwen3-VL-4B-Instruct"
    DEVICE: str = "cuda"
    DTYPE: str = "bfloat16"
    MAX_NEW_TOKENS: int = 1500

    # S3 Settings
    S3_BUCKET_NAME: str
    S3_ACCESS_KEY: str
    S3_SECRET_KEY: str

    class Config:
        env_file = ".env"  # .env 파일에서 값을 읽어옴

# 설정값을 담은 인스턴스 생성
settings = Settings()