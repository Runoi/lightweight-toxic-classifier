
from pydantic import BaseModel
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    """
    Класс для хранения всех настроек и констант.
    Теперь строит абсолютные пути, что делает его надежным.
    """
    # Пути к артефактам модели
    MODEL_VERSION: str = "v1.0"
    # Теперь базовый путь - это абсолютный путь к корню проекта
    BASE_MODEL_PATH: Path = PROJECT_ROOT / "models"
    
    # Параметры, необходимые для предобработки
    MAX_SEQ_LEN: int = 128

    @property
    def model_path(self) -> str:
        # Используем pathlib для сборки путей
        return str(self.BASE_MODEL_PATH / self.MODEL_VERSION / "solo_cnn_int8.pth")

    @property
    def vocab_path(self) -> str:
        return str(self.BASE_MODEL_PATH / self.MODEL_VERSION / "vocab.json")

# Создаем один экземпляр настроек, который будет использоваться во всем приложении
settings = Settings()