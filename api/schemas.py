from pydantic import BaseModel, Field

class TextInput(BaseModel):
    """Схема для входящего текста."""
    text: str = Field(..., example="Какой прекрасный сегодня день!", description="Текст комментария для анализа.")

class PredictionOutput(BaseModel):
    """Схема для ответа с предсказанием."""
    is_toxic: bool = Field(..., example=False, description="True, если комментарий токсичен, иначе False.")
    toxic_score: float = Field(..., example=0.1234, description="Оценка токсичности от модели (вероятность от 0.0 до 1.0).")
    text: str = Field(..., example="Какой прекрасный сегодня день!", description="Оригинальный текст, который был проанализирован.")