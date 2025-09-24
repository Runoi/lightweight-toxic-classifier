import torch
import torch.nn.functional as F
from .preprocessor import TextPreprocessor 
from .schemas import PredictionOutput 

class ToxicityClassifier:
    """
    Инкапсулирует логику ML-модели:
    - Загрузка модели
    - Запуск предсказания (inference)
    - Форматирование результата
    """
    def __init__(self, model_path: str, preprocessor: TextPreprocessor):
        self.model = self._load_model(model_path)
        self.preprocessor = preprocessor
        print(f"ToxicityClassifier инициализирован с моделью: {model_path}")

    def _load_model(self, path: str):
        model = torch.jit.load(path)
        model.eval()
        return model

    def predict(self, text: str, threshold: float = 0.5) -> PredictionOutput:
        """Основной метод: принимает текст и порог, возвращает Pydantic-схему."""
        input_tensor = self.preprocessor.process(text)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = F.softmax(logits, dim=1).squeeze()
        
        if probabilities.dim() == 0:
            score_toxic = probabilities.item()
        else:
            score_toxic = probabilities[1].item()
            
        is_toxic = score_toxic > threshold
        
        return PredictionOutput(
            is_toxic=is_toxic,
            toxic_score=score_toxic,
            text=text
        )