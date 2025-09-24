# /api/main.py

from fastapi import FastAPI, Depends
from .schemas import TextInput, PredictionOutput
from .classifier import ToxicityClassifier
from .preprocessor import TextPreprocessor
from .config import settings



app = FastAPI(
    title="Toxic Comment Classifier API",
    description="API для определения токсичности текста",
    version=settings.MODEL_VERSION
)




preprocessor = TextPreprocessor(vocab_path=settings.vocab_path, max_len=settings.MAX_SEQ_LEN)
classifier = ToxicityClassifier(model_path=settings.model_path, preprocessor=preprocessor)

def get_classifier() -> ToxicityClassifier:
    """FastAPI зависимость, которая возвращает наш единственный экземпляр классификатора."""
    return classifier



@app.get("/", summary="Проверка работы API", tags=["General"])
def read_root():
    return {"status": "ok", "message": "API is running."}

@app.post("/predict", response_model=PredictionOutput, summary="Предсказать токсичность текста", tags=["Machine Learning"])
def predict(
    request: TextInput,
    model: ToxicityClassifier = Depends(get_classifier),
    threshold: float = 0.5
):
    """
    Принимает текст и возвращает предсказание токсичности.
    """
    return model.predict(text=request.text, threshold=threshold)