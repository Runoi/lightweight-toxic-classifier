# Артефакты модели: Версия 1.0 (v1.0)

Этот каталог содержит все необходимые артефакты для использования и дообучения модели классификации токсичных комментариев **версии 1.0**.

## Описание файлов

-   `solo_cnn_fp32.pth`: **"Мастер-модель"** в формате FP32 (32-битные числа). Используется как основа для дальнейшего дообучения на новых данных.
-   `solo_cnn_int8.pth`: **Квантованная (INT8) и оптимизированная модель** для инференса. **Используйте этот файл в продакшене (API, бот).**
-   `vocab.json`: Словарь, сопоставляющий токены с их индексами. Необходим для предобработки текста и связан с этой версией модели.
-   `readme.md`: Этот файл с документацией.

---

## Характеристики модели

### Метрики качества (на валидационной выборке)

| Класс         | Precision | Recall | F1-score |
|---------------|-----------|--------|----------|
| Нетоксичный (0) | 0.78      | 0.88   | 0.83     |
| **Токсичный (1)** | **0.68**  | **0.51**   | **0.58** |

-   **Accuracy (общая точность):** 0.76
-   **Размер квантованной модели:** ~5 МБ

### Параметры и конфигурация

| Параметр        | Значение     | Описание                                  |
|-----------------|--------------|-------------------------------------------|
| `VOCAB_SIZE`    | 20000        | Размер словаря                            |
| `MAX_SEQ_LEN`   | 128          | Максимальная длина последовательности токенов |
| `EMBED_DIM`     | 256          | Размерность векторов слов (эмбеддингов)     |
| `NUM_FILTERS`   | 256          | Количество сверточных фильтров            |
| `KERNEL_SIZES`  | [3, 4, 5]    | Размеры ядер сверток                      |
| `DROPOUT_RATE`  | 0.3          | Коэффициент dropout для регуляризации      |

---

## Краткая методология создания

Модель была создана с нуля ("сольная" CNN) и прошла через несколько итераций отладки.
-   **Предобработка:** Вместо лемматизации (которая не работала в среде Kaggle) используется надежная токенизация с помощью `natasha.Segmenter`.
-   **Архитектура:** Для стабилизации обучения и совместимости с квантизацией в архитектуру CNN были добавлены `nn.BatchNorm1d`, `nn.ReLU` (как модуль), а также "мосты" `QuantStub` и `DeQuantStub`.
-   **Оптимизация:** Модель была квантована с использованием Post Training Static Quantization, включая явное **слияние (fusion) модулей `Conv-BN-ReLU`** для максимальной производительности.

---

## Пайплайн для инференса (Inference)

### Важность настройки порога (threshold)

Модель выдает вероятность токсичности (`toxic_score`). Порог (`threshold`) — это "планка уверенности", которую вы устанавливаете, чтобы принять решение (`is_toxic = toxic_score > threshold`). Значение `0.5` — это стандарт, но его **нужно настраивать** под вашу задачу:
-   **Низкий порог (e.g., `0.35`):** Находит больше токсичных комментариев (`Recall ↑`), но чаще ошибается (`Precision ↓`). Хорошо для предварительной модерации.
-   **Высокий порог (e.g., `0.7`):** Меньше ложных срабатываний (`Precision ↑`), но пропускает больше токсичных комментариев (`Recall ↓`). Хорошо для автоматического бана.

### Код для использования

**Минимальные зависимости:** `torch`, `natasha`

Полный, самодостаточный код для предсказания:

```python
import torch
import torch.nn.functional as F
import json
import re
from natasha import Segmenter, Doc

# --- ШАГ 1: Определение функций предобработки (идентичны тем, что при обучении) ---

segmenter = Segmenter()

def robust_tokenizer(text):
    if not isinstance(text, str) or not text.strip(): return []
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    doc = Doc(text)
    doc.segment(segmenter)
    tokens = [token.text.lower() for token in doc.tokens if token.text.isalpha()]
    return tokens

def tokenize_text_robust(text, vocab, max_len):
    tokens = robust_tokenizer(text)
    indexed_tokens = [vocab.get(token, vocab.get('<unk>', 1)) for token in tokens]
    padding = [vocab.get('<pad>', 0)] * (max_len - len(indexed_tokens))
    indexed_tokens = indexed_tokens[:max_len] + padding[:max(0, max_len - len(indexed_tokens))]
    return torch.tensor(indexed_tokens, dtype=torch.long)

# --- ШАГ 2: Загрузка артефактов ---

try:
    model = torch.jit.load('solo_cnn_int8.pth') # Загружаем квантованную модель
    model.eval()
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print("Модель и словарь v1.0 успешно загружены.")
except FileNotFoundError:
    print("Ошибка: Файлы 'solo_cnn_int8.pth' и 'vocab.json' должны находиться в той же папке.")
    exit()

# --- ШАГ 3: Функция для предсказания ---

def predict(text, threshold=0.5): # <-- ЭТОТ ПАРАМЕТР НУЖНО НАСТРАИВАТЬ!
    """Принимает сырой текст и возвращает словарь с результатом."""
    input_ids = tokenize_text_robust(text, vocab, max_len=128).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = F.softmax(logits, dim=1).squeeze()
    
    if probabilities.dim() == 0:
        score_toxic = probabilities.item()
    else:
        score_toxic = probabilities.item()
        
    is_toxic = score_toxic > threshold
    return {"is_toxic": is_toxic, "toxic_score": f"{score_toxic:.4f}"}

# --- ШАГ 4: Пример вызова ---

print("\n--- Тестовые предсказания ---")
print(predict("автор ты просто гений и молодец"))
print(predict("автор ты просто идиот и пишешь чушь"))
print(predict("Это просто худший сервис из всех, что я пробовал.", threshold=0.4))
```

### Жизненный цикл модели

-   **Использование в продакшене:** Используйте `solo_cnn_int8.pth` и `vocab.json`.
-   **Дообучение:** При накоплении новых данных используйте `solo_cnn_fp32.pth` как отправную точку для дообучения. После дообучения создайте новую папку (`v1.1`) с обновленными артефактами.

