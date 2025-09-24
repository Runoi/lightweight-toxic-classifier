# /api/preprocessor.py

import torch
import json
import re
from natasha import Segmenter, Doc
from typing import List

class TextPreprocessor:
    """
    Инкапсулирует всю логику предобработки текста:
    - Загрузка словаря
    - Токенизация и очистка
    - Преобразование в тензор
    """
    def __init__(self, vocab_path: str, max_len: int):
        self.vocab = self._load_vocab(vocab_path)
        self.max_len = max_len
        self.segmenter = Segmenter()
        print(f"TextPreprocessor инициализирован со словарем размером {len(self.vocab)}.")

    def _load_vocab(self, path: str) -> dict:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _robust_tokenizer(self, text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip(): return []
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+', '', text)
        doc = Doc(text)
        doc.segment(self.segmenter)
        tokens = [token.text.lower() for token in doc.tokens if token.text.isalpha()]
        return tokens

    def process(self, text: str) -> torch.Tensor:
        """Основной метод: принимает сырой текст, возвращает готовый тензор."""
        tokens = self._robust_tokenizer(text)
        indexed_tokens = [self.vocab.get(token, self.vocab.get('<unk>', 1)) for token in tokens]
        padding = [self.vocab.get('<pad>', 0)] * (self.max_len - len(indexed_tokens))
        indexed_tokens = indexed_tokens[:self.max_len] + padding[:max(0, self.max_len - len(indexed_tokens))]
        return torch.tensor(indexed_tokens, dtype=torch.long).unsqueeze(0)