# Шаг 1: Базовый образ Python
FROM python:3.11-slim

# Шаг 2: Установка рабочей директории
WORKDIR /app

# Шаг 3: Копирование и установка зависимостей
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Шаг 4: Копирование кода приложения и моделей
COPY . .

# Шаг 5: Указание порта, на котором будет работать приложение
EXPOSE 8000

# Шаг 6: Команда для запуска API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]