FROM registry.astralinux.ru/library/astra/ubi18-python311:latest

# Добавляем репозитории AstraLinux
RUN echo "deb https://dl.astralinux.ru/astra/stable/orel/repository orel main contrib non-free" >> /etc/apt/sources.list

# Устанавливаем необходимые утилиты
RUN apt-get update && \
    apt-get install -y poppler-utils \
    tesseract-ocr \ 
    tesseract-ocr-rus \
    libgl1-mesa-glx \
    python3-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Обновляем системные pip и setuptools с флагом --break-system-packages
RUN python3 -m pip install --upgrade --force-reinstall --break-system-packages "pip==25.3" "setuptools==80.9.0"

# Удаляем системные пакеты после обновления
RUN apt-get remove -y python3-pip python3-setuptools python3-wheel || true

# Создаем пользователя
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Базовая папка
WORKDIR /app

# Копируем requirements.txt ДО переключения пользователя
COPY requirements.txt .

# Создаем виртуальное окружение и устанавливаем зависимости как ROOT
RUN python3 -m venv venv && \
    venv/bin/python -m ensurepip --upgrade && \
    # Сначала обновляем pip
    venv/bin/python -m pip install --upgrade pip && \
    venv/bin/pip install --no-cache-dir -r requirements.txt

# Создаем файл лога (теперь у appuser есть права на запись)
RUN touch /app/log.log && chmod 666 /app/log.log

# Меняем владельца всей директории /app на appuser
#RUN chown -R appuser:appuser /app
RUN chmod -R a+rw /app

# Копируем остальные файлы
COPY  api_client.py .
COPY ocr_processor.py .
COPY  logger.py .
COPY  main.py .
#COPY sub_modules/tesstrain/data/rus/rus.lstm /usr/share/tesseract-ocr/5/tessdata

#Переключаемся на пользователя
USER appuser

# Запускаем приложение
CMD ["venv/bin/python", "main.py"]
#CMD ["tail", "-f", "/dev/null"]