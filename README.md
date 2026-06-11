# Клонирование репозитория 
Что бы клонировать репозиторий с сылками на чужой репозиторий необходимо выполнить команду 

    git clone --recurse-submodules https://github.com/DELAGREEN/teach_tesserasseract.git

# Сборка репозитория
Первым этапом является добавления сторонних репозиториев в свой проект с помощью команды  

Добавляем tesstrain:

    git submodule add https://github.com/tesseract-ocr/tesstrain.git

# Чел на Youtube 
    https://www.youtube.com/watch?v=KE4xEzFGSU8&ab_channel=GabrielGarcia

# Обучение

    cd sub_modules/tesstrain/

<br>

    TESSDATA_PREFIX=../tesseract/tessdata make training MODEL_NAME=rus START_MODEL=rus TESSDATA=../tesseract/tessdata MAX_ITERATIONS=10000

<br>

    mkdir langdata

<br>

    cd langdata

<br>

    git clone https://github.com/tesseract-ocr/langdata_lstm.git/

<br>

    make unicharset lists proto-model tesseract-langdata training MODEL_NAME=rus MAX_ITERATIONS=100000

Распаковать файлы из папки языка в root-langdata

# Переменные окружения 
    services:
    app:
        build: .
        environment:
        - LOG_LEVEL=DEBUG
        - LOG_FILE=/logs/app.log

# Сборка Docker контейнера
<br> 

**ВНИМАНИЕ**
<br>
**Бывает так что проект собирается не с первого раза, нужно просто запустить сборку повторно.**
- Что бы собрать проект нужно запустить `docker-compose` с файлом `docker-compose.yml` проект сам соберется
- В файле .env нужно указать путь монтирования к хост машине
- **Если хотите взаимодействовать с файлами в хостмашине от обычного пользователя, раскоментируйте строки `UID` и `GID`в .env файле.

<br>

# Пересборка
## Останавливаем и удаляем все контейнеры
    docker-compose down --rmi all --volumes --remove-orphans

## Удаляем все Docker образы, контейнеры и volumes
    docker system prune -a -f
<br>

    docker volume prune -f

## Пересобираем образ с чистого листа
    docker-compose build --no-cache

## Запускаем
    docker-compose up -d

# Пример как зайти в контейнер под root
    docker-compose run --rm --user root container_name /bin/bash
<br>

    docker exec -it -u root container_name /bin/bash


# Tesseract

### Место нахождение файла .trainedata
После переобучения закинуть сюда

    /usr/share/tesseract-ocr/5/tessdata