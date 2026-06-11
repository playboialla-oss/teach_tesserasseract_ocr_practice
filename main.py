import configparser
import os
import time
import signal
import sys
from dotenv import load_dotenv
from ocr_processor import OCRProcessor
from api_client import APIClient
from logger import logger

# ----------------------- Загрузка переменных окружения -----------------------
load_dotenv()

# ----------------------- Загрузка конфигурации -----------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.ini")
with open(CONFIG_PATH, encoding="utf-8") as f:
    config_content = os.path.expandvars(f.read())

config = configparser.ConfigParser(interpolation=None)
config.read_string(config_content)

def get_config(section, key, fallback=None, type=str):
    """Получить значение сначала из переменной окружения, потом из config.ini."""
    env_key = f"{section}_{key}".upper()           # API_BASE_URL, OCR_DPI ...
    env_value = os.environ.get(env_key)
    if env_value is not None:
        if type == bool:
            return env_value.lower() in ('true', '1', 'yes')
        return type(env_value)
    # если нет в окружении – берём из config.ini
    if type == bool:
        return config.getboolean(section, key, fallback=fallback)
    elif type == int:
        return config.getint(section, key, fallback=fallback)
    elif type == float:
        return config.getfloat(section, key, fallback=fallback)
    else:
        return config.get(section, key, fallback=fallback)

#Общие настройки
EXTENSIONS_FILE = [ext.strip() for ext in get_config("Processing", "extensions", fallback=".dwg.pdf").split(",") if ext.strip()]
IS_PRODUCTION = get_config("Processing", "is_production", fallback=True, type=bool)
SAVE_IMAGES = get_config("Processing", "save_images", fallback=False, type=bool)
DEBUG = get_config("Processing", "debug", fallback=False, type=bool)

# Параметры API
API_BASE_URL = get_config("API", "base_url")
API_AUTH_ENDPOINT = get_config("API", "authenticate_endpoint")
API_GET_OBJECTS_ID = get_config("API", "get_objects_id")
API_GET_BLOB_ID = get_config("API", "get_blob_id")
API_DOWNLOAD_ENDPOINT = get_config("API", "download_endpoint")
API_SEND_CONTENT = get_config("API", "send_content")
API_AUTH_USERNAME = get_config("API", "username")
API_AUTH_PASSWORD = get_config("API", "password")

# Параметры OCR
DPI = get_config("OCR", "dpi", fallback=300, type=int)
LANG = get_config("OCR", "lang", fallback="rus", type=str)
PSM = get_config("OCR", "psm", fallback=6, type=int)
USE_ADVANCED_RECOGNITION = get_config("OCR", "use_advanced_recognition", fallback=False, type=bool)
MIN_AREA_RATIO = get_config("OCR", "min_area_ratio", fallback=0.0005, type=float)
WIDTH_TOLERANCE = get_config("OCR", "width_tolerance", fallback=0.3, type=float)
KERNEL_WIDTH = get_config("OCR", "kernel_width", fallback=20, type=int)
KERNEL_HEIGHT = get_config("OCR", "kernel_height", fallback=30, type=int)
DILATION_ITERATIONS = get_config("OCR", "dilation_iterations", fallback=3, type=int)
CHAR_WHITELIST = get_config("OCR", "char_whitelist", fallback="", type=str)

#OS Settings 
RESULTS_DIR = get_config("Paths", "results_dir", fallback="results", type=str)

# Ключевые слова
SELECTION_KEYWORDS = [kw.strip() for kw in get_config("Selection", "keywords", fallback="", type=str).split(",") if kw.strip()]
EXCLUDE_KEYWORDS = [kw.strip() for kw in get_config("Exclusion", "exclude_keywords", fallback="", type=str).split(",") if kw.strip()]

# Интервал опроса
POLLING_INTERVAL = get_config("API", "polling_interval", fallback=60, type=int)

# Максимальное количество файлов за один цикл (можно поставить 0 для без ограничений)
MAX_FILES_PER_CYCLE = get_config("Processing", "max_files_per_cycle", fallback=0, type=int)

#Защита конфигурации
if not IS_PRODUCTION and not DEBUG:
    raise RuntimeError(
        "Недопустимая конфигурация: оба флага is_production и debug установлены в False. "
        "Обработка не будет выполнять ни отправку, ни логирование."
    )

os.makedirs(RESULTS_DIR, exist_ok=True)

# Сборка URL
objects_url = f"{API_BASE_URL}{API_GET_OBJECTS_ID}"
authenticate_url_template  = f"{API_BASE_URL}{API_AUTH_ENDPOINT}"
blob_url_template = f"{API_BASE_URL}{API_GET_BLOB_ID}"
download_url_template = f"{API_BASE_URL}{API_DOWNLOAD_ENDPOINT}"
send_url_template = f"{API_BASE_URL}{API_SEND_CONTENT}"

ocr = OCRProcessor(
        dpi=DPI, lang=LANG, psm=PSM,
        min_area_ratio=MIN_AREA_RATIO,
        width_tolerance=WIDTH_TOLERANCE,
        save_images=SAVE_IMAGES, save_dir=RESULTS_DIR,
        use_advanced=USE_ADVANCED_RECOGNITION,
        kernel_width=KERNEL_WIDTH,
        kernel_height=KERNEL_HEIGHT,
        dilation_iterations=DILATION_ITERATIONS,
        debug=DEBUG,
        production_mode=IS_PRODUCTION,
        char_whitelist=CHAR_WHITELIST
    )

client = APIClient(
    username=API_AUTH_USERNAME,
    password=API_AUTH_PASSWORD,
    ocr_processor=ocr,
    debug=DEBUG
)


# ----------------------- Основной цикл -----------------------
if __name__ == "__main__":
    logger.info("Сервис запущен")
    # Первичная аутентификация
    if not client.authenticate(authenticate_url_template):
        logger.error("Не удалось авторизоваться при старте, выход")
        sys.exit(1)

    def graceful_shutdown(signum, frame):
        logger.info("Получен сигнал завершения, выход")
        sys.exit(0)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    while True:
        try:
            logger.info("--- Новый цикл опроса ---")
            # Получаем список необработанных объектов
            ids = client.get_objects_id(objects_url)
            if not ids:
                logger.info("Нет объектов для обработки")
            else:
                # Получаем blobId нужных файлов
                client.get_blob_id(blob_url_template, extensions=EXTENSIONS_FILE)
                # Обрабатываем (можно ограничить количество за цикл)
                max_files = MAX_FILES_PER_CYCLE if MAX_FILES_PER_CYCLE > 0 else None
                client.process_all_files(
                    download_url_template=download_url_template,
                    send_url_template=send_url_template,
                    selection_keywords=SELECTION_KEYWORDS,
                    exclude_keywords=EXCLUDE_KEYWORDS,
                    max_files=max_files
                )
        except KeyboardInterrupt:
            logger.info("Ручная остановка сервиса")
            break
        except Exception as e:
            logger.exception(f"Критическая ошибка в цикле: {e}")
            # Ждем немного перед следующей попыткой, чтобы не заспамить логи
            time.sleep(10)
            continue

        logger.info(f"Ожидание {POLLING_INTERVAL} секунд до следующего опроса...")
        time.sleep(POLLING_INTERVAL)