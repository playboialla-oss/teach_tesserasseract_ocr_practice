import requests
from logger import logger
import uuid
from typing import Optional, List, Callable
from ocr_processor import OCRProcessor

class APIClient:
    def __init__(self, username, password, ocr_processor=None, debug:bool=False, is_production:bool=True):
        self.username = username
        self.password = password
        self.token = None
        self.ids = []
        self.files = []
        self.session = requests.Session()
        self.ocr = ocr_processor
        self.debug = debug
        self.is_production = is_production
        self.authenticate_url = None

    def authenticate(self, authenticate_url):
        """Получение токена авторизации"""
        self.authenticate_url = authenticate_url
        try:
            payload = {
                "loginName": self.username,
                "password": self.password,
                "passwordType": "plainText",
                "roleID": 0,
                "accessLevelID": 0
            }
            logger.info(f"Аутентификация на {authenticate_url}")
            #response = self.session.post(self.authenticate_url, json=payload)
            response = self.session.post(authenticate_url, json=payload)
            response.raise_for_status()
            data = response.json()
            self.token = data.get("accessToken")
            if self.token:
                self.session.headers.update({"Authorization": f"Bearer {self.token}"})
                logger.info("Аутентификация успешна")
                return True
            else:
                logger.error("Токен не найден в ответе")
                return False
        except Exception as e:
            logger.error(f"Ошибка аутентификации: {e}")
            return False

    def _request_with_reauth(self, method, url, **kwargs):
        """
        Выполняет HTTP-запрос, в случае 401 пытается переавторизоваться и повторить.
        Возвращает response.
        """
        try:
            response = method(url, **kwargs)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.warning("Получен 401, выполняю повторную аутентификацию")
                if self.authenticate(self.authenticate_url):
                    # повторяем запрос
                    response = method(url, **kwargs)
                    response.raise_for_status()
                    return response
            raise

    def get_objects_id(self, objects_url):
        """Получение списка объектов с авто-реавторизацией"""
        try:
            payload = {
                "objectTypeId": -1,
                "attributeIdsToSelect": [-2],
                "conditions": [
                    {
                        "attributeId": 30357,
                        "relationalOperator": "Equal",
                        "logicalOperator": "none",
                        "groupID": 0,
                        "value": "false",
                        "content": "text"
                    }
                ]
            }
            logger.info(f"Запрос списка объектов: {objects_url}")
            response = self._request_with_reauth(self.session.post, objects_url, json=payload)
            objects = response.json()
            logger.info(f"Получено объектов: {len(objects) if isinstance(objects, list) else 'неизвестно'}")
            self.ids = [obj["objectId"] for obj in objects if "objectId" in obj]
            return self.ids
        except Exception as e:
            logger.error(f"Ошибка получения списка объектов: {e}")
            return None

    def get_blob_id(self, blob_url_template, extensions=None):
        """Сбор метаданных файлов с авто-реавторизацией"""
        if extensions is None:
            extensions = [".dwg.pdf"]
        self.files.clear()
        for obj_id in self.ids:
            try:
                blob_url = blob_url_template.format(obj_id=obj_id)
                logger.info(f"Получение blob id: {blob_url}")
                response = self._request_with_reauth(self.session.get, blob_url)
                metadata = response.json()
                if not isinstance(metadata, dict):
                    logger.debug(f"Неверный тип ответа для объекта {obj_id}: {type(metadata)}")
                    continue
                for attr in metadata.get("attributes", []):
                    if attr.get("attributeFieldType") != "ftFile":
                        continue
                    for file_info in attr.get("fileInfoCollection", []):
                        fname = file_info.get("fileName", "")
                        if not fname:
                            continue
                        if any(fname.lower().endswith(ext) for ext in extensions):
                            self.files.append({
                                "obj_id": obj_id,
                                "blob_id": file_info.get("blobId"),
                                "file_name": fname
                            })
            except Exception as e:
                logger.error(f"Ошибка обработки объекта {obj_id}: {e}")
        logger.info(f"Найдено файлов для обработки: {len(self.files)}")
        return self.files

    def get_file(self, download_url_template, obj_id, blob_id):
        """Скачивание файла с авто-реавторизацией"""
        try:
            download_url = download_url_template.format(obj_id=obj_id, blob_id=blob_id)
            logger.debug(f"Скачивание файла: {download_url}")
            response = self._request_with_reauth(self.session.get, download_url)
            if response.content[:4] == b'%PDF' or 'application/pdf' in response.headers.get('content-type', ''):
                logger.info(f"Файл obj={obj_id}, blob={blob_id} скачан, размер: {len(response.content)} байт")
                return response.content
            else:
                logger.warning(f"Скачанный файл obj={obj_id}, blob={blob_id} не является PDF")
                return None
        except Exception as e:
            logger.error(f"Ошибка скачивания файла obj={obj_id}, blob={blob_id}: {e}")
            return None

    def send_content(self, send_url_template, obj_id, text: list, first_attribute_id=30357, last_attribute_id=30356):
        try:
            send_url = send_url_template.format(obj_id=obj_id)
            payload = [
                {
                    "attributeID": first_attribute_id,
                    "values": ["true"]
                },
                {
                    "attributeID": last_attribute_id,
                    "values": text
                }
            ]
            logger.info(f"Отправка текста для объекта {obj_id}: {send_url}")
            response = self._request_with_reauth(self.session.post, send_url, json=payload)
            logger.info(f"Текст для объекта {obj_id} успешно отправлен")
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки текста для объекта {obj_id}: {e}")
            return False

    def process_all_files(self, download_url_template, send_url_template,
                          selection_keywords=None, exclude_keywords=None, max_files=None):
        """Потоковая обработка файлов текущего списка self.files"""
        if not self.files:
            logger.info("Нет файлов для обработки")
            return

        if not self.ocr:
            logger.error("OCR-процессор не задан")
            return

        files_to_process = self.files[:max_files] if max_files else self.files
        total = len(files_to_process)
        logger.info(f"Начинаю обработку {total} файлов...")

        for idx, file_info in enumerate(files_to_process, 1):
            obj_id = file_info["obj_id"]
            blob_id = file_info["blob_id"]
            fname = file_info.get("file_name", f"неизвестно_{uuid.uuid4().hex[:8]}")
            logger.info(f"--- Обработка {idx}/{total}: объект {obj_id}, файл '{fname}' ---")

            pdf_bytes = self.get_file(download_url_template, obj_id, blob_id)
            if not pdf_bytes:
                continue

            try:
                raw_text = self.ocr.extract_text(pdf_bytes, fname)
                logger.info(f"Распознано символов: {len(raw_text)}")
            except Exception as e:
                logger.error(f"Ошибка OCR: {e}")
                raw_text = ""
            #Если не Production не отправляем сообщение на сервер IPS
            raw_stripped = raw_text.strip()
            if not raw_stripped:
                logger.info("Текст пуст, отправка не требуется")
                continue
                        
            # TODO(DELAGREEN): Загрушка
            cleared_text = raw_stripped.replace("$", "3").strip()
            text = [cleared_text]
            try:
                self.send_content(send_url_template, obj_id, text)
                logger.debug(f"Распознанный текст: {text}")
            except Exception as e:
                logger.error(f"Ошибка отправки для {obj_id}: {e}")

            #явно очищаем память
            del pdf_bytes, raw_text

        logger.info("Обработка завершена")