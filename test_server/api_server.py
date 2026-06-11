import os
import hashlib
from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import uuid
import io

app = FastAPI(title="Mock Production API with Real Files")

# ---------- Модели данных (без изменений) ----------
class AuthRequest(BaseModel):
    loginName: str
    password: str
    passwordType: str = "plainText"
    roleID: int = 0
    accessLevelID: int = 0

class AuthResponse(BaseModel):
    accessToken: str
    refreshToken: str
    expireTime: str

class Condition(BaseModel):
    attributeId: int
    relationalOperator: str
    logicalOperator: str = "none"
    groupID: int = 0
    value: str
    content: Optional[str] = None

class SelectRequest(BaseModel):
    objectTypeId: int = -1
    attributeIdsToSelect: List[int] = [-2]
    conditions: List[Condition] = []

class AttributeFileInfo(BaseModel):
    blobId: int
    realFileSize: int
    packedFileSize: int
    modifyDate: str
    fileName: str
    arcMethod: str = "notPacked"
    note: str = ""
    fileType: str = "ftNormal"
    authorId: int = 0
    fileStorageId: int = 0

class AttributeValue(BaseModel):
    attributeId: int
    attributeFieldType: str = "ftFile"
    isMultiple: bool = True
    fileInfoCollection: List[AttributeFileInfo] = []

class ObjectFilesResponse(BaseModel):
    objectVersionId: int
    objectType: int
    readOnly: bool
    attributes: List[AttributeValue]

class UpdateAttributeRequest(BaseModel):
    attributeID: int
    values: List[str]

# ---------- Хранилища ----------
sessions: Dict[str, dict] = {}
objects_store: Dict[int, Dict[str, Any]] = {}
custom_attributes: Dict[tuple, List[str]] = {}

# Базовая директория с файлами (настраивается)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOCK_FILES_ROOT = os.path.join(BASE_DIR, "mock_files")

def get_blob_id_from_filename(filename: str) -> int:
    """Преобразует имя файла в стабильный числовой blobId (хеш)"""
    # Используем SHA256 и берём первые 9 цифр (максимум 999,999,999)
    hash_digest = hashlib.sha256(filename.encode()).hexdigest()
    # Преобразуем шестнадцатеричную строку в целое число
    blob_id = int(hash_digest[:8], 16)  # 8 символов = 32 бита, даст до 4 млрд
    return blob_id

def scan_object_files(object_id: int) -> List[AttributeFileInfo]:
    """Сканирует папку mock_files/{object_id}/ и возвращает список файлов с метаданными"""
    object_dir = os.path.join(MOCK_FILES_ROOT, str(object_id))
    if not os.path.isdir(object_dir):
        return []  # Нет папки — нет

    file_infos = []
    for entry in os.listdir(object_dir):
        full_path = os.path.join(object_dir, entry)
        if not os.path.isfile(full_path):
            continue
        stat = os.stat(full_path)
        modify_date = datetime.fromtimestamp(stat.st_mtime).isoformat()
        blob_id = get_blob_id_from_filename(entry)
        file_infos.append(AttributeFileInfo(
            blobId=blob_id,
            realFileSize=stat.st_size,
            packedFileSize=stat.st_size,  # для мока без сжатия
            modifyDate=modify_date,
            fileName=entry,
            arcMethod="notPacked",
            note="",
            fileType="ftNormal",
            authorId=1,
            fileStorageId=0
        ))
    return file_infos

# Инициализация объектов (можно также добавить статические объекты)
def init_objects():
    # Объект 101 (как в примере)
    objects_store[101] = {
        "objectTypeId": 1,
        "objectVersionId": 1001,
        "readOnly": False,
        "attributes": {
            30357: {"values": ["false"], "fieldType": "ftBoolean", "isMultiple": False},
            30356: {"values": ["Исходный текст", "Еще строка"], "fieldType": "ftString", "isMultiple": True},
        }
    }
    # Объект 102
    objects_store[102] = {
        "objectTypeId": 2,
        "objectVersionId": 1002,
        "readOnly": True,
        "attributes": {
            30357: {"values": ["false"], "fieldType": "ftBoolean", "isMultiple": False},
            30356: {"values": ["Исходный текст", "Еще строка"], "fieldType": "ftString", "isMultiple": True},
        }
    }

init_objects()

# ---------- Вспомогательные функции (аутентификация без изменений) ----------
def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization format. Use Bearer <token>")
    token = parts[1]
    if token not in sessions:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    expire = datetime.fromisoformat(sessions[token]["expireTime"])
    if expire < datetime.now():
        del sessions[token]
        raise HTTPException(status_code=401, detail="Token expired")
    return sessions[token]["login"]

def generate_token(login: str) -> str:
    token = str(uuid.uuid4())
    expire_time = datetime.now() + timedelta(hours=1)
    expire_str = expire_time.isoformat()
    sessions[token] = {
        "login": login,
        "expireTime": expire_str,
        "refreshToken": str(uuid.uuid4())
    }
    return token, sessions[token]["refreshToken"], expire_str

# ---------- Эндпоинты ----------
@app.post("/core/api/Auth/authenticate", response_model=AuthResponse)
async def authenticate(auth_data: AuthRequest):
    if not auth_data.loginName or not auth_data.password:
        raise HTTPException(status_code=400, detail="Login and password required")
    access_token, refresh_token, expire = generate_token(auth_data.loginName)
    return AuthResponse(accessToken=access_token, refreshToken=refresh_token, expireTime=expire)

@app.post("/core/api/objects/select")
async def select_objects(request: SelectRequest, auth: str = Depends(verify_token)):
    result = []
    for obj_id, obj_data in objects_store.items():
        match = True
        for cond in request.conditions:
            if cond.relationalOperator != "Equal":
                match = False
                break
            attrs = obj_data.get("attributes", {})
            if cond.attributeId not in attrs:
                match = False
                break
            attr_vals = attrs[cond.attributeId]["values"]
            if cond.value not in [str(v) for v in attr_vals]:
                match = False
                break
        if match:
            result.append({"objectId": obj_id, "attributes": []})
    return result

@app.get("/core/api/files/objects/{object_id}", response_model=ObjectFilesResponse)
async def get_file_list(object_id: int, auth: str = Depends(verify_token)):
    """Возвращает метаданные файлов из реальной папки mock_files/{object_id}/"""
    if object_id not in objects_store:
        raise HTTPException(status_code=404, detail="Object not found")
    obj = objects_store[object_id]
    
    # Сканируем папку с файлами для этого объекта
    file_infos = scan_object_files(object_id)
    
    # Формируем один атрибут (ID=5001), который содержит все файлы
    attributes_list = []
    if file_infos:
        attributes_list.append(AttributeValue(
            attributeId=5001,
            attributeFieldType="ftFile",
            isMultiple=True,
            fileInfoCollection=file_infos
        ))
    # Если файлов нет, всё равно возвращаем пустой attributes (по спецификации можно)
    return ObjectFilesResponse(
        objectVersionId=obj.get("objectVersionId", 0),
        objectType=obj.get("objectTypeId", 0),
        readOnly=obj.get("readOnly", False),
        attributes=attributes_list
    )

@app.get("/core/api/files/objects/{obj_id}/files/{blob_id}")
async def download_file(obj_id: int, blob_id: int, isNeedUnpack: bool = True, auth: str = Depends(verify_token)):
    """Отдаёт реальный файл, соответствующий blobId (вычисленному из имени)"""
    # Находим файл в папке объекта, у которого blobId совпадает
    file_infos = scan_object_files(obj_id)
    target_file = None
    for fi in file_infos:
        if fi.blobId == blob_id:
            target_file = fi
            break
    if not target_file:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(MOCK_FILES_ROOT, str(obj_id), target_file.fileName)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File missing on server")
    
    # Определяем MIME-тип по расширению (можно расширить)
    media_type = "application/octet-stream"
    if target_file.fileName.endswith(".pdf"):
        media_type = "application/pdf"
    elif target_file.fileName.endswith(".jpg") or target_file.fileName.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif target_file.fileName.endswith(".png"):
        media_type = "image/png"
    elif target_file.fileName.endswith(".txt"):
        media_type = "text/plain"
    
    return FileResponse(file_path, media_type=media_type, headers={"api-version": "1.0"})

@app.post("/core/api/objects/{obj_id}/attributes")
async def update_attributes(obj_id: int, updates: List[UpdateAttributeRequest], auth: str = Depends(verify_token)):
    # Выводим в консоль сервера всё, что пришло
    print(f"\n=== ПОЛУЧЕНЫ АТРИБУТЫ для объекта {obj_id} ===")
    for upd in updates:
        print(f"  Атрибут {upd.attributeID}:")
        for val in upd.values:
            print(f"    {val}")
    print("=" * 50)

    if obj_id not in objects_store:
        raise HTTPException(status_code=404, detail="Object not found")
    for upd in updates:
        key = (obj_id, upd.attributeID)
        custom_attributes[key] = upd.values
    return {"status": "ok", "message": f"Updated {len(updates)} attributes for object {obj_id}"}

@app.get("/debug/attributes/{obj_id}")
async def debug_attributes(obj_id: int):
    result = {}
    for (oid, aid), vals in custom_attributes.items():
        if oid == obj_id:
            result[aid] = vals
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)