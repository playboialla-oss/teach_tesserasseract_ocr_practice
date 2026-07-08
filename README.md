# OCR на PDF-чертежах

Практическая работа по OCR на инженерных PDF-чертежах.

Цель проекта — найти на PDF-чертежах блок технических требований / текстовых примечаний, распознать текст и сохранить координаты найденной области.

## Основной скрипт

Текущая экспериментальная версия:

`experiments/check_pdf_v5_post.py`

Подробное описание экспериментов:

`experiments/README_ocr_experiments.md`

## Что делает программа

Программа выполняет пайплайн:

PDF → изображение → OCR → текстовые кандидаты → оценка кандидатов → выбранный блок → JSON/debug

На выходе создаются:

- `output.json` с координатами и распознанным текстом;
- `debug_selected/` с выбранными блоками;
- `debug_all_candidates/` со всеми кандидатами;
- `debug_rejected/` с отклонёнными областями.

## Установка на Ubuntu

Системные зависимости:

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-rus poppler-utils python3-venv
sudo apt install -y libgl1 libglib2.0-0
