#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ITEM_RE = re.compile(r"^\s*(\d{1,2})[\).]?\s+(.+)$")
DASH_RE = re.compile(r"^\s*[-–—]\s+(.+)$")

STRONG_WORDS = [
    "ГОСТ", "ОСТ", "ТУ", "РД",
    "МАТЕРИАЛ", "МАТЕРИАЛ-ЗАМЕНИТЕЛЬ",
    "МАРКИРОВАТЬ", "МАРКИРОВКА",
    "ПОКРЫТИЕ", "ПОКРЫТЬ",
    "ДОПУСК", "ДОПУСКИ", "ОБЩИЕ ДОПУСКИ",
    "НЕУКАЗАННЫЕ", "ОТКЛОНЕНИЯ",
    "РАЗМЕР", "РАЗМЕРЫ", "СПРАВОК",
    "РАДИУС", "РАДИУСЫ",
    "ОБЕСПЕЧ", "ИНСТР",
    "КОНТРОЛЬ", "СВАРК", "ИСПЫТАН",
    "БИРКЕ", "КТ",
]

SIGNATURE_WORDS = [
    "ДОКУМЕНТ ПОДПИСАН",
    "СЕРТИФИКАТ",
    "ВЛАДЕЛЕЦ ПОДПИСИ",
    "ДЕЙСТВИТЕЛЬНО С",
    "ЭЦП",
]

TITLE_WORDS = [
    "ИЗМ.", "ЛИСТ", "№ ДОКУМ", "ПОДП.", "ДАТА",
    "РАЗРАБ.", "ПРОВ.", "Т.КОНТР.", "Н.КОНТР.",
    "УТВ.", "МАССА", "МАСШТАБ", "ФОРМАТ",
    "КОПИРОВАЛ", "CRC", "ID ВЕРСИИ",
]

DIM_WORDS = [
    "ОТВ.", "ОТВ", "ФАСК", "ФАСКИ",
    "РАДИУС", "РАДИУСА", "РАДИУСОВ",
]


def norm(text):
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def upper(text):
    return norm(text).upper().replace("Ё", "Е")


def bbox_union(items):
    return [
        min(i["bbox"][0] for i in items),
        min(i["bbox"][1] for i in items),
        max(i["bbox"][2] for i in items),
        max(i["bbox"][3] for i in items),
    ]


def rect_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

    return inter / (area_a + area_b - inter)


def overlap_x(a, b):
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    return inter / max(1.0, min(ax2 - ax1, bx2 - bx1))


def has_strong_word(text):
    u = upper(text)
    return any(w in u for w in STRONG_WORDS)


def is_signature(text):
    u = upper(text)
    return any(w in u for w in SIGNATURE_WORDS)


def is_title_block(text):
    u = upper(text)
    return any(w in u for w in TITLE_WORDS)




def is_short_dimension_item(text):
    t = norm(text)
    u = upper(t)

    m = ITEM_RE.match(t)
    if not m:
        return False

    rest = upper(m.group(2)).strip()
    rest_len = len(rest)

    # Важно: это нормальные технические требования, а не размерные подписи.
    # Пример: "1 Радиусы внутренних скруглений не более 0,4 мм."
    if re.search(r"ВНУТРЕН|СКРУГЛЕН|НЕ\s+БОЛЕЕ|НЕУКАЗАН|ДОПУСК|МАРКИРОВАТЬ|МАТЕРИАЛ|ПОКРЫТИЕ|ГОСТ|ОСТ|ТУ", rest):
        return False

    # Важно: слово "ОТВЕРСТИЕ" не должно попадать под "ОТВ.".
    if re.match(r"^ОТВЕРСТИ[ЕЯ]", rest):
        return False

    # Точно размерные/геометрические подписи:
    # "2 радиуса", "2 фаски", "4 отв. Ø18", "2 отв. M4-6H".
    if re.match(r"^(ОТВ\.?|ФАСК[А-Я]*|РАДИУС[А-Я]*|R[0-9]|RA\s|Ø|Ç|M[0-9])(\s|\.|$)", rest):
        if rest_len <= 65:
            return True

    if re.match(r"^[0-9]+\s*(ОТВ\.?|ФАСК[А-Я]*|РАДИУС[А-Я]*)", rest):
        if rest_len <= 65:
            return True

    return False


def item_number(text):
    m = ITEM_RE.match(norm(text))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def classify_line(text):
    t = norm(text)

    if not t:
        return "empty"

    if is_signature(t):
        return "signature"

    if is_title_block(t):
        return "title"

    if is_short_dimension_item(t):
        return "bad_dim"

    if ITEM_RE.match(t):
        # Настоящий пункт: номер + длинный текст или ключевые слова.
        m = ITEM_RE.match(t)
        rest = m.group(2)
        if has_strong_word(t) or len(rest) >= 18:
            return "item"

    if DASH_RE.match(t):
        return "dash"

    return "continuation"



def trim_words_to_embedded_requirement(ws):
    """
    Если PyMuPDF собрал строку так:
    "R3 1 Радиусы внутренних скруглений..."
    то возвращаем слова начиная с "1".

    Это нужно для КОМПАС-PDF, где текст ТТ иногда попадает
    в одну строку с размерной подписью.
    """
    if not ws:
        return ws

    for idx in range(len(ws)):
        token = norm(str(ws[idx][4]))
        token_clean = token.strip().replace(".", "").replace(")", "")

        if not re.match(r"^\d{1,2}\*?$", token_clean):
            continue

        rest_text = norm(" ".join(str(w[4]) for w in ws[idx:]))
        rest_upper = upper(rest_text)

        # Не режем на размерных строках типа "2 радиуса", "4 отв."
        if is_short_dimension_item(rest_text):
            continue

        # Настоящий пункт ТТ: после номера есть сильные слова
        # или строка достаточно длинная.
        if has_strong_word(rest_text) or len(rest_text) >= 28:
            return ws[idx:]

    return ws



def extract_lines(page):
    words = page.get_text("words", sort=True)

    groups = {}

    for w in words:
        x0, y0, x1, y1, text, block_no, line_no, word_no = w[:8]
        text = norm(text)

        if not text:
            continue

        key = (block_no, line_no)
        groups.setdefault(key, []).append(w)

    lines = []

    for key, ws in groups.items():
        ws = sorted(ws, key=lambda z: z[0])

        # fix2: если настоящая строка ТТ начинается внутри строки,
        # обрезаем всё, что стоит перед номером пункта.
        ws = trim_words_to_embedded_requirement(ws)

        text = norm(" ".join(str(w[4]) for w in ws))

        if not text:
            continue

        bbox = [
            min(w[0] for w in ws),
            min(w[1] for w in ws),
            max(w[2] for w in ws),
            max(w[3] for w in ws),
        ]

        line = {
            "text": text,
            "bbox": bbox,
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "width": bbox[2] - bbox[0],
            "height": bbox[3] - bbox[1],
            "kind": classify_line(text),
            "item_number": item_number(text),
        }

        lines.append(line)

    lines.sort(key=lambda l: (l["y1"], l["x1"]))
    return lines


def median_line_height(lines):
    hs = sorted(l["height"] for l in lines if l["height"] > 0)
    if not hs:
        return 8.0
    return hs[len(hs) // 2]


def same_requirement_zone(line, block_bbox, base_x, page_w):
    b = line["bbox"]

    # Строка примерно в той же колонке.
    if abs(line["x1"] - base_x) <= 35:
        return True

    # Строка продолжения может быть чуть правее начала пункта.
    if base_x <= line["x1"] <= base_x + page_w * 0.22:
        return True

    # Горизонтальное пересечение с уже собранным блоком.
    if overlap_x(b, block_bbox) >= 0.25:
        return True

    return False


def build_candidate_from_anchor(lines, anchor_index, page_w, page_h):
    anchor = lines[anchor_index]

    if anchor["kind"] != "item":
        return None

    base_x = anchor["x1"]
    collected = [anchor]
    block_bbox = anchor["bbox"][:]
    med_h = median_line_height(lines)

    last_y2 = anchor["y2"]

    for j in range(anchor_index + 1, len(lines)):
        line = lines[j]

        if line["y1"] < anchor["y1"] - 1:
            continue

        gap = line["y1"] - last_y2

        # Если ушли далеко вниз — блок закончился.
        if gap > max(18.0, med_h * 3.2):
            # Но если это следующий настоящий пункт в той же зоне, можно продолжить.
            if line["kind"] != "item" or not same_requirement_zone(line, block_bbox, base_x, page_w):
                break

        if line["kind"] in ("signature", "title"):
            # ЭЦП и штамп ниже ТТ.
            if len(collected) >= 1:
                break
            continue

        if line["kind"] == "bad_dim":
            # "2 радиуса" и подобное отдельно не берём.
            continue

        if line["kind"] == "item":
            if not same_requirement_zone(line, block_bbox, base_x, page_w):
                continue

            collected.append(line)
            block_bbox = bbox_union(collected)
            last_y2 = max(last_y2, line["y2"])
            continue

        if line["kind"] in ("dash", "continuation"):
            if same_requirement_zone(line, block_bbox, base_x, page_w):
                # Берём подпункты и продолжения:
                # "- фосфора...", "бирке.", длинные строки без номера.
                if line["kind"] == "dash" or len(line["text"]) >= 5:
                    collected.append(line)
                    block_bbox = bbox_union(collected)
                    last_y2 = max(last_y2, line["y2"])
            continue

    if not collected:
        return None

    text = "\n".join(l["text"] for l in collected)
    nums = [l["item_number"] for l in collected if l.get("item_number") is not None]

    return {
        "lines": collected,
        "bbox": bbox_union(collected),
        "text": text,
        "item_numbers": nums,
        "line_count": len(collected),
    }


def score_candidate(c, page_w, page_h):
    text = c["text"]
    u = upper(text)
    bbox = c["bbox"]

    item_count = len(c["item_numbers"])
    dash_count = sum(1 for l in c["lines"] if l["kind"] == "dash")
    cont_count = sum(1 for l in c["lines"] if l["kind"] == "continuation")
    strong_count = sum(1 for w in STRONG_WORDS if w in u)

    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    area_rel = (bw * bh) / max(1.0, page_w * page_h)

    score = 0.0
    reasons = []

    if item_count >= 1:
        score += 30
        reasons.append("+items")

    if item_count >= 2:
        score += 35
        reasons.append("+multi_items")

    if dash_count > 0:
        score += 12
        reasons.append("+dash_subitems")

    if cont_count > 0:
        score += 8
        reasons.append("+continuations")

    if strong_count:
        score += min(50, strong_count * 8)
        reasons.append("+keywords")

    # Последовательность пунктов 1,2,3...
    nums = c["item_numbers"]
    if nums:
        uniq = []
        for n in nums:
            if n not in uniq:
                uniq.append(n)

        if len(uniq) >= 2:
            seq_bonus = 0
            for a, b in zip(uniq, uniq[1:]):
                if b == a + 1:
                    seq_bonus += 8
            if seq_bonus:
                score += seq_bonus
                reasons.append("+sequence")

        if 1 in uniq:
            score += 8
            reasons.append("+has_1")

    # Блок ТТ обычно не должен быть гигантским.
    if area_rel > 0.20:
        score -= 60
        reasons.append("-huge")

    if bw > page_w * 0.75:
        score -= 45
        reasons.append("-too_wide")

    if bh > page_h * 0.45:
        score -= 35
        reasons.append("-too_tall")

    # Слишком короткие одиночные пункты подозрительны.
    if item_count == 1 and len(text) < 45 and dash_count == 0:
        score -= 45
        reasons.append("-short_single")

    # Блоки ЭЦП/штампа не нужны.
    if any(is_signature(l["text"]) or is_title_block(l["text"]) for l in c["lines"]):
        score -= 100
        reasons.append("-signature_or_title")

    c["score"] = round(score, 2)
    c["reasons"] = reasons
    c["features"] = {
        "item_count": item_count,
        "dash_count": dash_count,
        "continuation_count": cont_count,
        "strong_word_count": strong_count,
        "area_rel": round(area_rel, 5),
    }

    return c


def make_candidates(lines, page_w, page_h):
    candidates = []

    for i, line in enumerate(lines):
        if line["kind"] != "item":
            continue

        c = build_candidate_from_anchor(lines, i, page_w, page_h)

        if not c:
            continue

        c = score_candidate(c, page_w, page_h)

        if c["score"] < 15:
            continue

        candidates.append(c)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Убираем дубли: кандидат со 2-го пункта часто лежит внутри кандидата с 1-го.
    unique = []

    for c in candidates:
        duplicate = False

        for old in unique:
            if rect_iou(c["bbox"], old["bbox"]) > 0.55:
                duplicate = True
                break

            # Если bbox почти внутри старого.
            cx1, cy1, cx2, cy2 = c["bbox"]
            ox1, oy1, ox2, oy2 = old["bbox"]

            inside = (
                cx1 >= ox1 - 5 and cy1 >= oy1 - 5 and
                cx2 <= ox2 + 5 and cy2 <= oy2 + 5
            )

            if inside:
                duplicate = True
                break

        if not duplicate:
            unique.append(c)

    return unique


def select_blocks(candidates, max_blocks):
    selected = []

    for c in candidates:
        if c["score"] < 35:
            continue

        bb = c["bbox"]
        overlap = False

        for old in selected:
            if rect_iou(bb, old["bbox"]) > 0.25:
                overlap = True
                break

        if overlap:
            continue

        selected.append(c)

        if len(selected) >= max_blocks:
            break

    return selected


def render_page(page, dpi):
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom


def scale_bbox(b, zoom):
    return [int(v * zoom) for v in b]


def draw_debug(page, candidates, selected, out_selected, out_all, dpi):
    img, zoom = render_page(page, dpi)

    # all candidates
    img_all = img.copy()
    draw_all = ImageDraw.Draw(img_all)

    for idx, c in enumerate(candidates, start=1):
        b = scale_bbox(c["bbox"], zoom)
        draw_all.rectangle(b, outline=(180, 180, 180), width=2)
        draw_all.text((b[0], max(0, b[1] - 14)), f"{idx}:{c['score']}", fill=(80, 80, 80))

    for idx, c in enumerate(selected, start=1):
        b = scale_bbox(c["bbox"], zoom)
        draw_all.rectangle(b, outline=(0, 0, 255), width=4)
        draw_all.text((b[0], max(0, b[1] - 18)), f"SEL {idx}:{c['score']}", fill=(0, 0, 255))

    out_all.parent.mkdir(parents=True, exist_ok=True)
    img_all.save(out_all)

    # selected only
    img_sel = img.copy()
    draw_sel = ImageDraw.Draw(img_sel)

    for idx, c in enumerate(selected, start=1):
        b = scale_bbox(c["bbox"], zoom)
        draw_sel.rectangle(b, outline=(0, 0, 255), width=5)
        draw_sel.text((b[0], max(0, b[1] - 18)), f"TT {idx} score={c['score']}", fill=(0, 0, 255))

        # строки внутри блока — красным
        for line in c["lines"]:
            lb = scale_bbox(line["bbox"], zoom)
            draw_sel.rectangle(lb, outline=(255, 0, 0), width=2)

    out_selected.parent.mkdir(parents=True, exist_ok=True)
    img_sel.save(out_selected)



def line_in_x_zone(line, base_x, page_w):
    # Строки пункта и его продолжений обычно начинаются рядом.
    if abs(line["x1"] - base_x) <= 45:
        return True

    # Продолжения и подпункты могут быть немного правее.
    if base_x <= line["x1"] <= base_x + page_w * 0.30:
        return True

    return False


def add_cluster_candidates(candidates, lines, page_w, page_h):
    """
    v2/fix:
    Строит кандидаты не только от одного якоря, а от всей зоны ТТ.
    Это нужно для случаев:
    - пункт 1 имеет подпункты через дефис;
    - блок дробится на 1-3 и 5-7;
    - продолжение строки не начинается с номера.
    """
    item_lines = [
        l for l in lines
        if l["kind"] == "item"
        and l.get("item_number") is not None
        and 1 <= int(l["item_number"]) <= 30
        and not is_short_dimension_item(l["text"])
    ]

    if not item_lines:
        return candidates

    # Группируем пункты по близкому X — фактически по колонке ТТ.
    clusters = []

    for line in sorted(item_lines, key=lambda l: (l["x1"], l["y1"])):
        placed = False

        for cl in clusters:
            base_x = cl["base_x"]

            if abs(line["x1"] - base_x) <= max(55, page_w * 0.08):
                cl["items"].append(line)
                xs = [x["x1"] for x in cl["items"]]
                cl["base_x"] = sum(xs) / len(xs)
                placed = True
                break

        if not placed:
            clusters.append({
                "base_x": line["x1"],
                "items": [line],
            })

    new_candidates = []

    med_h = median_line_height(lines)

    for cl in clusters:
        items = sorted(cl["items"], key=lambda l: (l["y1"], l["x1"]))

        # Отбрасываем маленькие изолированные размерные группы.
        if len(items) == 1:
            txt = items[0]["text"]
            if len(txt) < 50 and not has_strong_word(txt):
                continue

        base_x = cl["base_x"]
        y_top = min(l["y1"] for l in items)
        y_bottom = max(l["y2"] for l in items)

        # Расширяем вниз/вверх, чтобы поймать продолжения и подпункты.
        y_min = y_top - med_h * 1.5
        y_max = y_bottom + med_h * 3.5

        collected = []

        for line in lines:
            if line["y2"] < y_min or line["y1"] > y_max:
                continue

            if line["kind"] in ("signature", "title", "bad_dim"):
                continue

            if not line_in_x_zone(line, base_x, page_w):
                continue

            # Берём пункты, подпункты и продолжения внутри зоны.
            if line["kind"] in ("item", "dash", "continuation"):
                # Не берём совсем короткие изолированные слова.
                if line["kind"] == "continuation" and len(line["text"]) < 4:
                    continue

                collected.append(line)

        if not collected:
            continue

        collected = sorted(collected, key=lambda l: (l["y1"], l["x1"]))

        # Убираем мусор после большого разрыва: если после блока внезапно начинается штамп.
        cleaned = []
        last_y2 = None

        for line in collected:
            if last_y2 is not None:
                gap = line["y1"] - last_y2

                if gap > max(22, med_h * 4.0):
                    # Если это следующий номерной пункт в той же зоне — оставляем.
                    # Иначе считаем, что блок закончился.
                    if line["kind"] != "item":
                        break

            cleaned.append(line)
            last_y2 = max(last_y2 or line["y2"], line["y2"])

        if not cleaned:
            continue

        nums = [l["item_number"] for l in cleaned if l.get("item_number") is not None]
        text = "\n".join(l["text"] for l in cleaned)

        c = {
            "lines": cleaned,
            "bbox": bbox_union(cleaned),
            "text": text,
            "item_numbers": nums,
            "line_count": len(cleaned),
        }

        c = score_candidate(c, page_w, page_h)

        # Бонус за длинный цельный блок с несколькими пунктами.
        uniq_nums = []
        for n in nums:
            if n not in uniq_nums:
                uniq_nums.append(n)

        if len(uniq_nums) >= 3:
            c["score"] += 30
            c["reasons"].append("+cluster_multi_items")

        if 1 in uniq_nums and len(uniq_nums) >= 2:
            c["score"] += 15
            c["reasons"].append("+cluster_starts_with_1")

        if any(l["kind"] == "dash" for l in cleaned):
            c["score"] += 15
            c["reasons"].append("+cluster_has_subitems")

        c["score"] = round(float(c["score"]), 2)
        new_candidates.append(c)

    merged = list(candidates) + new_candidates
    merged.sort(key=lambda x: x["score"], reverse=True)

    unique = []

    for c in merged:
        duplicate = False

        for old in unique:
            if rect_iou(c["bbox"], old["bbox"]) > 0.55:
                duplicate = True
                break

            # Если один bbox почти внутри другого, оставляем тот, у кого score выше.
            cx1, cy1, cx2, cy2 = c["bbox"]
            ox1, oy1, ox2, oy2 = old["bbox"]

            inside = (
                cx1 >= ox1 - 5 and
                cy1 >= oy1 - 5 and
                cx2 <= ox2 + 5 and
                cy2 <= oy2 + 5
            )

            if inside:
                duplicate = True
                break

        if not duplicate:
            unique.append(c)

    return unique


def process_pdf(pdf, out_dir, dpi, max_blocks):
    doc = fitz.open(pdf)
    page = doc[0]
    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    lines = extract_lines(page)
    candidates = make_candidates(lines, page_w, page_h)
    candidates = add_cluster_candidates(candidates, lines, page_w, page_h)
    selected = select_blocks(candidates, max_blocks)

    safe_name = pdf.name.replace("/", "_").replace("\\", "_")
    debug_selected = out_dir / "debug_selected" / f"{safe_name}.png"
    debug_all = out_dir / "debug_all_candidates" / f"{safe_name}.png"

    draw_debug(page, candidates, selected, debug_selected, debug_all, dpi)

    result = {
        "file": str(pdf),
        "page": 1,
        "page_size_pdf_points": {
            "width": page_w,
            "height": page_h,
        },
        "line_count": len(lines),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected": [],
        "candidates": [],
        "debug_selected": str(debug_selected),
        "debug_all_candidates": str(debug_all),
    }

    for c in selected:
        result["selected"].append({
            "bbox_pdf_points": {
                "x1": c["bbox"][0],
                "y1": c["bbox"][1],
                "x2": c["bbox"][2],
                "y2": c["bbox"][3],
                "width": c["bbox"][2] - c["bbox"][0],
                "height": c["bbox"][3] - c["bbox"][1],
            },
            "score": c["score"],
            "reasons": c["reasons"],
            "features": c["features"],
            "item_numbers": c["item_numbers"],
            "text": c["text"],
            "lines": [
                {
                    "text": l["text"],
                    "kind": l["kind"],
                    "bbox": l["bbox"],
                    "item_number": l["item_number"],
                }
                for l in c["lines"]
            ],
        })

    for c in candidates[:20]:
        result["candidates"].append({
            "bbox_pdf_points": c["bbox"],
            "score": c["score"],
            "reasons": c["reasons"],
            "features": c["features"],
            "item_numbers": c["item_numbers"],
            "text": c["text"],
        })

    doc.close()
    return result


def find_pdfs(path):
    path = Path(path)

    if path.is_file():
        return [path]

    pdfs = []

    for p in path.rglob("*"):
        if p.suffix.lower() == ".pdf":
            pdfs.append(p)

    return sorted(pdfs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="PDF-файл или папка с PDF")
    parser.add_argument("--out", default="results/check_pdf_kompas_textlayer_fix2")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-blocks", type=int, default=2)

    args = parser.parse_args()

    inp = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(inp)

    print(f"Найдено PDF: {len(pdfs)}")
    print(f"Выходная папка: {out_dir}")

    results = []

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf}")

        try:
            res = process_pdf(pdf, out_dir, args.dpi, args.max_blocks)
            results.append(res)
            print(f"  selected: {res['selected_count']} / candidates: {res['candidate_count']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "file": str(pdf),
                "error": str(e),
            })

    out_json = out_dir / "output.json"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("Готово:")
    print(out_json)
    print(out_dir / "debug_selected")
    print(out_dir / "debug_all_candidates")


if __name__ == "__main__":
    raise SystemExit(main())
