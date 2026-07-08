#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from pytesseract import Output


GOOD_KEYWORDS = [
    "ГОСТ", "ОСТ", "ТУ", "РД",
    "РАЗМЕР", "РАЗМЕРЫ",
    "МАТЕРИАЛ",
    "МАРКИРОВАТЬ", "МАРКИРОВКА",
    "ПОКРЫТИЕ", "ПОКРЫТЬ",
    "ДОПУСКАЕТСЯ", "ДОПУСКИ", "ОБЩИЕ ДОПУСКИ",
    "СВАРКА", "СВАРНОЙ", "КОНТРОЛЬ",
    "ТРЕБОВАНИЯ", "ТЕХНИЧЕСКИЕ", "ПРИМЕЧАНИЯ",
    "НЕУКАЗАННЫЕ", "ОСТАЛЬНЫЕ", "ПРЕДЕЛЬНЫЕ",
    "ОТКЛОНЕНИЯ", "ОБЕСПЕЧ", "ИСПЫТАНИЯ",
    "ПРАВИЛ", "ПРАВИЛА",
]

BAD_SIGNATURE_WORDS = [
    "ДОКУМЕНТ ПОДПИСАН",
    "СЕРТИФИКАТ",
    "ВЛАДЕЛЕЦ ПОДПИСИ",
    "ДЕЙСТВИТЕЛЬНО",
    "ЭЦП",
    "ПОДПИСАН",
]

BAD_STAMP_WORDS = [
    "ЛИТ", "МАССА", "МАСШТАБ", "ЛИСТ", "ЛИСТОВ",
    "ИЗМ", "ПОДП", "ДАТА", "РАЗРАБ", "ПРОВ",
    "Т.КОНТР", "Н.КОНТР", "УТВ", "ФОРМАТ",
    "ОБОЗНАЧЕНИЕ", "НАИМЕНОВАНИЕ", "КОПИРОВАЛ",
]

DIM_PATTERNS = [
    r"\bRa\s*\d",
    r"\bR\s*\d",
    r"[Ø⌀Ф]\s*\d",
    r"\bM\d+",
    r"\d+\s*°",
    r"\d+\s*[xх]\s*\d+",
    r"\b[А-ЯA-Z]-[А-ЯA-Z]\b",
    r"\b\d+[,.]?\d*\s*[±]\s*\d",
    r"\b\d+[Hh][0-9]\b",
]

ITEM_LINE_RE = re.compile(r"^\s*(\d{1,2})\s*([\).*:-]|\s)\s*\S+")
ITEM_ANY_RE = re.compile(r"(^|\n)\s*\d{1,2}\s*([\).*:-]|\s)\s*")


def norm(s):
    s = str(s).replace("ё", "е").replace("Ё", "Е")
    return re.sub(r"\s+", " ", s.strip())


def safe_name(path):
    s = Path(path).name
    for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        s = s.replace(ch, "_")
    return s


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    aa = max(1, (ax2 - ax1) * (ay2 - ay1))
    bb = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / (aa + bb - inter)

def render_first_page(pdf_path, dpi):
    pages = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=1,
        last_page=1,
        fmt="png",
        thread_count=1,
    )

    if not pages:
        raise RuntimeError(f"Не удалось отрендерить PDF: {pdf_path}")

    rgb = np.array(pages[0].convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=7)

    bin_img = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        15,
    )

    return bin_img


def list_pdfs(input_path):
    p = Path(input_path).expanduser()

    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]

    if p.is_dir():
        return sorted(p.rglob("*.pdf"))

    return []


def dim_count(text):
    count = 0

    for pat in DIM_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            count += 1

    return count


def keyword_hits(text):
    u = text.upper()
    return sorted(set(kw for kw in GOOD_KEYWORDS if kw in u))


def is_signature_line(text):
    u = text.upper()
    return any(w in u for w in BAD_SIGNATURE_WORDS)


def word_rows(img, lang, psms, page_w):
    rows = []

    for psm in psms:
        config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"

        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=config,
            output_type=Output.DICT,
        )

        n = len(data.get("text", []))

        for i in range(n):
            text = norm(data["text"][i])

            if not text:
                continue

            try:
                conf = float(data["conf"][i])
            except Exception:
                conf = -1.0

            if conf < 10:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            if w < 2 or h < 4:
                continue

            # Важное отличие v4.1:
            # убираем слова из самой левой боковой рамки до группировки.
            if x + w < page_w * 0.08:
                continue

            rows.append({
                "text": text,
                "bbox": (x, y, x + w, y + h),
                "conf": conf,
                "psm": psm,
                "key": (
                    psm,
                    int(data["block_num"][i]),
                    int(data["par_num"][i]),
                    int(data["line_num"][i]),
                ),
            })

    return rows

def split_words_to_lines(words, page_w):
    by_key = {}

    for w in words:
        by_key.setdefault(w["key"], []).append(w)

    lines = []

    for key, ws in by_key.items():
        ws = sorted(ws, key=lambda z: z["bbox"][0])

        heights = [z["bbox"][3] - z["bbox"][1] for z in ws]
        med_h = np.median(heights) if heights else 12

        groups = []
        cur = []
        prev = None

        for w in ws:
            if prev is None:
                cur = [w]
            else:
                gap = w["bbox"][0] - prev["bbox"][2]

                # Важное отличие v4.1:
                # если Tesseract склеил далёкие куски одной строкой,
                # режем строку по большому горизонтальному промежутку.
                if gap > max(45, 3.2 * med_h, page_w * 0.035):
                    groups.append(cur)
                    cur = [w]
                else:
                    cur.append(w)

            prev = w

        if cur:
            groups.append(cur)

        for gs in groups:
            text = norm(" ".join(g["text"] for g in gs))

            if len(text) < 2:
                continue

            x1 = min(g["bbox"][0] for g in gs)
            y1 = min(g["bbox"][1] for g in gs)
            x2 = max(g["bbox"][2] for g in gs)
            y2 = max(g["bbox"][3] for g in gs)

            lines.append({
                "text": text,
                "bbox": (x1, y1, x2, y2),
                "conf": float(np.mean([g["conf"] for g in gs])),
            })

    # Убираем дубли от PSM 11 и PSM 6.
    lines = sorted(lines, key=lambda z: z["conf"], reverse=True)
    kept = []

    for line in lines:
        duplicate = False

        for old in kept:
            if iou(line["bbox"], old["bbox"]) > 0.70:
                duplicate = True
                break

            if line["text"] == old["text"]:
                ly = (line["bbox"][1] + line["bbox"][3]) / 2
                oy = (old["bbox"][1] + old["bbox"][3]) / 2

                if abs(ly - oy) < 20:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(line)

    return sorted(kept, key=lambda z: (z["bbox"][1], z["bbox"][0]))


def is_noise_line(line, page_w, page_h):
    text = line["text"]
    u = text.upper()

    x1, y1, x2, y2 = line["bbox"]

    xc = (x1 + x2) / 2 / page_w
    yc = (y1 + y2) / 2 / page_h

    if is_signature_line(text):
        return True

    if yc > 0.90:
        return True

    if yc > 0.84 and any(w in u for w in BAD_STAMP_WORDS):
        return True

    if len(text) < 4 and not ITEM_LINE_RE.match(text):
        return True

    if dim_count(text) >= 2 and not keyword_hits(text):
        return True

    if xc < 0.08:
        return True

    return False


def union_bbox(group):
    return (
        min(l["bbox"][0] for l in group),
        min(l["bbox"][1] for l in group),
        max(l["bbox"][2] for l in group),
        max(l["bbox"][3] for l in group),
    )

def table_density(gray, bbox):
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(gray.shape[1], x2)
    y2 = min(gray.shape[0], y2)

    roi = gray[y1:y2, x1:x2]

    if roi.size == 0:
        return 0.0

    inv = cv2.threshold(roi, 210, 255, cv2.THRESH_BINARY_INV)[1]

    h, w = inv.shape[:2]

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, w // 8), 1),
    )

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(20, h // 8)),
    )

    horizontal = cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kernel)

    density = (
        cv2.countNonZero(horizontal) + cv2.countNonZero(vertical)
    ) / max(1, h * w)

    return float(density)


def make_candidates(lines, page_w, page_h):
    good = []

    for line in lines:
        if not is_noise_line(line, page_w, page_h):
            good.append(line)

    parent = list(range(len(good)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(good):
        ax1, ay1, ax2, ay2 = a["bbox"]
        ah = max(1, ay2 - ay1)
        aw = max(1, ax2 - ax1)

        for j in range(i + 1, len(good)):
            b = good[j]

            bx1, by1, bx2, by2 = b["bbox"]
            bh = max(1, by2 - by1)
            bw = max(1, bx2 - bx1)

            # v4.1: группируем только вертикально близкие строки.
            # Горизонтальное склеивание через весь лист убрано.
            vertical_gap = max(0, by1 - ay2, ay1 - by2)

            if vertical_gap > max(24, 2.2 * max(ah, bh)):
                continue

            x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            min_w = max(1, min(aw, bw))

            start_close = abs(ax1 - bx1) < page_w * 0.055
            centers_close = abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2) < page_w * 0.12

            if not (x_overlap / min_w > 0.10 or start_close or centers_close):
                continue

            ux1 = min(ax1, bx1)
            uy1 = min(ay1, by1)
            ux2 = max(ax2, bx2)
            uy2 = max(ay2, by2)

            # Не даём группе стать гигантской.
            if (ux2 - ux1) > page_w * 0.50:
                continue

            if (uy2 - uy1) > page_h * 0.28:
                continue

            union(i, j)

    comps = {}

    for i, line in enumerate(good):
        comps.setdefault(find(i), []).append(line)

    candidates = []

    for group in comps.values():
        group = sorted(group, key=lambda z: (z["bbox"][1], z["bbox"][0]))
        candidates.append(group)

    return candidates

def score_candidate(group, gray, page_w, page_h, cid):
    bbox = union_bbox(group)

    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    text = "\n".join(l["text"] for l in group)
    u = text.upper()

    line_count = len(group)
    char_count = len(text.replace("\n", " "))

    kws = keyword_hits(text)
    item_count = len(ITEM_ANY_RE.findall(text))
    dims = dim_count(text)

    stamp_count = sum(1 for w in BAD_STAMP_WORDS if w in u)
    signature_count = sum(1 for w in BAD_SIGNATURE_WORDS if w in u)

    table_score = table_density(gray, bbox)

    x_center = (x1 + x2) / 2 / page_w
    y_center = (y1 + y2) / 2 / page_h
    area_rel = (bw * bh) / max(1, page_w * page_h)

    avg_conf = float(np.mean([l["conf"] for l in group])) if group else 0.0

    score = 0.0
    reasons = []

    if kws:
        add = min(48, 8 * len(kws))
        score += add
        reasons.append("+keywords:" + ",".join(kws[:8]))

    if item_count >= 1:
        score += 8
        reasons.append(f"+items:{item_count}")

    if item_count >= 2:
        score += min(28, item_count * 6)
        reasons.append("+multi_items")

    if line_count >= 2:
        score += min(22, line_count * 2.5)
        reasons.append(f"+lines:{line_count}")

    if char_count >= 60:
        score += 12
        reasons.append("+text60")

    if char_count >= 120:
        score += 10
        reasons.append("+text120")

    if avg_conf >= 40:
        score += 3
        reasons.append("+conf")

    if 0.10 < x_center < 0.92 and y_center < 0.82:
        score += 5
        reasons.append("+pos")

    if dims:
        score -= min(32, dims * 7)
        reasons.append(f"-dim:{dims}")

    if stamp_count:
        score -= min(45, stamp_count * 8)
        reasons.append(f"-stamp:{stamp_count}")

    if signature_count:
        score -= min(60, signature_count * 15)
        reasons.append(f"-sig:{signature_count}")

    if table_score > 0.045:
        score -= min(40, table_score * 520)
        reasons.append(f"-table:{table_score:.3f}")

    if x_center < 0.10:
        score -= 30
        reasons.append("-left")

    if y_center > 0.84:
        score -= 30
        reasons.append("-bottom")

    if x_center > 0.55 and y_center > 0.73 and not (item_count >= 2 and len(kws) >= 1):
        score -= 30
        reasons.append("-br_zone")

    if area_rel > 0.10:
        score -= 55
        reasons.append("-big_strict")

    if bw > page_w * 0.50:
        score -= 45
        reasons.append("-wide_strict")

    if char_count < 18:
        score -= 25
        reasons.append("-short")

    return {
        "id": cid,
        "bbox_px": {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": bw,
            "height": bh,
        },
        "text": text,
        "score": round(float(score), 2),
        "decision": "rejected",
        "reasons": reasons,
        "features": {
            "line_count": line_count,
            "char_count": char_count,
            "keyword_hits": ", ".join(kws),
            "item_count": item_count,
            "dim_count": dims,
            "stamp_count": stamp_count,
            "signature_count": signature_count,
            "table_score": round(table_score, 5),
            "x_center_rel": round(x_center, 4),
            "y_center_rel": round(y_center, 4),
            "area_rel": round(area_rel, 5),
            "avg_conf": round(avg_conf, 2),
        },
    }


def draw_candidates(bgr, candidates, out_path, mode):
    img = bgr.copy()

    for c in candidates:
        if mode == "selected" and c["decision"] != "selected":
            continue

        if mode == "rejected" and c["decision"] != "rejected":
            continue

        b = c["bbox_px"]

        x1 = b["x1"]
        y1 = b["y1"]
        x2 = b["x2"]
        y2 = b["y2"]

        if mode == "rejected":
            color = (160, 160, 160)
            thickness = 2
        elif c["score"] >= 35:
            color = (0, 180, 0)
            thickness = 3
        elif c["score"] >= 10:
            color = (0, 180, 255)
            thickness = 3
        else:
            color = (160, 160, 160)
            thickness = 1

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        label = f"id={c['id']} score={c['score']}"

        cv2.putText(
            img,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def v5_requirement_strength(c, page_w, page_h):
    f = c.get("features", {})
    b = c.get("bbox_px", {})

    kw_text = str(f.get("keyword_hits", "") or "")
    kws = [x.strip().upper() for x in kw_text.split(",") if x.strip()]

    item_count = int(f.get("item_count", 0) or 0)
    dim_count = int(f.get("dim_count", 0) or 0)
    stamp_count = int(f.get("stamp_count", 0) or 0)
    signature_count = int(f.get("signature_count", 0) or 0)

    table_score = float(f.get("table_score", 0) or 0)
    area_rel = float(f.get("area_rel", 0) or 0)
    line_count = int(f.get("line_count", 0) or 0)
    char_count = int(f.get("char_count", 0) or 0)

    width_rel = float(b.get("width", 0) or 0) / max(1, page_w)
    height_rel = float(b.get("height", 0) or 0) / max(1, page_h)

    x_center = (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2 / max(1, page_w)
    y_center = (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2 / max(1, page_h)

    strong_words = {
        "МАТЕРИАЛ", "МАРКИРОВАТЬ", "МАРКИРОВКА",
        "ПОКРЫТИЕ", "ПОКРЫТЬ",
        "ДОПУСКАЕТСЯ", "ДОПУСКИ", "ОБЩИЕ ДОПУСКИ",
        "СВАРКА", "СВАРНОЙ", "КОНТРОЛЬ",
        "ТРЕБОВАНИЯ", "ТЕХНИЧЕСКИЕ", "ПРИМЕЧАНИЯ",
        "НЕУКАЗАННЫЕ", "ОТКЛОНЕНИЯ", "ИСПЫТАНИЯ",
    }

    gost_like = any(k in {"ГОСТ", "ОСТ", "ТУ", "РД"} for k in kws)
    strong_kw = any(k in strong_words for k in kws)
    has_kw = bool(kws)

    strength = 0.0
    reasons = []

    if item_count >= 1:
        strength += 18
        reasons.append("+v5_items")

    if item_count >= 2:
        strength += 28
        reasons.append("+v5_multi_items")

    if strong_kw:
        strength += 28
        reasons.append("+v5_strong_keywords")

    if gost_like and item_count >= 1:
        strength += 18
        reasons.append("+v5_gost_with_items")

    if line_count >= 3:
        strength += 12
        reasons.append("+v5_lines")

    if char_count >= 80:
        strength += 12
        reasons.append("+v5_text80")

    if char_count >= 160:
        strength += 8
        reasons.append("+v5_text160")

    # Компактность важнее огромного score.
    if area_rel < 0.08:
        strength += 10
        reasons.append("+v5_compact")
    elif area_rel > 0.14:
        strength -= 45
        reasons.append("-v5_big_area")

    if width_rel > 0.62:
        strength -= 35
        reasons.append("-v5_too_wide")

    if height_rel > 0.36:
        strength -= 30
        reasons.append("-v5_too_tall")

    # Верхние служебные блоки часто ложные.
    if y_center < 0.20 and not strong_kw:
        strength -= 35
        reasons.append("-v5_top_service")

    # Нижний правый штамп.
    if x_center > 0.55 and y_center > 0.73 and not (item_count >= 2 or strong_kw):
        strength -= 45
        reasons.append("-v5_title_block_zone")

    if signature_count > 0:
        strength -= 100
        reasons.append("-v5_signature")

    if stamp_count > 0 and not (item_count >= 2 and strong_kw):
        strength -= 45
        reasons.append("-v5_stamp")

    # Размеры и фаски без признаков требований.
    if dim_count >= 2 and not strong_kw:
        strength -= 45
        reasons.append("-v5_dimensions")

    # Таблица/сетка: не запрет, а штраф.
    if table_score > 0.09 and not (item_count >= 2 and strong_kw):
        strength -= 45
        reasons.append("-v5_table")
    elif table_score > 0.09:
        strength -= 8
        reasons.append("-v5_table_soft")

    # Блоки только с ГОСТ без пунктов часто являются выносками/размерами.
    if gost_like and not strong_kw and item_count == 0 and line_count <= 2:
        strength -= 35
        reasons.append("-v5_gost_callout")

    c["v5_strength"] = round(float(strength), 2)
    c.setdefault("reasons", []).extend(reasons)

    requirement_like = strength >= 25 and (
        item_count >= 1 or strong_kw or (gost_like and line_count >= 3)
    )

    return requirement_like, strength


def v5_reselect_candidates(candidates, page_w, page_h, max_blocks):
    for c in candidates:
        req_like, strength = v5_requirement_strength(c, page_w, page_h)
        c["v5_requirement_like"] = bool(req_like)
        c["score"] = round(float(c.get("score", 0)) + float(strength), 2)

    candidates.sort(
        key=lambda c: (
            1 if c.get("v5_requirement_like") else 0,
            c.get("v5_strength", -999),
            c.get("score", -999),
        ),
        reverse=True,
    )

    selected = []

    for c in candidates:
        if not c.get("v5_requirement_like"):
            continue

        if c.get("v5_strength", 0) < 25:
            continue

        b = c.get("bbox_px", {})

        bb = (
            b.get("x1", 0),
            b.get("y1", 0),
            b.get("x2", 0),
            b.get("y2", 0),
        )

        overlaps = False
        for old in selected:
            ob = old.get("bbox_px", {})
            old_bb = (
                ob.get("x1", 0),
                ob.get("y1", 0),
                ob.get("x2", 0),
                ob.get("y2", 0),
            )

            if iou(bb, old_bb) > 0.30:
                overlaps = True
                break

        if overlaps:
            continue

        selected.append(c)

        if len(selected) >= max_blocks:
            break

    for c in candidates:
        c["decision"] = "selected" if c in selected else "rejected"

    return selected


def v5_post_has_strong_keyword(c):
    f = c.get("features", {})
    kw_text = str(f.get("keyword_hits", "") or "").upper()

    strong_words = [
        "МАТЕРИАЛ",
        "МАРКИРОВАТЬ",
        "МАРКИРОВКА",
        "ПОКРЫТИЕ",
        "ПОКРЫТЬ",
        "ДОПУСКАЕТСЯ",
        "ДОПУСКИ",
        "ОБЩИЕ ДОПУСКИ",
        "СВАРКА",
        "СВАРНОЙ",
        "КОНТРОЛЬ",
        "ТРЕБОВАНИЯ",
        "ТЕХНИЧЕСКИЕ",
        "ПРИМЕЧАНИЯ",
        "НЕУКАЗАННЫЕ",
        "ОТКЛОНЕНИЯ",
        "ИСПЫТАНИЯ",
        "ОБЕСПЕЧ",
    ]

    return any(w in kw_text for w in strong_words)


def v5_post_is_pure_dimension_or_table(c):
    f = c.get("features", {})
    text = str(c.get("text", "") or "").upper()

    item_count = int(f.get("item_count", 0) or 0)
    dim_count = int(f.get("dim_count", 0) or 0)
    table_score = float(f.get("table_score", 0) or 0)
    char_count = int(f.get("char_count", 0) or 0)

    has_strong_kw = v5_post_has_strong_keyword(c)

    dimension_words = [
        "RA", "R ", "Ø", "⌀", "Ф", "ФАСК", "ФАСКИ",
        "M", "X45", "Х45", "A-A", "А-А",
    ]

    dimension_text_hits = sum(1 for w in dimension_words if w in text)

    if dim_count >= 1 and not has_strong_kw and item_count == 0:
        return True

    if dimension_text_hits >= 2 and not has_strong_kw:
        return True

    if table_score > 0.075 and not has_strong_kw and item_count == 0:
        return True

    if char_count < 35 and not has_strong_kw and item_count == 0:
        return True

    return False


def v5_post_candidate_ok(c, page_w, page_h, strong_exists):
    f = c.get("features", {})
    b = c.get("bbox_px", {})

    item_count = int(f.get("item_count", 0) or 0)
    signature_count = int(f.get("signature_count", 0) or 0)
    stamp_count = int(f.get("stamp_count", 0) or 0)
    table_score = float(f.get("table_score", 0) or 0)
    area_rel = float(f.get("area_rel", 0) or 0)

    width_rel = float(b.get("width", 0) or 0) / max(1, page_w)
    height_rel = float(b.get("height", 0) or 0) / max(1, page_h)

    x_center = (float(b.get("x1", 0)) + float(b.get("x2", 0))) / 2 / max(1, page_w)
    y_center = (float(b.get("y1", 0)) + float(b.get("y2", 0))) / 2 / max(1, page_h)

    has_strong_kw = v5_post_has_strong_keyword(c)

    if signature_count > 0:
        c.setdefault("reasons", []).append("-post_signature")
        return False

    if area_rel > 0.18:
        c.setdefault("reasons", []).append("-post_huge_area")
        return False

    if width_rel > 0.74:
        c.setdefault("reasons", []).append("-post_too_wide")
        return False

    if height_rel > 0.44 and not (has_strong_kw and item_count >= 2):
        c.setdefault("reasons", []).append("-post_too_tall")
        return False

    if v5_post_is_pure_dimension_or_table(c):
        c.setdefault("reasons", []).append("-post_dimension_or_table")
        return False

    if strong_exists and not has_strong_kw and item_count == 0:
        c.setdefault("reasons", []).append("-post_skip_weak_after_strong")
        return False

    if x_center > 0.55 and y_center > 0.76 and stamp_count > 0:
        c.setdefault("reasons", []).append("-post_title_block")
        return False

    if y_center < 0.18 and not has_strong_kw:
        c.setdefault("reasons", []).append("-post_top_service")
        return False

    if table_score > 0.10 and not (has_strong_kw and item_count >= 1):
        c.setdefault("reasons", []).append("-post_table_without_requirements")
        return False

    # Дополнительный блок должен быть похож на требования:
    # или есть сильные слова, или есть пункты.
    if not has_strong_kw and item_count == 0:
        c.setdefault("reasons", []).append("-post_not_requirement_like")
        return False

    return True


def v5_post_reselect_candidates(candidates, page_w, page_h, max_blocks):
    # Сначала используем старую v5-логику, чтобы пересчитать v5_strength.
    _ = v5_reselect_candidates(candidates, page_w, page_h, max_blocks)

    candidates.sort(
        key=lambda c: (
            c.get("v5_requirement_like", False),
            c.get("v5_strength", -999),
            c.get("score", -999),
        ),
        reverse=True,
    )

    strong_exists = False

    for c in candidates:
        if c.get("v5_requirement_like") and v5_post_has_strong_keyword(c):
            if c.get("v5_strength", 0) >= 25:
                strong_exists = True
                break

    selected = []

    for c in candidates:
        if not c.get("v5_requirement_like"):
            continue

        if c.get("v5_strength", 0) < 20:
            continue

        if not v5_post_candidate_ok(c, page_w, page_h, strong_exists):
            continue

        b = c.get("bbox_px", {})

        bb = (
            b.get("x1", 0),
            b.get("y1", 0),
            b.get("x2", 0),
            b.get("y2", 0),
        )

        overlaps = False

        for old in selected:
            ob = old.get("bbox_px", {})
            old_bb = (
                ob.get("x1", 0),
                ob.get("y1", 0),
                ob.get("x2", 0),
                ob.get("y2", 0),
            )

            if iou(bb, old_bb) > 0.30:
                overlaps = True
                break

        if overlaps:
            continue

        selected.append(c)

        if len(selected) >= max_blocks:
            break

    for c in candidates:
        c["decision"] = "selected" if c in selected else "rejected"

    return selected


def process_pdf(pdf_path, out_dir, dpi, max_blocks, lang, psms):
    bgr = render_first_page(pdf_path, dpi)

    page_h, page_w = bgr.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    ocr_img = preprocess_for_ocr(bgr)

    words = word_rows(ocr_img, lang, psms, page_w)
    lines = split_words_to_lines(words, page_w)
    groups = make_candidates(lines, page_w, page_h)

    candidates = []

    for i, group in enumerate(groups, start=1):
        candidates.append(score_candidate(group, gray, page_w, page_h, i))

    # v4.2_fix: пересчитываем score для финального выбора.
    # Главная идея: блок с пунктами + ГОСТ/Материал/Допуски важнее,
    # чем просто чистая область без ключевых слов.
    for c in candidates:
        f = c.get("features", {})
        b = c.get("bbox_px", {})

        kw_text = str(f.get("keyword_hits", "") or "").strip()
        has_kw = bool(kw_text)

        item_count = int(f.get("item_count", 0) or 0)
        dim_count = int(f.get("dim_count", 0) or 0)
        table_score = float(f.get("table_score", 0) or 0)
        area_rel = float(f.get("area_rel", 0) or 0)

        width_rel = float(b.get("width", 0) or 0) / max(1, page_w)
        height_rel = float(b.get("height", 0) or 0) / max(1, page_h)

        selection_score = float(c.get("score", 0) or 0)

        strong = has_kw and item_count >= 1
        very_strong = has_kw and item_count >= 2

        if very_strong:
            selection_score += 35
            c.setdefault("reasons", []).append("+fix_very_strong_items_keywords")
        elif strong:
            selection_score += 22
            c.setdefault("reasons", []).append("+fix_strong_item_keyword")

        if has_kw and item_count == 0:
            selection_score += 8
            c.setdefault("reasons", []).append("+fix_keywords_no_items")

        if not has_kw and item_count == 0:
            selection_score -= 40
            c.setdefault("reasons", []).append("-fix_no_keywords_no_items")

        if dim_count >= 2 and not has_kw:
            selection_score -= 40
            c.setdefault("reasons", []).append("-fix_dims_without_keywords")

        # Линии/таблица — это штраф, но не абсолютный запрет,
        # если блок похож на требования.
        if table_score > 0.045 and not very_strong:
            selection_score -= 25
            c.setdefault("reasons", []).append("-fix_table_soft")
        elif table_score > 0.045 and very_strong:
            selection_score -= 6
            c.setdefault("reasons", []).append("-fix_table_allowed_for_requirements")

        if area_rel > 0.16:
            selection_score -= 70
            c.setdefault("reasons", []).append("-fix_huge_area")
        elif area_rel > 0.11 and not very_strong:
            selection_score -= 35
            c.setdefault("reasons", []).append("-fix_big_area_not_strong")

        if width_rel > 0.70:
            selection_score -= 70
            c.setdefault("reasons", []).append("-fix_too_wide")
        elif width_rel > 0.55 and not very_strong:
            selection_score -= 35
            c.setdefault("reasons", []).append("-fix_wide_not_strong")

        if height_rel > 0.36 and not very_strong:
            selection_score -= 35
            c.setdefault("reasons", []).append("-fix_tall_not_strong")

        c["raw_score"] = c.get("score", 0)
        c["score"] = round(float(selection_score), 2)
        c["selection_score"] = c["score"]

    candidates.sort(key=lambda c: c["score"], reverse=True)

    for i, c in enumerate(candidates, start=1):
        c["id"] = i

    selected = v5_post_reselect_candidates(candidates, page_w, page_h, max_blocks)

    selected_ids = set(c["id"] for c in selected)

    for c in candidates:
        c["decision"] = "selected" if c["id"] in selected_ids else "rejected"

    base = safe_name(pdf_path)

    draw_candidates(
        bgr,
        candidates,
        Path(out_dir) / "debug_selected" / f"{base}.png",
        "selected",
    )

    draw_candidates(
        bgr,
        candidates,
        Path(out_dir) / "debug_all_candidates" / f"{base}.png",
        "all",
    )

    draw_candidates(
        bgr,
        candidates,
        Path(out_dir) / "debug_rejected" / f"{base}.png",
        "rejected",
    )

    return {
        "file": str(pdf_path),
        "page": 1,
        "dpi": dpi,
        "page_size_px": {
            "width": page_w,
            "height": page_h,
        },
        "ocr_word_count": len(words),
        "ocr_line_count": len(lines),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected": selected,
        "candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input")
    parser.add_argument("--out", default="results/check_pdf_v5_post")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-blocks", type=int, default=3)
    parser.add_argument("--lang", default="rus+eng")
    parser.add_argument("--psm", default="11")

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    psms = []

    for p in args.psm.split(","):
        p = p.strip()

        if p:
            psms.append(int(p))

    if not psms:
        psms = [11]

    pdfs = list_pdfs(args.input)

    if not pdfs:
        print(f"PDF не найдены: {args.input}", file=sys.stderr)
        return 2

    print(f"Найдено PDF: {len(pdfs)}")
    print(f"Выходная папка: {out_dir}")
    print(f"DPI: {args.dpi}")
    print(f"lang: {args.lang}")
    print(f"PSM: {psms}")
    print(f"max_blocks: {args.max_blocks}")

    results = {
        "note": (
            "Эксперимент v5_post: post-filter после v5 для удаления лишних размеров, фасок, таблиц и служебных блоков. Проверено на DWG-PDF инженерных чертежах. "
            "Для PDF из КОМПАС/ASCON нужно дополнительно проверить. "
            "v5_post сначала выбирает основной блок требований, затем оставляет дополнительные блоки только если они похожи на продолжение требований."
        ),
        "input": str(args.input),
        "out": str(out_dir),
        "dpi": args.dpi,
        "lang": args.lang,
        "psm": psms,
        "max_blocks": args.max_blocks,
        "files": [],
    }

    for i, pdf in enumerate(pdfs, start=1):
        print(f"[{i}/{len(pdfs)}] {pdf}")

        try:
            item = process_pdf(
                pdf_path=pdf,
                out_dir=out_dir,
                dpi=args.dpi,
                max_blocks=args.max_blocks,
                lang=args.lang,
                psms=psms,
            )

            results["files"].append(item)

            top_score = None
            if item["selected"]:
                top_score = item["selected"][0]["score"]

            print(
                f"  selected: {item['selected_count']} / "
                f"candidates: {item['candidate_count']} / "
                f"top_score: {top_score}"
            )

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results["files"].append({
                "file": str(pdf),
                "error": repr(e),
            })

    output_json = out_dir / "output.json"

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Готово: {output_json}")
    print(f"Debug selected: {out_dir / 'debug_selected'}")
    print(f"Debug all candidates: {out_dir / 'debug_all_candidates'}")
    print(f"Debug rejected: {out_dir / 'debug_rejected'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
