#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path


WEIRD_CHARS_RE = re.compile(r"[ÇÅàã•]")
CYR_RE = re.compile(r"[А-Яа-яЁё]")
LAT_RE = re.compile(r"[A-Za-z]")

RX_TITLE_LEAK = re.compile(
    r"\b(Изм\.|Лист|№\s*докум\.|Подп\.|Дата|Разраб\.|Пров\.|Масса|Масштаб)\b",
    re.IGNORECASE,
)

RX_MK = re.compile(r"(ГОСТ\s+\d[\d.\-]*-)(mK)\b")
RX_GROUP_NO_SPACE = re.compile(r"\bГр\.([IVX]+(?:-[А-Я])?)\b")
RX_DECIMAL_DOT_MEASURE = re.compile(r"\b(\d+)\.(\d+)\s*(мм|%)\b", re.IGNORECASE)
RX_SUSPICIOUS_EDE = re.compile(r"\bсборочной\s+еде\.?\b", re.IGNORECASE)

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9_.\-]+|[^\s]")


def norm(text):
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def is_allowed_latin_token(token):
    t = token.strip()

    allowed_patterns = [
        r"^Ra$",
        r"^R\d+([,.]\d+)?$",
        r"^M\d+.*$",
        r"^[A-ZА-Я]?\d+[A-ZА-Я]?\d*$",
        r"^[Hh]\d+.*$",
        r"^[IVX]+(-[А-Я])?$",
        r"^CRC$",
        r"^ID$",
    ]

    for pat in allowed_patterns:
        if re.match(pat, t):
            return True

    return False


def find_mixed_script_tokens(text):
    tokens = TOKEN_RE.findall(text)
    bad = []

    for token in tokens:
        if CYR_RE.search(token) and LAT_RE.search(token):
            if not is_allowed_latin_token(token):
                bad.append(token)

    return bad


def apply_safe_corrections(line):
    original = line
    corrected = line
    corrections = []

    # 1. Убрать хвост штампа, если он случайно приклеился к строке ТТ.
    m = RX_TITLE_LEAK.search(corrected)
    if m and m.start() > 20:
        before = corrected
        corrected = corrected[:m.start()].rstrip()
        corrections.append({
            "rule": "trim_title_block_leak",
            "before": before,
            "after": corrected,
            "confidence": 0.95,
        })

    # 2. ГОСТ ...-mK -> ГОСТ ...-мК
    before = corrected
    corrected = RX_MK.sub(r"\1мК", corrected)
    if corrected != before:
        corrections.append({
            "rule": "fix_latin_mK_to_cyrillic_мК",
            "before": before,
            "after": corrected,
            "confidence": 0.98,
        })

    # 3. Гр.I -> Гр. I
    before = corrected
    corrected = RX_GROUP_NO_SPACE.sub(r"Гр. \1", corrected)
    if corrected != before:
        corrections.append({
            "rule": "fix_group_spacing",
            "before": before,
            "after": corrected,
            "confidence": 0.97,
        })

    # 4. 0.4 мм -> 0,4 мм
    # Но ГОСТ 30893.2-2002 трогать нельзя, поэтому меняем только если рядом есть мм или %.
    before = corrected
    corrected = RX_DECIMAL_DOT_MEASURE.sub(r"\1,\2 \3", corrected)
    if corrected != before:
        corrections.append({
            "rule": "fix_decimal_dot_in_measurement",
            "before": before,
            "after": corrected,
            "confidence": 0.93,
        })

    return corrected, corrections


def analyze_line(line):
    original = line
    corrected, corrections = apply_safe_corrections(original)

    warnings = []
    review_required = False

    if WEIRD_CHARS_RE.search(original):
        warnings.append("weird_glyphs_found")
        review_required = True

    if RX_MK.search(original):
        warnings.append("latin_mK_in_gost")

    if RX_GROUP_NO_SPACE.search(original):
        warnings.append("missing_space_after_Гр.")

    if RX_DECIMAL_DOT_MEASURE.search(original):
        warnings.append("decimal_dot_in_measurement")

    if RX_TITLE_LEAK.search(original) and RX_TITLE_LEAK.search(original).start() > 20:
        warnings.append("possible_title_block_leak")

    if RX_SUSPICIOUS_EDE.search(original):
        warnings.append("suspicious_word_еде_after_сборочной")
        review_required = True

    mixed = find_mixed_script_tokens(original)

    # Не все смешанные латиница/кириллица требуют ручной проверки.
    # Например:
    # - "Гр.I" безопасно исправляется в "Гр. I";
    # - "ГОСТ ...-mK" безопасно исправляется в "ГОСТ ...-мК".
    unsafe_mixed = []

    for token in mixed:
        token_u = token.upper()

        safe = False

        if token_u.startswith("ГР."):
            safe = True

        if token == "mK":
            safe = True

        if token.endswith("mK"):
            safe = True

        if not safe:
            unsafe_mixed.append(token)

    if unsafe_mixed:
        warnings.append("mixed_latin_cyrillic_tokens: " + ", ".join(unsafe_mixed))
        review_required = True

    confidence = 1.0

    if review_required:
        confidence -= 0.25

    if warnings:
        confidence -= min(0.25, 0.05 * len(warnings))

    confidence = max(0.0, round(confidence, 2))

    return {
        "text_original": original,
        "corrected_text": corrected,
        "warnings": warnings,
        "review_required": review_required,
        "confidence": confidence,
        "corrections": corrections,
    }


def analyze_block(block):
    text_original = block.get("text", "")
    lines = text_original.splitlines()

    analyzed_lines = [analyze_line(line) for line in lines]

    corrected_text = "\n".join(l["corrected_text"] for l in analyzed_lines)

    warnings = []
    corrections = []
    review_required = False

    for line in analyzed_lines:
        warnings.extend(line["warnings"])
        corrections.extend(line["corrections"])
        if line["review_required"]:
            review_required = True

    unique_warnings = []
    for w in warnings:
        if w not in unique_warnings:
            unique_warnings.append(w)

    if analyzed_lines:
        avg_conf = sum(l["confidence"] for l in analyzed_lines) / len(analyzed_lines)
    else:
        avg_conf = 0.0

    suspicious = bool(unique_warnings)

    return {
        "text_original": text_original,
        "corrected_text": corrected_text,
        "text_quality": {
            "suspicious": suspicious,
            "review_required": review_required,
            "overall_confidence": round(avg_conf, 2),
            "warnings": unique_warnings,
        },
        "lines_quality": analyzed_lines,
        "correction_log": corrections,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/check_pdf_kompas_textlayer_fix2/output.json",
        help="Путь к output.json от КОМПАС-скрипта",
    )
    parser.add_argument(
        "--out",
        default="results/text_quality_kompas",
        help="Папка для отчёта проверки текста",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(input_path.read_text(encoding="utf-8"))

    checked = []

    report_lines = []
    report_lines.append("Проверка качества извлечённого текста ТТ")
    report_lines.append("=" * 80)

    for item in data:
        file_name = item.get("file")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(str(file_name))

        new_item = dict(item)
        new_selected = []

        for idx, block in enumerate(item.get("selected", []), start=1):
            analysis = analyze_block(block)

            new_block = dict(block)
            new_block.update(analysis)
            new_selected.append(new_block)

            tq = analysis["text_quality"]

            report_lines.append("")
            report_lines.append(f"BLOCK {idx}")
            report_lines.append(f"confidence: {tq['overall_confidence']}")
            report_lines.append(f"suspicious: {tq['suspicious']}")
            report_lines.append(f"review_required: {tq['review_required']}")

            if tq["warnings"]:
                report_lines.append("warnings:")
                for w in tq["warnings"]:
                    report_lines.append(f"  - {w}")
            else:
                report_lines.append("warnings: нет")

            if analysis["correction_log"]:
                report_lines.append("corrections:")
                for c in analysis["correction_log"]:
                    report_lines.append(f"  - {c['rule']}")
            else:
                report_lines.append("corrections: нет")

            report_lines.append("")
            report_lines.append("ORIGINAL TEXT:")
            report_lines.append(analysis["text_original"])
            report_lines.append("")
            report_lines.append("CORRECTED TEXT:")
            report_lines.append(analysis["corrected_text"])

        new_item["selected"] = new_selected
        checked.append(new_item)

    out_json = out_dir / "output_quality.json"
    out_txt = out_dir / "report.txt"

    out_json.write_text(json.dumps(checked, ensure_ascii=False, indent=2), encoding="utf-8")
    out_txt.write_text("\n".join(report_lines), encoding="utf-8")

    print("Готово:")
    print(out_json)
    print(out_txt)


if __name__ == "__main__":
    main()
