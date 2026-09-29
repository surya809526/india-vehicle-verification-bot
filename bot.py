import os
import re
import json
import asyncio
import tempfile
import urllib.request
from collections import Counter

import cv2
import numpy as np
import pytesseract
from aiohttp import web

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))

RTO_API = "https://trafficchallan.com/api/rto-codes.json"
RTO_DB = {}


# ============================================================
# INDIA STATE / UT CODES
# ============================================================

STATE_CODES = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "MP": "Madhya Pradesh",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "DL": "Delhi",
    "AN": "Andaman and Nicobar Islands",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Daman and Diu",
}


# ============================================================
# RTO DATABASE
# ============================================================

def load_rto_database():

    global RTO_DB

    print("[RTO] Downloading database...")

    try:

        request = urllib.request.Request(
            RTO_API,
            headers={
                "User-Agent": "IndiaVehicleOCR/1.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=25
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

        count = 0

        # Format 1:
        # {"states":[{"state_name":"...","codes":[...]}]}

        if isinstance(data, dict):

            states = data.get("states", [])

            for state in states:

                state_name = str(
                    state.get(
                        "state_name",
                        ""
                    )
                ).strip()

                codes = state.get(
                    "codes",
                    []
                )

                for item in codes:

                    code = str(
                        item.get(
                            "code",
                            ""
                        )
                    ).upper()

                    office = str(
                        item.get(
                            "office",
                            ""
                        )
                    ).strip()

                    code = re.sub(
                        r"[^A-Z0-9]",
                        "",
                        code
                    )

                    if code:

                        RTO_DB[code] = {
                            "state": state_name,
                            "city": office
                        }

                        count += 1

        # Format 2:
        # direct dictionary
        elif isinstance(data, list):

            for item in data:

                if not isinstance(
                    item,
                    dict
                ):
                    continue

                code = str(
                    item.get("code", "")
                ).upper()

                office = str(
                    item.get(
                        "office",
                        item.get(
                            "city",
                            ""
                        )
                    )
                ).strip()

                state_name = str(
                    item.get(
                        "state",
                        ""
                    )
                ).strip()

                code = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    code
                )

                if code:

                    RTO_DB[code] = {
                        "state": state_name,
                        "city": office
                    }

                    count += 1

        print(
            f"[RTO] Loaded {count} records"
        )

    except Exception as e:

        print(
            "[RTO ERROR]",
            repr(e)
        )


# ============================================================
# RTO LOOKUP
# ============================================================

def lookup_rto(plate):

    plate = clean_text(
        plate
    )

    # BH SERIES
    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{1,2}",
        plate
    ):

        return {
            "state": "Bharat Series",
            "city": "No state/RTO city encoded",
            "code": "BH"
        }

    if len(plate) < 4:

        return {
            "state": "Unknown",
            "city": "Unknown",
            "code": None
        }

    state_code = plate[:2]

    state_name = STATE_CODES.get(
        state_code,
        "Unknown"
    )

    # State + RTO number
    match = re.match(
        r"^([A-Z]{2})(\d{1,3})",
        plate
    )

    if not match:

        return {
            "state": state_name,
            "city": "Unknown",
            "code": None
        }

    digits = match.group(2)

    possible_codes = []

    # Longest first
    if len(digits) >= 3:

        possible_codes.append(
            state_code + digits[:3]
        )

    if len(digits) >= 2:

        possible_codes.append(
            state_code + digits[:2]
        )

    possible_codes.append(
        state_code + digits[:1]
    )

    for code in possible_codes:

        if code in RTO_DB:

            return {
                "state": RTO_DB[code]["state"]
                or state_name,
                "city": RTO_DB[code]["city"]
                or "Unknown",
                "code": code
            }

    return {
        "state": state_name,
        "city": "RTO code not found",
        "code": possible_codes[0]
    }


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.upper()

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


# ============================================================
# COMMON OCR ERRORS
# ============================================================

def correct_ocr_text(text):

    text = clean_text(
        text
    )

    if not text:
        return ""

    # OCR often reads these incorrectly.
    replacements = str.maketrans({
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "G": "6",
        "T": "7",
        "Y": "7",
        "B": "8"
    })

    return text.translate(
        replacements
    )


# ============================================================
# HEALTH SERVER
# ============================================================

async def health(request):

    return web.Response(
        text="India Vehicle OCR Bot running"
    )


async def start_server():

    app = web.Application()

    app.router.add_get(
        "/",
        health
    )

    app.router.add_get(
        "/health",
        health
    )

    runner = web.AppRunner(
        app
    )

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        PORT
    )

    await site.start()

    print(
        f"[SERVER] Running on {PORT}"
    )

    return runner


# ============================================================
# IMAGE LOAD
# ============================================================

def load_image(path):

    image = cv2.imread(
        path
    )

    if image is None:

        raise ValueError(
            "Unable to read image"
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] {w}x{h}"
    )

    # Keep high enough resolution for OCR
    if w > 2400:

        ratio = 2400 / float(w)

        image = cv2.resize(
            image,
            (
                2400,
                int(h * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# PLATE CANDIDATE DETECTION
# ============================================================

def detect_plate_candidates(image):

    h, w = image.shape[:2]

    candidates = []

    # --------------------------------------------------------
    # A. Whole image
    # --------------------------------------------------------

    candidates.append(
        (
            1,
            image.copy()
        )
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # B. Edge based detection
    # --------------------------------------------------------

    blur = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    edges = cv2.Canny(
        blur,
        40,
        180
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 7)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 80 or ch < 20:
            continue

        ratio = cw / float(ch)

        # Covers many car/bike plate proportions.
        if not (
            1.4 <= ratio <= 9.0
        ):
            continue

        area_ratio = (
            cw * ch
        ) / float(
            w * h
        )

        if area_ratio < 0.0005:
            continue

        if area_ratio > 0.60:
            continue

        pad_x = int(
            cw * 0.18
        )

        pad_y = int(
            ch * 0.45
        )

        x1 = max(
            0,
            x - pad_x
        )

        y1 = max(
            0,
            y - pad_y
        )

        x2 = min(
            w,
            x + cw + pad_x
        )

        y2 = min(
            h,
            y + ch + pad_y
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size:

            score = (
                abs(
                    ratio - 4.0
                )
            )

            candidates.append(
                (
                    10 - score,
                    crop
                )
            )

    # Highest score first
    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Remove almost identical crops
    final = []

    for score, crop in candidates:

        if crop is None:
            continue

        if crop.size == 0:
            continue

        duplicate = False

        ch, cw = crop.shape[:2]

        for old in final:

            oh, ow = old.shape[:2]

            if abs(
                (cw / max(ch, 1)) -
                (ow / max(oh, 1))
            ) < 0.05:

                if abs(
                    cw - ow
                ) < 20:

                    duplicate = True
                    break

        if not duplicate:

            final.append(
                crop
            )

        if len(final) >= 10:
            break

    print(
        f"[PLATE DETECTOR] "
        f"{len(final)} candidates"
    )

    return final


# ============================================================
# OCR IMAGE VARIANTS
# ============================================================

def make_ocr_variants(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # Upscale small plates
    if w < 1000:

        scale = 1200 / float(w)

        gray = cv2.resize(
            gray,
            (
                1200,
                max(
                    100,
                    int(h * scale)
                )
            ),
            interpolation=cv2.INTER_CUBIC
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    denoise = cv2.GaussianBlur(
        enhanced,
        (3, 3),
        0
    )

    _, otsu = cv2.threshold(
        denoise,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    inverted = cv2.bitwise_not(
        otsu
    )

    return [
        ("gray", gray),
        ("enhanced", enhanced),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("inverted", inverted)
    ]


# ============================================================
# WHOLE PLATE OCR
# ============================================================

def whole_plate_ocr(image):

    results = []

    variants = make_ocr_variants(
        image
    )

    for name, variant in variants:

        for psm in (
            6,
            7,
            8,
            13
        ):

            config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=config,
                    lang="eng",
                    timeout=2.5
                )

                text = clean_text(
                    text
                )

                if text:

                    print(
                        f"[RAW OCR] "
                        f"{name}/PSM{psm}: "
                        f"'{text}'"
                    )

                    results.append(
                        text
                    )

            except Exception as e:

                print(
                    "[OCR ERROR]",
                    name,
                    psm,
                    repr(e)
                )

    return results


# ============================================================
# TWO-LINE DETECTION
# ============================================================

def split_plate_lines(plate):

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    if h < 40:
        return [plate]

    # Horizontal darkness/edge profile
    edges = cv2.Canny(
        gray,
        50,
        150
    )

    profile = np.sum(
        edges,
        axis=1
    )

    # Smooth profile
    profile = cv2.GaussianBlur(
        profile.astype(
            np.float32
        ).reshape(-1, 1),
        (1, 9),
        0
    ).flatten()

    # Find strongest horizontal regions
    threshold = np.percentile(
        profile,
        55
    )

    active = profile > threshold

    groups = []

    start = None

    for i, value in enumerate(
        active
    ):

        if value and start is None:

            start = i

        elif not value and start is not None:

            if i - start >= 5:

                groups.append(
                    (
                        start,
                        i
                    )
                )

            start = None

    if start is not None:

        groups.append(
            (
                start,
                h
            )
        )

    # If no useful split:
    if len(groups) < 2:

        mid = h // 2

        top = plate[
            :mid
        ]

        bottom = plate[
            mid:
        ]

        if (
            top.shape[0] >= 15
            and
            bottom.shape[0] >= 15
        ):

            return [
                top,
                bottom
            ]

        return [plate]

    # Merge nearby groups
    merged = []

    for group in groups:

        if not merged:

            merged.append(
                list(group)
            )

        else:

            prev = merged[-1]

            if group[0] - prev[1] < 10:

                prev[1] = group[1]

            else:

                merged.append(
                    list(group)
                )

    # Use at most 2 main lines
    if len(merged) > 2:

        merged = sorted(
            merged,
            key=lambda x:
            x[1] - x[0],
            reverse=True
        )[:2]

        merged.sort(
            key=lambda x:
            x[0]
        )

    lines = []

    for y1, y2 in merged:

        pad = 5

        y1 = max(
            0,
            y1 - pad
        )

        y2 = min(
            h,
            y2 + pad
        )

        line = plate[
            y1:y2,
            :
        ]

        if line.shape[0] >= 15:

            lines.append(
                line
            )

    if len(lines) >= 2:

        print(
            "[2-LINE] Two plate lines detected"
        )

        return lines

    return [plate]


# ============================================================
# OCR A SINGLE LINE
# ============================================================

def ocr_single_line(line):

    results = []

    for name, img in make_ocr_variants(
        line
    ):

        for psm in (
            7,
            8,
            13
        ):

            config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            try:

                text = pytesseract.image_to_string(
                    img,
                    config=config,
                    lang="eng",
                    timeout=2
                )

                text = clean_text(
                    text
                )

                if text:

                    print(
                        f"[LINE OCR] "
                        f"{name}/PSM{psm}: "
                        f"{text}"
                    )

                    results.append(
                        text
                    )

            except Exception:
                pass

    return results


# ============================================================
# CHARACTER SEGMENTATION
# ============================================================

def character_ocr(char_img):

    gray = cv2.cvtColor(
        char_img,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.resize(
        gray,
        None,
        fx=6,
        fy=6,
        interpolation=cv2.INTER_CUBIC
    )

    variants = [
        gray
    ]

    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(
        otsu
    )

    results = []

    for img in variants:

        for psm in (
            10,
            8
        ):

            config = (
                f"--oem 3 --psm {psm} "
                "-c tessedit_char_whitelist="
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            try:

                text = pytesseract.image_to_string(
                    img,
                    config=config,
                    lang="eng",
                    timeout=1.5
                )

                text = clean_text(
                    text
                )

                if text:

                    results.append(
                        text[0]
                    )

            except Exception:
                pass

    if not results:
        return ""

    return Counter(
        results
    ).most_common(1)[0][0]


def segment_line(line):

    gray = cv2.cvtColor(
        line,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    if h < 20 or w < 50:

        return ""

    # Normalize
    target_h = 180

    scale = target_h / float(h)

    gray = cv2.resize(
        gray,
        (
            max(
                200,
                int(w * scale)
            ),
            target_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(
        gray
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    masks = [
        binary,
        cv2.bitwise_not(binary)
    ]

    best = []

    for mask in masks:

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        boxes = []

        for contour in contours:

            x, y, cw, ch = cv2.boundingRect(
                contour
            )

            area = cw * ch

            if area < 25:
                continue

            height_ratio = (
                ch / float(
                    gray.shape[0]
                )
            )

            width_ratio = (
                cw / float(
                    gray.shape[1]
                )
            )

            if not (
                0.35 <=
                height_ratio <=
                0.98
            ):
                continue

            if not (
                0.012 <=
                width_ratio <=
                0.30
            ):
                continue

            # Character generally taller than wide
            if cw > ch * 1.25:
                continue

            boxes.append(
                (
                    x,
                    y,
                    cw,
                    ch
                )
            )

        boxes.sort(
            key=lambda b: b[0]
        )

        if len(boxes) >= 2:

            if not best or (
                2 <= len(boxes) <= 10
            ):

                best = boxes

    if not best:

        return ""

    # A line normally contains <= 10 useful characters
    if len(best) > 10:

        best = sorted(
            best,
            key=lambda b:
            b[2] * b[3],
            reverse=True
        )[:10]

        best.sort(
            key=lambda b:
            b[0]
        )

    print(
        f"[CHAR BOXES] {len(best)}"
    )

    output = []

    for index, (
        x,
        y,
        cw,
        ch
    ) in enumerate(best):

        px = max(
            3,
            int(cw * 0.20)
        )

        py = max(
            3,
            int(ch * 0.15)
        )

        x1 = max(
            0,
            x - px
        )

        y1 = max(
            0,
            y - py
        )

        x2 = min(
            gray.shape[1],
            x + cw + px
        )

        y2 = min(
            gray.shape[0],
            y + ch + py
        )

        char_img = gray[
            y1:y2,
            x1:x2
        ]

        if char_img.size == 0:
            continue

        char_img = cv2.cvtColor(
            char_img,
            cv2.COLOR_GRAY2BGR
        )

        character = character_ocr(
            char_img
        )

        print(
            f"[CHAR {index + 1}] "
            f"{character or '?'}"
        )

        if character:

            output.append(
                character
            )

    result = "".join(
        output
    )

    print(
        "[CHAR LINE]",
        result
    )

    return result


# ============================================================
# BUILD TWO-LINE REGISTRATION
# ============================================================

def build_registration_from_lines(
    lines
):

    if not lines:
        return []

    possibilities = []

    # --------------------------------------------------------
    # OCR each line
    # --------------------------------------------------------

    line_texts = []

    for line in lines:

        texts = ocr_single_line(
            line
        )

        char_text = segment_line(
            line
        )

        if char_text:
            texts.append(
                char_text
            )

        cleaned = []

        for text in texts:

            text = clean_text(
                text
            )

            if text:

                cleaned.append(
                    text
                )

        line_texts.append(
            cleaned
        )

    # --------------------------------------------------------
    # Combine all possible line readings
    # --------------------------------------------------------

    if len(line_texts) == 1:

        for value in line_texts[0]:

            possibilities.append(
                value
            )

    else:

        first = line_texts[0]
        second = line_texts[1]

        for a in first:

            for b in second:

                combined = (
                    clean_text(a) +
                    clean_text(b)
                )

                if combined:

                    possibilities.append(
                        combined
                    )

    return possibilities


# ============================================================
# PLATE VALIDATION
# ============================================================

def valid_registration(text):

    text = clean_text(
        text
    )

    # BH:
    # YYBH####XX
    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{1,2}",
        text
    ):

        return True

    # Standard:
    # UP78HM7865
    # DL01AB1234
    # MH12AB1234
    if re.fullmatch(
        r"[A-Z]{2}\d{1,3}[A-Z]{1,3}\d{1,4}",
        text
    ):

        return (
            text[:2]
            in STATE_CODES
        )

    return False


# ============================================================
# CANDIDATE GENERATION
# ============================================================

def generate_candidates(raw):

    raw = clean_text(
        raw
    )

    if not raw:
        return []

    results = []

    # Direct
    if valid_registration(
        raw
    ):

        results.append(
            raw
        )

    # Search state codes inside OCR
    for state in STATE_CODES:

        positions = [
            m.start()
            for m in re.finditer(
                state,
                raw
            )
        ]

        for pos in positions:

            part = raw[
                pos:
                pos + 13
            ]

            # Direct
            if valid_registration(
                part
            ):

                results.append(
                    part
                )

            # OCR correction
            corrected = correct_ocr_text(
                part
            )

            if valid_registration(
                corrected
            ):

                results.append(
                    corrected
                )

    # Try OCR correction
    corrected = correct_ocr_text(
        raw
    )

    if valid_registration(
        corrected
    ):

        results.append(
            corrected
        )

    return results


# ============================================================
# ANALYSE ONE PLATE
# ============================================================

def analyse_plate(
    plate,
    label
):

    print(
        "--------------------------------------"
    )

    print(
        f"[ANALYSE PLATE] {label}"
    )

    results = []

    # --------------------------------------------------------
    # Whole plate OCR
    # --------------------------------------------------------

    whole = whole_plate_ocr(
        plate
    )

    for text in whole:

        results.extend(
            generate_candidates(
                text
            )
        )

    # --------------------------------------------------------
    # Two-line support
    # --------------------------------------------------------

    lines = split_plate_lines(
        plate
    )

    print(
        f"[LINES] {len(lines)}"
    )

    combined = (
        build_registration_from_lines(
            lines
        )
    )

    for text in combined:

        results.extend(
            generate_candidates(
                text
            )
        )

    print(
        "[PLATE VALID RESULTS]",
        results
    )

    return results


# ============================================================
# COMPLETE IMAGE ANALYSIS
# ============================================================

def analyse_vehicle(
    path
):

    image = load_image(
        path
    )

    all_results = []

    # --------------------------------------------------------
    # TESSERACT TEST
    # --------------------------------------------------------

    try:

        print(
            "[TESSERACT]",
            pytesseract.get_tesseract_version()
        )

    except Exception as e:

        print(
            "[TESSERACT ERROR]",
            repr(e)
        )

    # --------------------------------------------------------
    # Detect plate candidates
    # --------------------------------------------------------

    candidates = detect_plate_candidates(
        image
    )

    for index, plate in enumerate(
        candidates
    ):

        result = analyse_plate(
            plate,
            f"CANDIDATE_{index + 1}"
        )

        all_results.extend(
            result
        )

    # --------------------------------------------------------
    # Center crop fallback
    # --------------------------------------------------------

    h, w = image.shape[:2]

    center = image[
        int(h * 0.15):
        int(h * 0.95),
        int(w * 0.05):
        int(w * 0.95)
    ]

    result = analyse_plate(
        center,
        "CENTER"
    )

    all_results.extend(
        result
    )

    print(
        "======================================"
    )

    print(
        "[ALL OCR CANDIDATES]",
        all_results
    )

    if not all_results:

        return None

    counter = Counter(
        all_results
    )

    print(
        "[VOTES]",
        dict(counter)
    )

    return counter.most_common(
        1
    )[0][0]


# ============================================================
# TELEGRAM START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE SCANNER\n\n"
        "📸 Kisi bhi Indian vehicle ki "
        "number-plate photo bhejo.\n\n"
        "Supported:\n"
        "🏍️ Bike / Scooter\n"
        "🚗 Car / SUV\n"
        "🛺 Auto / E-Rickshaw\n"
        "🚌 Bus\n"
        "🚚 Truck\n"
        "🚜 Tractor\n"
        "🚐 Commercial vehicles\n"
        "⬜ White plates\n"
        "🟨 Yellow plates\n"
        "1️⃣ Single-line plates\n"
        "2️⃣ Two-line plates\n"
        "🇮🇳 BH-Series\n\n"
        "Bot registration number, "
        "state aur RTO registration area "
        "identify karega.\n\n"
        "⚠️ Registration area current "
        "vehicle location nahi hai."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Plate ki clear photo bhejo.\n\n"
        "Camera se li hui close-up photo "
        "best rahegi."
    )


# ============================================================
# PHOTO HANDLER
# ============================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    status = await update.message.reply_text(
        "🔎 Vehicle plate scan ho rahi hai...\n\n"
        "Plate detection + single/two-line OCR "
        "+ character OCR + RTO lookup."
    )

    temp_path = None

    try:

        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:

            temp_path = temp.name

        await telegram_file.download_to_drive(
            temp_path
        )

        print(
            "=========================================="
        )

        print(
            "[PHOTO] Vehicle image received"
        )

        plate = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        if not plate:

            await status.edit_text(
                "❌ Number plate read nahi ho payi.\n\n"
                "OCR ko reliable registration number "
                "nahi mila.\n\n"
                "Render logs mein:\n"
                "`[RAW OCR]`\n"
                "`[LINES]`\n"
                "`[CHAR BOXES]`\n"
                "`[CHAR 1]`\n"
                "`[CHAR 2]`\n"
                "check karo."
            )

            return

        info = lookup_rto(
            plate
        )

        response = (
            "✅ VEHICLE REGISTRATION FOUND\n\n"
            f"🔢 Registration: {plate}\n"
            f"🇮🇳 State/UT: {info['state']}\n"
        )

        if info.get("code"):

            response += (
                f"🏢 RTO Code: "
                f"{info['code']}\n"
            )

        response += (
            f"🏙️ Registration Area: "
            f"{info['city']}\n\n"
            "ℹ️ Registration area vehicle ki "
            "current location nahi batata."
        )

        await status.edit_text(
            response
        )

    except Exception as e:

        print(
            "[HANDLER ERROR]",
            repr(e)
        )

        try:

            await status.edit_text(
                "❌ Processing error.\n"
                "Render logs check karo."
            )

        except Exception:
            pass

    finally:

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except Exception:
                pass


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Number plate ki photo bhejo."
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN environment variable missing"
        )

    load_rto_database()

    server = await start_server()

    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start
        )
    )

    application.add_handler(
        CommandHandler(
            "help",
            help_command
        )
    )

    application.add_handler(
        MessageHandler(
            filters.PHOTO,
            photo_handler
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT &
            ~filters.COMMAND,
            text_handler
        )
    )

    print(
        "=========================================="
    )

    print(
        "INDIA VEHICLE OCR BOT"
    )

    print(
        "Universal plate detection: ON"
    )

    print(
        "Single-line OCR: ON"
    )

    print(
        "Two-line OCR: ON"
    )

    print(
        "Character OCR: ON"
    )

    print(
        "BH-Series: ON"
    )

    print(
        "RTO Lookup: ON"
    )

    print(
        "=========================================="
    )

    await application.initialize()

    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

    print(
        "[BOT] RUNNING"
    )

    try:

        await asyncio.Event().wait()

    finally:

        await application.updater.stop()
        await application.stop()
        await application.shutdown()
        await server.cleanup()


if __name__ == "__main__":

    asyncio.run(
        main()
        )
