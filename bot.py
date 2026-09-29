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
    filters,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = int(os.getenv("PORT", "10000"))

RTO_API = "https://trafficchallan.com/api/rto-codes.json"
RTO_DB = {}

# =========================================================
# STATE CODES
# =========================================================

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
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "DL": "Delhi",
    "AN": "Andaman and Nicobar Islands",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DD": "Daman and Diu",
}

# =========================================================
# LOAD RTO DATABASE
# =========================================================

def load_rto_database():
    global RTO_DB

    print("[RTO] Loading RTO database...")

    try:
        req = urllib.request.Request(
            RTO_API,
            headers={"User-Agent": "VehicleOCRBot/2.0"}
        )

        with urllib.request.urlopen(req, timeout=20) as response:
            data = json.loads(
                response.read().decode("utf-8")
            )

        count = 0

        for state in data.get("states", []):
            state_name = state.get("state_name", "")

            for item in state.get("codes", []):
                code = str(
                    item.get("code", "")
                ).upper()

                code = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    code
                )

                office = str(
                    item.get("office", "")
                ).strip()

                if code:
                    RTO_DB[code] = {
                        "state": state_name,
                        "city": office
                    }
                    count += 1

        print(
            f"[RTO] Loaded {count} RTO records."
        )

    except Exception as e:
        print(
            "[RTO ERROR]",
            repr(e)
        )


# =========================================================
# RTO LOOKUP
# =========================================================

def lookup_rto(plate):

    plate = plate.upper()

    # BH series
    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{2}",
        plate
    ):
        return {
            "state": "Bharat Series",
            "city": "Not encoded in BH plate",
            "code": "BH"
        }

    if len(plate) < 4:
        return {
            "state": "Unknown",
            "city": "Unknown",
            "code": None
        }

    state = plate[:2]

    state_name = STATE_CODES.get(
        state,
        "Unknown"
    )

    # Try 3, 2 and 1 digit RTO codes.
    # Most normal plates use 1-2 digits,
    # but some datasets contain 3-digit identifiers.

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

    possible = []

    if len(digits) >= 3:
        possible.append(
            state + digits[:3]
        )

    if len(digits) >= 2:
        possible.append(
            state + digits[:2].zfill(2)
        )

    if len(digits) >= 1:
        possible.append(
            state + digits[:1].zfill(2)
        )

    for code in possible:
        if code in RTO_DB:
            return {
                "state": RTO_DB[code]["state"],
                "city": RTO_DB[code]["city"],
                "code": code
            }

    return {
        "state": state_name,
        "city": "RTO code not found",
        "code": possible[-1] if possible else None
    }


# =========================================================
# HEALTH SERVER
# =========================================================

async def health(request):
    return web.Response(
        text="Vehicle OCR Bot is running."
    )


async def start_health_server():

    app = web.Application()

    app.router.add_get("/", health)
    app.router.add_get("/health", health)

    runner = web.AppRunner(app)

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        PORT
    )

    await site.start()

    print(
        f"[SERVER] Port {PORT}"
    )

    return runner


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            "Cannot read image"
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] {w}x{h}"
    )

    # Do NOT make already-small plate photos smaller.
    if w > 1800:

        ratio = 1800 / float(w)

        image = cv2.resize(
            image,
            (
                1800,
                int(h * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# FOUR POINT PERSPECTIVE CORRECTION
# =========================================================

def order_points(points):

    pts = np.array(
        points,
        dtype=np.float32
    )

    s = pts.sum(axis=1)
    d = np.diff(
        pts,
        axis=1
    ).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array(
        [tl, tr, br, bl],
        dtype=np.float32
    )


def four_point_transform(image, points):

    rect = order_points(points)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    max_width = int(
        max(
            width_a,
            width_b
        )
    )

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    max_height = int(
        max(
            height_a,
            height_b
        )
    )

    if max_width < 50 or max_height < 15:
        return None

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        destination
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return warped


# =========================================================
# PLATE CANDIDATE DETECTION
# =========================================================

def detect_plate_crops(image):

    print(
        "[DETECT] Searching possible plate regions..."
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Preserve details while reducing noise.
    blur = cv2.bilateralFilter(
        gray,
        9,
        75,
        75
    )

    edges = cv2.Canny(
        blur,
        50,
        180
    )

    kernels = [
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (17, 5)
        ),
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (25, 7)
        ),
    ]

    candidates = []

    h, w = gray.shape

    for kernel in kernels:

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

            area = cv2.contourArea(
                contour
            )

            if area < 300:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            approx = cv2.approxPolyDP(
                contour,
                0.03 * perimeter,
                True
            )

            x, y, cw, ch = cv2.boundingRect(
                contour
            )

            if ch <= 0:
                continue

            ratio = cw / float(ch)

            # Indian plates are usually wider than tall.
            if not (
                2.0 <= ratio <= 7.5
            ):
                continue

            # Avoid taking almost the entire image.
            if cw > w * 0.95:
                continue

            if ch > h * 0.50:
                continue

            score = 0

            # Preferred ratio
            if 3.0 <= ratio <= 6.5:
                score += 3
            else:
                score += 1

            # Four-corner shape
            if len(approx) == 4:
                score += 3

            # Larger candidate
            score += min(
                5,
                int(
                    area /
                    (w * h)
                    * 100
                )
            )

            candidates.append(
                {
                    "score": score,
                    "x": x,
                    "y": y,
                    "w": cw,
                    "h": ch,
                    "approx": approx
                }
            )

    # Remove duplicate/near-duplicate boxes.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    selected = []

    for candidate in candidates:

        x1 = candidate["x"]
        y1 = candidate["y"]
        x2 = x1 + candidate["w"]
        y2 = y1 + candidate["h"]

        duplicate = False

        for old in selected:

            ox1 = old["x"]
            oy1 = old["y"]
            ox2 = ox1 + old["w"]
            oy2 = oy1 + old["h"]

            ix1 = max(
                x1,
                ox1
            )

            iy1 = max(
                y1,
                oy1
            )

            ix2 = min(
                x2,
                ox2
            )

            iy2 = min(
                y2,
                oy2
            )

            iw = max(
                0,
                ix2 - ix1
            )

            ih = max(
                0,
                iy2 - iy1
            )

            intersection = iw * ih

            area_a = (
                candidate["w"] *
                candidate["h"]
            )

            if area_a <= 0:
                continue

            if intersection / float(
                area_a
            ) > 0.65:

                duplicate = True
                break

        if not duplicate:

            selected.append(
                candidate
            )

        if len(selected) >= 8:
            break

    crops = []

    for index, candidate in enumerate(
        selected
    ):

        x = candidate["x"]
        y = candidate["y"]
        cw = candidate["w"]
        ch = candidate["h"]

        # Add generous padding because contour
        # often catches only the border.
        px = int(
            cw * 0.18
        )

        py = int(
            ch * 0.75
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
            w,
            x + cw + px
        )

        y2 = min(
            h,
            y + ch + py
        )

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size:

            print(
                f"[DETECT] Candidate {index + 1}: "
                f"{crop.shape[1]}x{crop.shape[0]}"
            )

            crops.append(
                crop
            )

    print(
        f"[DETECT] {len(crops)} candidate crops"
    )

    return crops


# =========================================================
# PREPARE OCR IMAGES
# =========================================================

def prepare_ocr_images(image):

    if image is None or image.size == 0:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # OCR needs enough horizontal resolution.
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

    elif w > 1800:

        scale = 1800 / float(w)

        gray = cv2.resize(
            gray,
            (
                1800,
                max(
                    100,
                    int(h * scale)
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    # Small denoise
    denoise = cv2.bilateralFilter(
        gray,
        7,
        50,
        50
    )

    # Contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        denoise
    )

    # OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # Adaptive
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7
    )

    # Black text / white background.
    # Also create inverted version for dark plates.
    inverted = cv2.bitwise_not(
        otsu
    )

    return [
        ("gray", gray),
        ("enhanced", enhanced),
        ("otsu", otsu),
        ("adaptive", adaptive),
        ("inverted", inverted),
    ]


# =========================================================
# OCR
# =========================================================

def run_ocr(image, psm):

    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    try:

        text = pytesseract.image_to_string(
            image,
            config=config,
            lang="eng",
            timeout=3
        )

        return text or ""

    except RuntimeError as e:

        print(
            "[OCR TIMEOUT]",
            repr(e)
        )

        return ""

    except Exception as e:

        print(
            "[OCR ERROR]",
            repr(e)
        )

        return ""


# =========================================================
# OCR CHARACTER NORMALIZATION
# =========================================================

LETTER_TO_DIGIT = {
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
    "B": "8",
}

DIGIT_TO_LETTER = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "4": "A",
    "3": "E",
    "9": "G",
}


def clean_text(text):

    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


# =========================================================
# STATE PREFIX CORRECTION
# =========================================================

def fix_state_prefix(text):

    text = clean_text(
        text
    )

    if len(text) < 2:
        return []

    prefix = text[:2]
    rest = text[2:]

    possibilities = {
        prefix
    }

    # Character confusion in first two positions.
    maps = {
        "0": ["O", "D"],
        "1": ["I", "L", "T"],
        "2": ["Z"],
        "5": ["S"],
        "6": ["G"],
        "7": ["T"],
        "8": ["B"],
        "9": ["G"],
    }

    p0 = [prefix[0]]
    p1 = [prefix[1]]

    if prefix[0] in maps:
        p0.extend(
            maps[prefix[0]]
        )

    if prefix[1] in maps:
        p1.extend(
            maps[prefix[1]]
        )

    for a in p0:
        for b in p1:

            candidate = a + b

            if candidate in STATE_CODES:
                possibilities.add(
                    candidate
                )

    return [
        p + rest
        for p in possibilities
    ]


# =========================================================
# GENERATE PLATE CANDIDATES
# =========================================================

def generate_plate_candidates(text):

    raw = clean_text(
        text
    )

    if len(raw) < 6:
        return []

    candidates = []

    # -----------------------------------------------------
    # Direct
    # -----------------------------------------------------

    candidates.append(
        raw
    )

    # -----------------------------------------------------
    # Correct state prefix
    # -----------------------------------------------------

    for item in fix_state_prefix(
        raw
    ):

        candidates.append(
            item
        )

    # -----------------------------------------------------
    # Correct numeric RTO portion
    # -----------------------------------------------------

    for item in list(
        candidates
    ):

        if len(item) < 4:
            continue

        state = item[:2]

        if state not in STATE_CODES:
            continue

        rest = item[2:]

        # Try correcting first 1-3 characters
        # after state to digits.
        for digit_count in (
            1,
            2,
            3
        ):

            if len(rest) <= digit_count:
                continue

            number_part = rest[
                :digit_count
            ]

            remaining = rest[
                digit_count:
            ]

            fixed_number = ""

            possible = True

            for char in number_part:

                if char.isdigit():

                    fixed_number += char

                elif char in LETTER_TO_DIGIT:

                    fixed_number += (
                        LETTER_TO_DIGIT[
                            char
                        ]
                    )

                else:

                    possible = False
                    break

            if not possible:
                continue

            candidate = (
                state +
                fixed_number +
                remaining
            )

            candidates.append(
                candidate
            )

    # Unique
    return list(
        dict.fromkeys(
            candidates
        )
    )


# =========================================================
# VALIDATE PLATE
# =========================================================

def plate_score(text):

    text = clean_text(
        text
    )

    # BH
    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{2}",
        text
    ):

        return 100

    if len(text) < 7 or len(text) > 13:
        return 0

    if text[:2] not in STATE_CODES:
        return 0

    # Standard formats
    if re.fullmatch(
        r"[A-Z]{2}\d{1,3}[A-Z]{1,3}\d{1,4}",
        text
    ):

        score = 90

        # Normal Indian plates generally contain
        # several digits.
        if len(re.findall(
            r"\d",
            text
        )) >= 4:

            score += 5

        return score

    if re.fullmatch(
        r"[A-Z]{2}\d{1,3}\d{4,5}",
        text
    ):

        return 75

    return 0


# =========================================================
# EXTRACT CANDIDATES FROM OCR
# =========================================================

def extract_from_ocr(text):

    if not text:
        return []

    raw = text.upper()

    # Keep OCR tokens separately.
    tokens = re.findall(
        r"[A-Z0-9]+",
        raw
    )

    if not tokens:
        return []

    strings = []

    # Every token
    strings.extend(
        tokens
    )

    # Join all OCR tokens.
    strings.append(
        "".join(tokens)
    )

    # Join neighboring tokens.
    for i in range(
        len(tokens) - 1
    ):

        strings.append(
            tokens[i] +
            tokens[i + 1]
        )

    results = []

    for string in strings:

        for candidate in generate_plate_candidates(
            string
        ):

            score = plate_score(
                candidate
            )

            if score > 0:

                results.append(
                    (
                        candidate,
                        score
                    )
                )

    return results


# =========================================================
# OCR ONE IMAGE
# =========================================================

def analyse_one_image(image, label):

    all_results = []

    variants = prepare_ocr_images(
        image
    )

    # PSM:
    # 6 = block
    # 7 = single line
    # 8 = single word
    # 13 = raw line
    psms = (
        7,
        8,
        13
    )

    for variant_name, prepared in variants:

        for psm in psms:

            text = run_ocr(
                prepared,
                psm
            )

            print(
                f"[RAW OCR] "
                f"{label}/{variant_name}/PSM{psm}: "
                f"{repr(text)}"
            )

            found = extract_from_ocr(
                text
            )

            if found:

                print(
                    f"[OCR MATCH] "
                    f"{label}/{variant_name}/PSM{psm}: "
                    f"{found}"
                )

                all_results.extend(
                    found
                )

    return all_results


# =========================================================
# FINAL VOTING
# =========================================================

def choose_best(results):

    if not results:
        return None

    votes = Counter()

    best_score = {}

    for plate, score in results:

        votes[plate] += 1

        if (
            plate not in best_score
            or score > best_score[plate]
        ):

            best_score[plate] = score

    print(
        "[VOTE TABLE]",
        dict(votes)
    )

    ranked = []

    for plate, count in votes.items():

        score = best_score[plate]

        # Multiple independent OCR hits are valuable.
        final_score = (
            score +
            min(count, 5) * 12
        )

        ranked.append(
            (
                final_score,
                count,
                plate
            )
        )

    ranked.sort(
        reverse=True
    )

    print(
        "[RANKED]",
        ranked
    )

    best_total, count, plate = ranked[0]

    # Strong plate
    if (
        count >= 2
        and best_score[plate] >= 75
    ):

        print(
            f"[FINAL PLATE] {plate} "
            f"votes={count}"
        )

        return plate

    # A single very strong OCR read
    # is accepted only for a strict valid format.
    if (
        count == 1
        and best_score[plate] >= 90
    ):

        print(
            f"[FINAL PLATE] {plate} "
            f"single strong read"
        )

        return plate

    print(
        "[FINAL] Confidence too low."
    )

    return None


# =========================================================
# COMPLETE OCR PIPELINE
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    all_results = []

    # =====================================================
    # STEP 1
    # Original image
    # =====================================================

    print(
        "[STEP 1] Direct OCR on original image"
    )

    all_results.extend(
        analyse_one_image(
            image,
            "ORIGINAL"
        )
    )

    # =====================================================
    # STEP 2
    # Detect plate-shaped regions
    # =====================================================

    crops = detect_plate_crops(
        image
    )

    for index, crop in enumerate(
        crops
    ):

        label = (
            f"CROP_{index + 1}"
        )

        print(
            f"[STEP 2] OCR {label}"
        )

        all_results.extend(
            analyse_one_image(
                crop,
                label
            )
        )

    # =====================================================
    # STEP 3
    # Center crop fallback
    # =====================================================

    h, w = image.shape[:2]

    # Useful when plate detection contour fails.
    center = image[
        int(h * 0.20):
        int(h * 0.85),
        int(w * 0.05):
        int(w * 0.95)
    ]

    print(
        "[STEP 3] OCR center fallback"
    )

    all_results.extend(
        analyse_one_image(
            center,
            "CENTER"
        )
    )

    print(
        f"[TOTAL OCR RESULTS] "
        f"{len(all_results)}"
    )

    return choose_best(
        all_results
    )


# =========================================================
# TELEGRAM START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE REGISTRATION SCANNER\n\n"
        "📸 Number plate ki clear photo bhejo.\n\n"
        "Bot:\n"
        "🔢 Registration number read karega\n"
        "🇮🇳 State identify karega\n"
        "🏙️ Registration/RTO city lookup karega\n\n"
        "⚠️ City ka matlab registration area hai, "
        "current vehicle location nahi."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Best OCR ke liye:\n\n"
        "• Plate close-up\n"
        "• Clear focus\n"
        "• Good lighting\n"
        "• Reflection minimum\n"
        "• Plate tedhi na ho\n\n"
        "Full vehicle photo bhi supported hai."
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    status = await update.message.reply_text(
        "🔎 Plate scan ho rahi hai...\n\n"
        "AI OCR + image correction + RTO lookup."
    )

    temp_path = None

    try:

        # Telegram gives multiple sizes.
        # Last = highest available.
        photo = update.message.photo[-1]

        tg_file = await context.bot.get_file(
            photo.file_id
        )

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:

            temp_path = temp.name

        await tg_file.download_to_drive(
            temp_path
        )

        print(
            "=========================================="
        )

        print(
            "[PHOTO] New photo received"
        )

        plate = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        print(
            "=========================================="
        )

        if not plate:

            await status.edit_text(
                "❌ Plate number reliably read nahi ho paya.\n\n"
                "OCR ne multiple methods se try kiya, "
                "lekin confidence sufficient nahi tha.\n\n"
                "Render logs mein `[RAW OCR]` aur "
                "`[OCR MATCH]` lines check karo."
            )

            return

        info = lookup_rto(
            plate
        )

        response = (
            "✅ PLATE FOUND\n\n"
            f"🔢 Registration: {plate}\n"
            f"🇮🇳 State: {info['state']}\n"
        )

        if info.get("code"):

            response += (
                f"🏢 RTO Code: "
                f"{info['code']}\n"
            )

        response += (
            f"🏙️ Registration City/Area: "
            f"{info['city']}\n"
        )

        response += (
            "\n"
            "ℹ️ Yeh registration/RTO area hai.\n"
            "📍 Yeh vehicle ki current location nahi hai.\n"
            "👤 Owner details provide nahi ki jaati."
        )

        await status.edit_text(
            response
        )

    except Exception as e:

        print(
            "[PHOTO ERROR]",
            repr(e)
        )

        try:

            await status.edit_text(
                "❌ Image process nahi ho saki.\n"
                "Please dobara try karo."
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


# =========================================================
# TEXT
# =========================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Number plate ki photo bhejo."
    )


# =========================================================
# MAIN
# =========================================================

async def main():

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN environment variable missing."
        )

    # RTO database
    load_rto_database()

    health_runner = (
        await start_health_server()
    )

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
        "VEHICLE OCR BOT V2"
    )
    print(
        "OCR: ADVANCED DEBUG"
    )
    print(
        "PLATE DETECTION: ENABLED"
    )
    print(
        "PERSPECTIVE CORRECTION: ENABLED"
    )
    print(
        "RTO DATABASE: ENABLED"
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
        await health_runner.cleanup()


if __name__ == "__main__":

    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Bot stopped."
)
