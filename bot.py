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
# RTO DATABASE
# =========================================================

def load_rto_database():
    global RTO_DB

    print("[RTO] Loading database...")

    try:
        req = urllib.request.Request(
            RTO_API,
            headers={"User-Agent": "VehicleOCRBot/3.0"}
        )

        with urllib.request.urlopen(
            req,
            timeout=20
        ) as response:

            data = json.loads(
                response.read().decode("utf-8")
            )

        count = 0

        for state in data.get("states", []):

            state_name = state.get(
                "state_name",
                ""
            )

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
            f"[RTO] Loaded {count} records."
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

    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{2}",
        plate
    ):

        return {
            "state": "Bharat Series",
            "city": "State/city is not encoded",
            "code": "BH"
        }

    state = plate[:2]

    state_name = STATE_CODES.get(
        state,
        "Unknown"
    )

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
        "code": possible[-1]
        if possible else None
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

    app.router.add_get(
        "/",
        health
    )

    app.router.add_get(
        "/health",
        health
    )

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
# IMAGE
# =========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            "Image cannot be loaded."
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] {w}x{h}"
    )

    if w > 2000:

        ratio = 2000 / float(w)

        image = cv2.resize(
            image,
            (
                2000,
                int(h * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# OCR NORMALIZATION
# =========================================================

def clean(text):

    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper()
    )


# =========================================================
# TESSERACT CHARACTER OCR
# =========================================================

def character_ocr(char_img):

    # Character image is made large.
    char_img = cv2.resize(
        char_img,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    variants = []

    gray = cv2.cvtColor(
        char_img,
        cv2.COLOR_BGR2GRAY
    ) if len(char_img.shape) == 3 else char_img

    variants.append(gray)

    _, threshold = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(threshold)

    results = []

    for variant in variants:

        for psm in (10, 8):

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
                    timeout=2
                )

                text = clean(text)

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


# =========================================================
# CHARACTER SEGMENTATION
# =========================================================

def segment_characters(plate):

    if plate is None:
        return []

    if plate.size == 0:
        return []

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # Make plate horizontal.
    if h > w:
        plate = cv2.rotate(
            plate,
            cv2.ROTATE_90_CLOCKWISE
        )

        gray = cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2GRAY
        )

        h, w = gray.shape

    # Normalize height.
    target_h = 160

    scale = target_h / float(h)

    gray = cv2.resize(
        gray,
        (
            max(200, int(w * scale)),
            target_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    h, w = gray.shape

    # Contrast.
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    gray = clahe.apply(gray)

    # Threshold.
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # We need both normal and inverted because
    # Indian plates can differ in foreground/background.
    masks = [
        binary,
        cv2.bitwise_not(binary)
    ]

    best_boxes = []

    for mask_index, mask in enumerate(
        masks
    ):

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

            if area < 30:
                continue

            # Character should occupy a reasonable
            # vertical portion of plate.
            height_ratio = ch / float(h)

            width_ratio = cw / float(w)

            if not (
                0.35 <= height_ratio <= 0.95
            ):
                continue

            if not (
                0.015 <= width_ratio <= 0.25
            ):
                continue

            if cw > ch * 1.2:
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

        # Indian plate commonly has 8-12 characters.
        if 5 <= len(boxes) <= 14:

            best_boxes = boxes

            print(
                f"[SEGMENT] Mask {mask_index}: "
                f"{len(boxes)} character boxes"
            )

            break

        # Keep closest candidate.
        if len(boxes) > len(best_boxes):

            best_boxes = boxes

    # If segmentation produced too many boxes,
    # keep the largest reasonable horizontal sequence.
    if len(best_boxes) > 14:

        best_boxes = sorted(
            best_boxes,
            key=lambda b: b[2] * b[3],
            reverse=True
        )[:14]

        best_boxes.sort(
            key=lambda b: b[0]
        )

    print(
        f"[SEGMENT] Final boxes: "
        f"{len(best_boxes)}"
    )

    chars = []

    for index, (
        x,
        y,
        cw,
        ch
    ) in enumerate(best_boxes):

        pad_x = max(
            2,
            int(cw * 0.20)
        )

        pad_y = max(
            2,
            int(ch * 0.12)
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

        char_img = cv2.cvtColor(
            gray[y1:y2, x1:x2],
            cv2.COLOR_GRAY2BGR
        )

        if char_img.size:

            chars.append(
                char_img
            )

    return chars


# =========================================================
# PLATE RECTANGLE DETECTION
# =========================================================

def detect_plate_candidates(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

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

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (21, 7)
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

    h, w = gray.shape

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 100 or ch < 20:
            continue

        ratio = cw / float(ch)

        if not (
            2.0 <= ratio <= 8.0
        ):
            continue

        if cw > w * 0.95:
            continue

        score = 0

        if 3.0 <= ratio <= 6.5:
            score += 4

        else:
            score += 2

        area = cw * ch

        score += min(
            5,
            int(
                area /
                (w * h)
                * 100
            )
        )

        candidates.append(
            (
                score,
                x,
                y,
                cw,
                ch
            )
        )

    candidates.sort(
        reverse=True
    )

    crops = []

    for score, x, y, cw, ch in candidates[:8]:

        px = int(
            cw * 0.25
        )

        py = int(
            ch * 0.80
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
                f"[PLATE CANDIDATE] "
                f"{crop.shape[1]}x{crop.shape[0]}"
            )

            crops.append(
                crop
            )

    return crops


# =========================================================
# WHOLE PLATE OCR
# =========================================================

def whole_plate_ocr(plate):

    results = []

    gray = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

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

    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants = [
        ("gray", gray),
        ("enhanced", enhanced),
        ("binary", binary),
        (
            "inverted",
            cv2.bitwise_not(binary)
        ),
    ]

    for name, img in variants:

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
                    img,
                    config=config,
                    lang="eng",
                    timeout=3
                )

                text = clean(
                    text
                )

                if text:

                    print(
                        f"[WHOLE OCR] "
                        f"{name}/PSM{psm}: "
                        f"{text}"
                    )

                    results.append(
                        text
                    )

            except Exception:
                pass

    return results


# =========================================================
# PLATE FORMAT
# =========================================================

def valid_plate(text):

    text = clean(
        text
    )

    if re.fullmatch(
        r"\d{2}BH\d{4}[A-Z]{2}",
        text
    ):
        return True

    if len(text) < 7 or len(text) > 13:
        return False

    if text[:2] not in STATE_CODES:
        return False

    if re.fullmatch(
        r"[A-Z]{2}\d{1,3}[A-Z]{1,3}\d{1,4}",
        text
    ):
        return True

    if re.fullmatch(
        r"[A-Z]{2}\d{1,3}\d{4,5}",
        text
    ):
        return True

    return False


# =========================================================
# OCR CONFUSION CORRECTION
# =========================================================

def make_corrections(text):

    text = clean(
        text
    )

    candidates = {
        text
    }

    # State prefix corrections.
    prefix_map = {
        "0P": "UP",
        "OP": "UP",
        "VP": "UP",
        "U0": "UP",
        "0D": "OD",
        "OR": "OR",
        "0R": "OR",
        "1P": "UP",
    }

    if len(text) >= 2:

        prefix = text[:2]

        if prefix in prefix_map:

            candidates.add(
                prefix_map[prefix] +
                text[2:]
            )

    # Numeric section corrections.
    replacements = {
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

    if len(text) > 2:

        prefix = text[:2]
        rest = text[2:]

        converted = ""

        for char in rest:

            converted += replacements.get(
                char,
                char
            )

        candidates.add(
            prefix + converted
        )

    return list(
        candidates
    )


# =========================================================
# INDIVIDUAL CHARACTER PIPELINE
# =========================================================

def character_pipeline(plate):

    print(
        "[CHAR OCR] Starting character segmentation"
    )

    chars = segment_characters(
        plate
    )

    if len(chars) < 5:

        print(
            "[CHAR OCR] Not enough character boxes."
        )

        return []

    readings = []

    for index, char_img in enumerate(
        chars
    ):

        character = character_ocr(
            char_img
        )

        print(
            f"[CHAR {index + 1}] "
            f"{character or '?'}"
        )

        if character:

            readings.append(
                character
            )

        else:

            readings.append(
                "?"
            )

    raw = "".join(
        readings
    )

    print(
        f"[CHAR OCR RAW] {raw}"
    )

    return make_corrections(
        raw
    )


# =========================================================
# BUILD CANDIDATES
# =========================================================

def extract_whole_candidates(texts):

    results = []

    for text in texts:

        text = clean(
            text
        )

        # Direct
        if valid_plate(text):

            results.append(
                text
            )

        # Corrected
        for candidate in make_corrections(
            text
        ):

            if valid_plate(
                candidate
            ):

                results.append(
                    candidate
                )

        # Search inside OCR output.
        for state in STATE_CODES:

            matches = re.findall(
                state +
                r"[A-Z0-9]{5,11}",
                text
            )

            for match in matches:

                for candidate in make_corrections(
                    match
                ):

                    if valid_plate(
                        candidate
                    ):

                        results.append(
                            candidate
                        )

    return results


# =========================================================
# COMPLETE PLATE ANALYSIS
# =========================================================

def analyse_plate(plate, label):

    print(
        f"======================================"
    )

    print(
        f"[PLATE ANALYSIS] {label}"
    )

    results = []

    # Whole-line OCR
    whole_results = whole_plate_ocr(
        plate
    )

    results.extend(
        extract_whole_candidates(
            whole_results
        )
    )

    # Character OCR
    char_results = character_pipeline(
        plate
    )

    for candidate in char_results:

        if valid_plate(
            candidate
        ):

            results.append(
                candidate
            )

    print(
        f"[PLATE RESULTS] {results}"
    )

    return results


# =========================================================
# FINAL VOTE
# =========================================================

def choose_plate(results):

    if not results:

        return None

    counter = Counter(
        results
    )

    print(
        "[FINAL VOTES]",
        dict(counter)
    )

    ranked = counter.most_common()

    # Multiple independent detections
    if ranked[0][1] >= 2:

        print(
            f"[FINAL] {ranked[0][0]}"
        )

        return ranked[0][0]

    # Single valid strict plate
    if valid_plate(
        ranked[0][0]
    ):

        print(
            f"[FINAL SINGLE] "
            f"{ranked[0][0]}"
        )

        return ranked[0][0]

    return None


# =========================================================
# COMPLETE IMAGE ANALYSIS
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    all_results = []

    # =====================================================
    # 1. Treat whole image as plate
    # =====================================================

    print(
        "[STEP 1] Whole image as plate"
    )

    all_results.extend(
        analyse_plate(
            image,
            "WHOLE_IMAGE"
        )
    )

    # =====================================================
    # 2. Automatic plate detection
    # =====================================================

    crops = detect_plate_candidates(
        image
    )

    for index, crop in enumerate(
        crops
    ):

        all_results.extend(
            analyse_plate(
                crop,
                f"PLATE_{index + 1}"
            )
        )

    # =====================================================
    # 3. Center crop
    # =====================================================

    h, w = image.shape[:2]

    center = image[
        int(h * 0.15):
        int(h * 0.90),
        int(w * 0.03):
        int(w * 0.97)
    ]

    all_results.extend(
        analyse_plate(
            center,
            "CENTER"
        )
    )

    print(
        "======================================"
    )

    print(
        "[ALL RESULTS]",
        all_results
    )

    return choose_plate(
        all_results
    )


# =========================================================
# TELEGRAM
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE SCANNER\n\n"
        "📸 Number plate ki photo bhejo.\n\n"
        "Bot:\n"
        "🔢 Registration number read karega\n"
        "🇮🇳 State identify karega\n"
        "🏙️ Registration/RTO city lookup karega\n\n"
        "⚠️ City registration area hai, "
        "current location nahi."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Best result ke liye plate ka sharp "
        "close-up bhejo.\n\n"
        "Good lighting aur minimum reflection rakho."
    )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    status = await update.message.reply_text(
        "🔎 Plate scan ho rahi hai...\n\n"
        "Plate detection + character OCR + RTO lookup."
    )

    temp_path = None

    try:

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
            "[PHOTO] New image"
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
                "❌ Number plate read nahi ho payi.\n\n"
                "Is attempt mein OCR ko reliable "
                "registration number nahi mila.\n\n"
                "Render Logs mein `[CHAR OCR]`, "
                "`[CHAR 1]`, `[CHAR 2]` etc. dekho."
            )

            return

        info = lookup_rto(
            plate
        )

        message = (
            "✅ VEHICLE REGISTRATION FOUND\n\n"
            f"🔢 Number: {plate}\n"
            f"🇮🇳 State: {info['state']}\n"
        )

        if info.get("code"):

            message += (
                f"🏢 RTO Code: "
                f"{info['code']}\n"
            )

        message += (
            f"🏙️ Registration City/Area: "
            f"{info['city']}\n\n"
            "ℹ️ Yeh registration/RTO area hai, "
            "current vehicle location nahi."
        )

        await status.edit_text(
            message
        )

    except Exception as e:

        print(
            "[ERROR]",
            repr(e)
        )

        try:

            await status.edit_text(
                "❌ Image processing error."
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
            "BOT_TOKEN is missing."
        )

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
        "VEHICLE OCR BOT V3"
    )

    print(
        "CHARACTER LEVEL OCR: ENABLED"
    )

    print(
        "PLATE DETECTION: ENABLED"
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

    asyncio.run(
        main()
        )
