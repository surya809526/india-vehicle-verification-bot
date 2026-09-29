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

# =========================================================
# ALL INDIA RTO DATABASE
# =========================================================

RTO_API = "https://trafficchallan.com/api/rto-codes.json"

RTO_DB = {}


# =========================================================
# STATE FALLBACK
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
    "DD": "Dadra and Nagar Haveli and Daman and Diu",
}


# =========================================================
# LOAD RTO DATABASE
# =========================================================

def load_rto_database():

    global RTO_DB

    print("[RTO] Downloading all-India RTO database...")

    try:

        request = urllib.request.Request(
            RTO_API,
            headers={
                "User-Agent": "Vehicle-OCR-Bot/1.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        count = 0

        for state in data.get(
            "states",
            []
        ):

            state_name = state.get(
                "state_name",
                ""
            )

            for item in state.get(
                "codes",
                []
            ):

                code = (
                    item.get(
                        "code",
                        ""
                    )
                    .upper()
                    .replace(
                        "-",
                        ""
                    )
                    .replace(
                        " ",
                        ""
                    )
                )

                office = item.get(
                    "office",
                    ""
                )

                if code:

                    RTO_DB[code] = {
                        "state": state_name,
                        "city": office
                    }

                    count += 1

        print(
            f"[RTO] Loaded {count} RTO codes."
        )

        if count == 0:

            print(
                "[RTO] Database empty."
            )

    except Exception as e:

        print(
            "[RTO ERROR]",
            repr(e)
        )

        print(
            "[RTO] Using fallback state detection."
        )


# =========================================================
# RTO LOOKUP
# =========================================================

def lookup_rto(plate):

    plate = plate.upper()

    # -----------------------------------------------------
    # BH SERIES
    # Example: 22BH1234AA
    # -----------------------------------------------------

    if re.match(
        r"^\d{2}BH\d{4}[A-Z]{2}$",
        plate
    ):

        return {
            "state": "Bharat Series",
            "city": "State/RTO city is not encoded",
            "rto_code": "BH",
            "confidence": "not state-specific"
        }

    # -----------------------------------------------------
    # Standard state prefix
    # -----------------------------------------------------

    state_code = plate[:2]

    state_name = STATE_CODES.get(
        state_code,
        "Unknown"
    )

    # -----------------------------------------------------
    # Extract RTO code
    # Usually state + 1-3 digits
    # -----------------------------------------------------

    match = re.match(
        r"^([A-Z]{2})(\d{1,3})",
        plate
    )

    if not match:

        return {
            "state": state_name,
            "city": "RTO city could not be determined",
            "rto_code": None,
            "confidence": "low"
        }

    digits = match.group(2)

    possible_codes = []

    # Try 3 digit first
    if len(digits) >= 3:

        possible_codes.append(
            state_code +
            digits[:3]
        )

    # Then 2 digit
    if len(digits) >= 2:

        possible_codes.append(
            state_code +
            digits[:2].zfill(2)
        )

    # Then 1 digit
    if len(digits) >= 1:

        possible_codes.append(
            state_code +
            digits[:1].zfill(2)
        )

    for code in possible_codes:

        if code in RTO_DB:

            info = RTO_DB[code]

            return {
                "state": info.get(
                    "state",
                    state_name
                ),
                "city": info.get(
                    "city",
                    "Unknown"
                ),
                "rto_code": code,
                "confidence": "database match"
            }

    # -----------------------------------------------------
    # No database match
    # -----------------------------------------------------

    return {
        "state": state_name,
        "city": "RTO code not found in database",
        "rto_code": (
            state_code +
            digits.zfill(2)
        ),
        "confidence": "state only"
    }


# =========================================================
# HEALTH SERVER
# =========================================================

async def health(request):

    return web.Response(
        text="Vehicle OCR Bot is running."
    )


async def start_health_server():

    server = web.Application()

    server.router.add_get(
        "/",
        health
    )

    server.router.add_get(
        "/health",
        health
    )

    runner = web.AppRunner(
        server
    )

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        PORT
    )

    await site.start()

    print(
        f"[SERVER] Running on port {PORT}"
    )

    return runner


# =========================================================
# IMAGE LOAD
# =========================================================

def load_image(path):

    image = cv2.imread(
        path
    )

    if image is None:

        raise ValueError(
            "Image could not be loaded."
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] Original: {w}x{h}"
    )

    if w > 1600:

        ratio = 1600 / float(w)

        image = cv2.resize(
            image,
            (
                1600,
                int(h * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] Processing: {w}x{h}"
    )

    return image


# =========================================================
# CLEAN OCR
# =========================================================

def clean_text(text):

    if not text:

        return ""

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# OCR IMAGE VARIANTS
# =========================================================

def prepare_ocr_images(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    target_width = 1200

    if w != target_width:

        scale = target_width / float(w)

        gray = cv2.resize(
            gray,
            (
                target_width,
                max(
                    100,
                    int(h * scale)
                )
            ),
            interpolation=(
                cv2.INTER_CUBIC
                if w < target_width
                else cv2.INTER_AREA
            )
        )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
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

    return [
        ("gray", gray),
        ("enhanced", enhanced),
        ("binary", binary),
    ]


# =========================================================
# TESSERACT
# =========================================================

def run_ocr(image, psm):

    config = (
        f"--oem 3 --psm {psm} "
        "-c tessedit_char_whitelist="
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    try:

        return pytesseract.image_to_string(
            image,
            config=config,
            lang="eng",
            timeout=4
        ) or ""

    except Exception as e:

        print(
            f"[OCR ERROR] PSM={psm}: {repr(e)}"
        )

        return ""


# =========================================================
# OCR CONFUSION NORMALIZATION
# =========================================================

def normalize_plate(text):

    text = clean_text(
        text
    )

    if len(text) < 4:

        return []

    candidates = []

    # Original
    candidates.append(
        text
    )

    # Common OCR corrections
    variants = {
        "Z": "2",
        "O": "0",
        "Q": "0",
        "I": "1",
        "L": "1",
        "S": "5",
        "G": "6",
        "T": "7",
        "B": "8",
    }

    corrected = text

    # Don't blindly replace first two characters.
    # Only apply these to the numeric/series area.
    if len(text) > 2:

        prefix = text[:2]

        rest = text[2:]

        for old, new in variants.items():

            rest = rest.replace(
                old,
                new
            )

        corrected = (
            prefix +
            rest
        )

        candidates.append(
            corrected
        )

    # Fix common state-prefix OCR mistakes
    prefix_map = {
        "0P": "UP",
        "OP": "UP",
        "VP": "UP",
        "U0": "UP",
        "MP": "MP",
        "MH": "MH",
        "HP": "HP",
        "HR": "HR",
        "PB": "PB",
        "RJ": "RJ",
        "DL": "DL",
        "KA": "KA",
        "KL": "KL",
        "TN": "TN",
        "TS": "TS",
        "TG": "TG",
        "GJ": "GJ",
        "BR": "BR",
        "WB": "WB",
        "AS": "AS",
        "AP": "AP",
        "CG": "CG",
        "UK": "UK",
        "OD": "OD",
        "JH": "JH",
        "GA": "GA",
    }

    if len(text) >= 2:

        prefix = text[:2]

        if prefix in prefix_map:

            fixed = (
                prefix_map[prefix] +
                text[2:]
            )

            candidates.append(
                fixed
            )

    return list(
        dict.fromkeys(
            candidates
        )
    )


# =========================================================
# PLATE PATTERN
# =========================================================

def is_valid_plate(text):

    text = clean_text(
        text
    )

    # Bharat Series
    if re.match(
        r"^\d{2}BH\d{4}[A-Z]{2}$",
        text
    ):

        return True

    # Standard Indian registration
    if re.match(
        r"^[A-Z]{2}\d{1,3}[A-Z]{1,3}\d{1,4}$",
        text
    ):

        return True

    if re.match(
        r"^[A-Z]{2}\d{1,3}\d{4,5}$",
        text
    ):

        return True

    return False


# =========================================================
# CANDIDATE EXTRACTION
# =========================================================

def extract_candidates(text):

    if not text:

        return []

    raw = text.upper()

    # Individual OCR tokens
    tokens = re.findall(
        r"[A-Z0-9]+",
        raw
    )

    candidates = []

    # Each token
    for token in tokens:

        for variant in normalize_plate(
            token
        ):

            if is_valid_plate(
                variant
            ):

                candidates.append(
                    variant
                )

    # Entire OCR output joined
    joined = "".join(
        tokens
    )

    for variant in normalize_plate(
        joined
    ):

        if is_valid_plate(
            variant
        ):

            candidates.append(
                variant
            )

    # Search substrings
    for state in STATE_CODES:

        pattern = (
            state +
            r"[A-Z0-9]{5,11}"
        )

        for match in re.findall(
            pattern,
            joined
        ):

            for variant in normalize_plate(
                match
            ):

                if is_valid_plate(
                    variant
                ):

                    candidates.append(
                        variant
                    )

    return candidates


# =========================================================
# OCR ANALYSIS
# =========================================================

def analyse_closeup(image):

    print(
        "[OCR] Starting direct close-up OCR"
    )

    variants = prepare_ocr_images(
        image
    )

    all_candidates = []

    for name, prepared in variants:

        print(
            f"[OCR] Variant: {name}"
        )

        for psm in (7, 8):

            text = run_ocr(
                prepared,
                psm
            )

            print(
                f"[RAW OCR][{name}][PSM{psm}] "
                f"{repr(text)}"
            )

            candidates = extract_candidates(
                text
            )

            print(
                f"[CANDIDATES][{name}][PSM{psm}] "
                f"{candidates}"
            )

            all_candidates.extend(
                candidates
            )

    if not all_candidates:

        print(
            "[OCR] No valid plate candidate."
        )

        return None

    counter = Counter(
        all_candidates
    )

    print(
        "[OCR VOTES]",
        dict(counter)
    )

    best, votes = counter.most_common(
        1
    )[0]

    print(
        f"[OCR BEST] {best} "
        f"votes={votes}"
    )

    return best


# =========================================================
# REGION FALLBACK
# =========================================================

def find_plate_regions(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        80,
        180
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 5)
    )

    closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape

    results = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 80 or ch < 15:
            continue

        ratio = cw / float(ch)

        if 2.0 <= ratio <= 8.0:

            area = cw * ch

            results.append(
                (
                    area,
                    x,
                    y,
                    cw,
                    ch
                )
            )

    results.sort(
        reverse=True
    )

    regions = []

    for _, x, y, cw, ch in results[:3]:

        px = int(cw * 0.2)
        py = int(ch * 1.0)

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

            regions.append(
                crop
            )

    return regions


# =========================================================
# COMPLETE VEHICLE ANALYSIS
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    # -----------------------------------------------------
    # FIRST: Whole image OCR
    # -----------------------------------------------------

    plate = analyse_closeup(
        image
    )

    if plate:

        return plate

    # -----------------------------------------------------
    # SECOND: Detect possible plate regions
    # -----------------------------------------------------

    print(
        "[FALLBACK] Searching plate regions..."
    )

    regions = find_plate_regions(
        image
    )

    print(
        f"[FALLBACK] Regions found: "
        f"{len(regions)}"
    )

    for index, region in enumerate(
        regions
    ):

        print(
            f"[FALLBACK] OCR region {index + 1}"
        )

        plate = analyse_closeup(
            region
        )

        if plate:

            return plate

    return None


# =========================================================
# TELEGRAM START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE CITY FINDER\n\n"
        "📸 Number plate ki photo bhejo.\n\n"
        "Bot try karega:\n"
        "🔢 Registration number\n"
        "🇮🇳 State/UT\n"
        "🏙️ Registration/RTO city\n"
        "🏢 RTO code\n\n"
        "⚠️ City ka matlab registration/RTO area hai, "
        "current vehicle location nahi."
    )


# =========================================================
# HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Best result:\n\n"
        "• Number plate ka clear close-up\n"
        "• Good lighting\n"
        "• Minimum blur\n"
        "• Plate seedhi ho\n\n"
        "Example:\n"
        "UP32AB1234 → Lucknow, Uttar Pradesh\n"
        "UP78AB1234 → Kanpur Nagar, Uttar Pradesh\n\n"
        "BH-series mein city/state plate se directly "
        "determine nahi hoti."
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    status = await update.message.reply_text(
        "🔎 Number plate scan ho rahi hai...\n\n"
        "OCR + All-India RTO lookup."
    )

    temp_path = None

    try:

        photo = update.message.photo[-1]

        telegram_file = await context.bot.get_file(
            photo.file_id
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
            "========================================"
        )

        print(
            "[PHOTO] New vehicle image received"
        )

        plate = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        print(
            "========================================"
        )

        if not plate:

            await status.edit_text(
                "❌ Number plate reliably read nahi ho payi.\n\n"
                "OCR ne reliable registration number detect "
                "nahi kiya.\n\n"
                "📸 Clear close-up/original-quality plate photo "
                "try karo.\n\n"
                "Render logs mein `[RAW OCR]` lines available hain."
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

        if info.get(
            "rto_code"
        ):

            response += (
                f"🏢 RTO Code: "
                f"{info['rto_code']}\n"
            )

        if info.get(
            "city"
        ):

            response += (
                f"🏙️ Registration City/Area: "
                f"{info['city']}\n"
            )

        response += (
            "\n"
            "ℹ️ Yeh registration/RTO area hai.\n"
            "📍 Isse vehicle ki current location "
            "determine nahi hoti.\n"
            "👤 Owner details is bot se provide nahi hoti."
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
                "❌ Photo process nahi ho saki.\n"
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
# TEXT HANDLER
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

    # Load RTO data before starting bot
    load_rto_database()

    health_runner = await start_health_server()

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
        "========================================"
    )

    print(
        "VEHICLE OCR BOT STARTING"
    )

    print(
        "RTO DATABASE: LOADED"
    )

    print(
        "TELEGRAM POLLING: STARTING"
    )

    print(
        "========================================"
    )

    await application.initialize()

    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

    print(
        "BOT IS RUNNING"
    )

    try:

        await asyncio.Event().wait()

    finally:

        await application.updater.stop()

        await application.stop()

        await application.shutdown()

        await health_runner.cleanup()


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Bot stopped."
            )
