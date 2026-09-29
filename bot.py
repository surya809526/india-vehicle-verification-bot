import os
import re
import asyncio
import tempfile
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
# INDIAN STATE / UT CODES
# =========================================================

STATE_CODES = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
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
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
}

# =========================================================
# RTO REFERENCE
# =========================================================

RTO_CODES = {
    "UP14": "Ghaziabad",
    "UP15": "Meerut",
    "UP16": "Gautam Buddh Nagar / Noida",
    "UP20": "Bijnor",
    "UP21": "Moradabad",
    "UP22": "Rampur",
    "UP23": "Amroha",
    "UP25": "Bareilly",
    "UP27": "Shahjahanpur",
    "UP30": "Hardoi",
    "UP31": "Lakhimpur Kheri",
    "UP32": "Lucknow",
    "UP33": "Raebareli",
    "UP34": "Sitapur",
    "UP35": "Unnao",
    "UP36": "Barabanki",
    "UP37": "Fatehpur",
    "UP40": "Bahraich",
    "UP41": "Gonda",
    "UP42": "Ayodhya region",
    "UP43": "Sultanpur",
    "UP44": "Ambedkar Nagar",
    "UP47": "Gorakhpur",
    "UP50": "Azamgarh",
    "UP51": "Basti",
    "UP52": "Deoria",
    "UP53": "Maharajganj",
    "UP54": "Kushinagar",
    "UP55": "Siddharthnagar",
    "UP56": "Sant Kabir Nagar",
    "UP57": "Mau",
    "UP58": "Ballia",
    "UP60": "Jaunpur",
    "UP61": "Varanasi",
    "UP62": "Mirzapur",
    "UP63": "Sonbhadra",
    "UP64": "Ghazipur",
    "UP65": "Chandauli",
    "UP66": "Bhadohi",
    "UP70": "Prayagraj",
    "UP72": "Kaushambi",
    "UP75": "Jhansi",
    "UP76": "Lalitpur",
    "UP77": "Jalaun",
    "UP78": "Kanpur Nagar",
    "UP79": "Kanpur Dehat",
    "UP80": "Agra",
    "UP81": "Aligarh",
    "UP82": "Etah",
    "UP83": "Firozabad",
    "UP84": "Mainpuri",
    "UP85": "Mathura",
    "UP86": "Hathras",
    "UP87": "Kasganj",

    "DL01": "Delhi",
    "DL02": "Delhi",
    "DL03": "Delhi",
    "DL04": "Delhi",
    "DL05": "Delhi",
    "DL06": "Delhi",
    "DL07": "Delhi",
    "DL08": "Delhi",
    "DL09": "Delhi",
    "DL10": "Delhi",
    "DL11": "Delhi",
    "DL12": "Delhi",
    "DL13": "Delhi",
    "DL14": "Delhi",
    "DL15": "Delhi",
    "DL16": "Delhi",
    "DL17": "Delhi",

    "MH01": "Mumbai",
    "MH02": "Mumbai",
    "MH03": "Mumbai",
    "MH04": "Thane",
    "MH05": "Kalyan",
    "MH09": "Kolhapur",
    "MH10": "Sangli",
    "MH11": "Satara",
    "MH12": "Pune",
    "MH13": "Solapur",
    "MH14": "Pimpri-Chinchwad",
    "MH15": "Nashik",
    "MH16": "Ahmednagar",
    "MH19": "Jalgaon",
    "MH20": "Chhatrapati Sambhajinagar",
    "MH27": "Amravati",
    "MH30": "Akola",
    "MH31": "Nagpur",
}


# =========================================================
# RENDER HEALTH SERVER
# =========================================================

async def health(request):
    return web.Response(
        text="Vehicle OCR Bot is running."
    )


async def start_health_server():

    server = web.Application()

    server.router.add_get("/", health)
    server.router.add_get("/health", health)

    runner = web.AppRunner(server)

    await runner.setup()

    site = web.TCPSite(
        runner,
        "0.0.0.0",
        PORT
    )

    await site.start()

    print(
        f"[SERVER] HTTP server running on port {PORT}"
    )

    return runner


# =========================================================
# IMAGE LOAD
# =========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            "Could not load image."
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] Original size: {w}x{h}"
    )

    # Reasonable processing size
    max_width = 1600

    if w > max_width:

        ratio = max_width / float(w)

        image = cv2.resize(
            image,
            (
                max_width,
                max(
                    100,
                    int(h * ratio)
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    h, w = image.shape[:2]

    print(
        f"[IMAGE] Processing size: {w}x{h}"
    )

    return image


# =========================================================
# CLEAN OCR TEXT
# =========================================================

def clean_text(text):

    if not text:
        return ""

    text = text.upper()

    replacements = {
        " ": "",
        "\n": "",
        "\r": "",
        "\t": "",
        "-": "",
        "_": "",
        ".": "",
        ":": "",
        "/": "",
        "\\": "",
        "|": "I",
        "[": "",
        "]": "",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# COMMON OCR CORRECTIONS
# =========================================================

def correct_common_errors(text):

    text = clean_text(text)

    if len(text) < 2:
        return text

    # OCR commonly confuses these in the FIRST 2 letters
    prefix_fixes = {
        "0P": "UP",
        "OP": "UP",
        "0D": "OD",
        "OD": "OD",
        "OR": "OR",
        "0R": "OR",
        "1P": "UP",
        "HP": "HP",
        "MP": "MP",
        "MH": "MH",
        "RJ": "RJ",
        "HR": "HR",
        "PB": "PB",
        "BR": "BR",
        "GJ": "GJ",
        "KA": "KA",
        "KL": "KL",
        "TN": "TN",
        "TS": "TS",
        "AP": "AP",
        "CG": "CG",
        "UK": "UK",
        "WB": "WB",
        "AS": "AS",
        "JH": "JH",
        "DL": "DL",
    }

    prefix = text[:2]

    if prefix in prefix_fixes:

        text = (
            prefix_fixes[prefix]
            + text[2:]
        )

    return text


# =========================================================
# OCR IMAGE PREPARATION
# =========================================================

def prepare_ocr_images(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # Resize to OCR-friendly size
    target_width = 1200

    if w < 1200:

        scale = 1200 / float(w)

        new_w = 1200
        new_h = max(
            100,
            int(h * scale)
        )

        gray = cv2.resize(
            gray,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )

    elif w > 1200:

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
            interpolation=cv2.INTER_AREA
        )

    # Contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # OTSU
    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    return [
        ("gray", gray),
        ("enhanced", enhanced),
        ("binary", binary),
        ("adaptive", adaptive),
    ]


# =========================================================
# DIRECT OCR
# =========================================================

def run_tesseract(image, psm):

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
            timeout=4
        )

        return text or ""

    except RuntimeError as e:

        print(
            f"[OCR TIMEOUT] PSM {psm}: {e}"
        )

        return ""

    except Exception as e:

        print(
            f"[OCR ERROR] PSM {psm}: {e}"
        )

        return ""


# =========================================================
# EXTRACT PLATE CANDIDATES
# =========================================================

def extract_candidates(text):

    original = clean_text(
        text
    )

    corrected = correct_common_errors(
        original
    )

    candidates = []

    if not corrected:
        return candidates

    # -----------------------------------------------------
    # Normal Indian format
    # Example: UP32AB1234
    # -----------------------------------------------------

    patterns = [

        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",

        r"[A-Z]{2}\d{1,2}\d{4,5}",

    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            corrected
        )

        candidates.extend(
            matches
        )

    # -----------------------------------------------------
    # Search inside text
    # -----------------------------------------------------

    for state in STATE_CODES.keys():

        pattern = (
            state +
            r"\d{1,2}[A-Z0-9]{3,9}"
        )

        matches = re.findall(
            pattern,
            corrected
        )

        candidates.extend(
            matches
        )

    # -----------------------------------------------------
    # Try OCR confusion around numbers
    # -----------------------------------------------------

    flexible = corrected

    # These substitutions are ONLY used
    # for candidate generation.
    # We don't automatically trust them.

    flexible = flexible.replace(
        "Z",
        "2"
    )

    flexible = flexible.replace(
        "O",
        "0"
    )

    for state in STATE_CODES.keys():

        pattern = (
            state +
            r"\d{1,2}[A-Z0-9]{3,9}"
        )

        matches = re.findall(
            pattern,
            flexible
        )

        candidates.extend(
            matches
        )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    valid = []

    for candidate in candidates:

        candidate = clean_text(
            candidate
        )

        if not 7 <= len(candidate) <= 13:
            continue

        if candidate[:2] not in STATE_CODES:
            continue

        # Must contain a digit
        if not re.search(
            r"\d",
            candidate
        ):
            continue

        valid.append(
            candidate
        )

    return valid


# =========================================================
# PLATE VOTING
# =========================================================

def choose_plate(all_candidates):

    if not all_candidates:

        return None

    counter = Counter(
        all_candidates
    )

    print(
        "[OCR CANDIDATES]",
        dict(counter)
    )

    # Highest vote
    best_plate, best_count = (
        counter.most_common(1)[0]
    )

    # Require at least 2 independent OCR hits
    if best_count >= 2:

        print(
            f"[PLATE] Reliable candidate: "
            f"{best_plate} "
            f"(votes={best_count})"
        )

        return best_plate

    # One OCR hit can still be useful,
    # but only if it looks strongly like a plate.
    if best_count == 1:

        if re.match(
            r"^[A-Z]{2}\d{1,2}",
            best_plate
        ):

            print(
                f"[PLATE] Single strong candidate: "
                f"{best_plate}"
            )

            return best_plate

    return None


# =========================================================
# DIRECT CLOSE-UP OCR
# =========================================================

def analyse_closeup(image):

    print(
        "[MODE] Direct close-up OCR"
    )

    ocr_images = prepare_ocr_images(
        image
    )

    all_candidates = []

    for name, prepared in ocr_images:

        print(
            f"[OCR] Processing: {name}"
        )

        # PSM 7 = single line
        text7 = run_tesseract(
            prepared,
            7
        )

        print(
            f"[RAW OCR][{name}][PSM7] "
            f"{repr(text7)}"
        )

        candidates7 = extract_candidates(
            text7
        )

        all_candidates.extend(
            candidates7
        )

        # If found strong candidate, still do
        # one additional PSM 8 check.
        text8 = run_tesseract(
            prepared,
            8
        )

        print(
            f"[RAW OCR][{name}][PSM8] "
            f"{repr(text8)}"
        )

        candidates8 = extract_candidates(
            text8
        )

        all_candidates.extend(
            candidates8
        )

    return choose_plate(
        all_candidates
    )


# =========================================================
# FULL VEHICLE FALLBACK
# =========================================================

def find_plate_regions(image):

    regions = []

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

    possible = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 80 or ch < 15:
            continue

        ratio = cw / float(ch)

        if 2.0 <= ratio <= 8.0:

            possible.append(
                (
                    cw * ch,
                    x,
                    y,
                    cw,
                    ch
                )
            )

    possible.sort(
        reverse=True
    )

    for _, x, y, cw, ch in possible[:3]:

        pad_x = int(cw * 0.20)
        pad_y = int(ch * 1.0)

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

            regions.append(
                crop
            )

    return regions


# =========================================================
# COMPLETE ANALYSIS
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    # =====================================================
    # FIRST: Treat image as a close-up plate
    # =====================================================

    plate = analyse_closeup(
        image
    )

    if plate:

        return make_result(
            plate
        )

    # =====================================================
    # SECOND: Try detected plate regions
    # =====================================================

    print(
        "[MODE] Region fallback"
    )

    regions = find_plate_regions(
        image
    )

    print(
        f"[REGIONS] Found {len(regions)} possible regions"
    )

    for index, region in enumerate(
        regions
    ):

        print(
            f"[REGION] OCR {index + 1}"
        )

        plate = analyse_closeup(
            region
        )

        if plate:

            return make_result(
                plate
            )

    print(
        "[RESULT] No reliable plate detected."
    )

    return None


# =========================================================
# RESULT
# =========================================================

def make_result(plate):

    state_code = plate[:2]

    state_name = STATE_CODES.get(
        state_code
    )

    rto_code = None
    rto_name = None

    match = re.match(
        r"^([A-Z]{2})(\d{1,2})",
        plate
    )

    if match:

        number = match.group(2)

        rto_code = (
            state_code +
            number.zfill(2)
        )

        rto_name = RTO_CODES.get(
            rto_code
        )

    return {
        "plate": plate,
        "state": state_name,
        "rto_code": rto_code,
        "rto_name": rto_name,
    }


# =========================================================
# TELEGRAM /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE OCR BOT\n\n"
        "📸 Number plate ki photo bhejo.\n\n"
        "Main:\n"
        "🔢 Registration number read karunga\n"
        "🇮🇳 State/UT identify karunga\n"
        "🏢 Available RTO reference check karunga\n\n"
        "⚠️ Bot unknown number guess nahi karega."
    )


# =========================================================
# TELEGRAM /HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Best result ke liye:\n\n"
        "• Number plate close-up bhejo\n"
        "• Letters/numbers sharp hone chahiye\n"
        "• Reflection kam ho\n"
        "• Plate seedhi ho\n"
        "• Photo blur na ho\n\n"
        "Full vehicle photo bhi try kar sakte ho."
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
        "Direct OCR + plate verification chal raha hai."
    )

    temp_path = None

    try:

        # Highest resolution Telegram version
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
            "===================================="
        )

        print(
            "[PHOTO] New image received"
        )

        # OCR is CPU-heavy
        result = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        print(
            "===================================="
        )

        if not result:

            await status.edit_text(
                "❌ Number plate reliably read nahi ho payi.\n\n"
                "Maine OCR attempts kiye hain, "
                "lekin reliable registration number nahi mila.\n\n"
                "📸 Agar plate close-up hai, photo dobara "
                "original quality mein bhejo.\n\n"
                "⚠️ Bot guessed number nahi dega."
            )

            return

        response = (
            "✅ NUMBER PLATE FOUND\n\n"
            f"🔢 Registration: {result['plate']}\n"
            f"🇮🇳 State/UT: {result['state']}\n"
        )

        if result["rto_code"]:

            response += (
                f"🏢 RTO Code: "
                f"{result['rto_code']}\n"
            )

        if result["rto_name"]:

            response += (
                f"📍 Known registration area: "
                f"{result['rto_name']}\n"
            )

        else:

            response += (
                "📍 Registration area: "
                "Not available in reference data\n"
            )

        response += (
            "\n"
            "⚠️ Registration area current vehicle "
            "location nahi hoti.\n"
            "⚠️ Owner details provide nahi ki jaati."
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
                "❌ Image process nahi ho saki.\n\n"
                "Please dobara photo bhejo."
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
            "BOT_TOKEN is missing."
        )

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
        "Starting Telegram polling..."
    )

    await application.initialize()

    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

    print(
        "===================================="
    )

    print(
        "BOT IS RUNNING"
    )

    print(
        "TELEGRAM POLLING: ACTIVE"
    )

    print(
        "OCR DEBUG MODE: ACTIVE"
    )

    print(
        "===================================="
    )

    try:

        await asyncio.Event().wait()

    finally:

        print(
            "Stopping bot..."
        )

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
