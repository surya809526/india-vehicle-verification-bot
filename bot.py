import os
import re
import asyncio
import tempfile

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
# KNOWN RTO REFERENCE
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
        f"HTTP health server running on port {PORT}"
    )

    return runner


# =========================================================
# IMAGE LOAD
# =========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError("Image could not be loaded.")

    # Limit extremely large photos
    height, width = image.shape[:2]

    max_width = 1800

    if width > max_width:

        ratio = max_width / width

        image = cv2.resize(
            image,
            (
                int(width * ratio),
                int(height * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# CREATE OCR VARIANTS
# =========================================================

def create_variants(image):

    variants = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Normal grayscale
    variants.append(gray)

    # Upscale
    up = cv2.resize(
        gray,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    variants.append(up)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(up)

    variants.append(enhanced)

    # OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(otsu)

    # Inverted OTSU
    _, inv = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV +
        cv2.THRESH_OTSU
    )

    variants.append(inv)

    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9
    )

    variants.append(adaptive)

    # Sharpen
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    sharp = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    variants.append(sharp)

    return variants


# =========================================================
# FIND POSSIBLE PLATE REGIONS
# =========================================================

def find_plate_regions(image):

    regions = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Edge detection
    edges = cv2.Canny(
        gray,
        80,
        200
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 5)
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

    height, width = gray.shape

    image_area = width * height

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = w * h

        if area < image_area * 0.0005:
            continue

        if w < 80 or h < 15:
            continue

        ratio = w / float(h)

        # Indian plate usually wider than tall
        if 2.0 <= ratio <= 8.5:

            crop = image[
                max(0, y - int(h * 0.5)):
                min(height, y + h + int(h * 0.5)),
                max(0, x - int(w * 0.08)):
                min(width, x + w + int(w * 0.08))
            ]

            if crop.size > 0:

                regions.append(crop)

    # Add central/full image as fallback
    regions.append(image)

    return regions[:30]


# =========================================================
# NORMALIZE OCR TEXT
# =========================================================

def normalize_text(text):

    text = text.upper()

    text = text.replace(
        "\n",
        ""
    )

    text = text.replace(
        "\r",
        ""
    )

    text = text.replace(
        " ",
        ""
    )

    # Common OCR symbols
    replacements = {
        "-": "",
        "_": "",
        ".": "",
        ":": "",
        "/": "",
        "\\": "",
        "|": "I",
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
# OCR
# =========================================================

def perform_ocr(image):

    outputs = []

    variants = create_variants(
        image
    )

    # Different layouts
    psm_modes = [
        6,
        7,
        11,
        13
    ]

    for variant in variants:

        for psm in psm_modes:

            try:

                text = pytesseract.image_to_string(
                    variant,
                    config=(
                        f"--oem 3 "
                        f"--psm {psm}"
                    ),
                    lang="eng",
                    timeout=8
                )

                if text:

                    outputs.append(
                        normalize_text(text)
                    )

            except Exception as e:

                print(
                    "OCR attempt failed:",
                    str(e)
                )

    return outputs


# =========================================================
# CORRECT COMMON OCR MISTAKES
# =========================================================

def fix_state_prefix(text):

    if len(text) < 2:
        return text

    first_two = text[:2]

    corrections = {
        "0P": "OP",
        "0D": "OD",
        "0R": "OR",
        "1P": "IP",
        "UP": "UP",
        "MP": "MP",
        "MH": "MH",
        "DL": "DL",
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
        "HP": "HP",
        "WB": "WB",
        "OD": "OD",
        "AS": "AS",
        "JH": "JH",
    }

    return corrections.get(
        first_two,
        first_two
    ) + text[2:]


# =========================================================
# EXTRACT PLATE
# =========================================================

def extract_plate(outputs):

    candidates = []

    for raw in outputs:

        text = fix_state_prefix(
            raw
        )

        # Standard format:
        # UP32AB1234
        matches = re.findall(
            r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",
            text
        )

        candidates.extend(
            matches
        )

        # Some plates may have no series letters
        matches = re.findall(
            r"[A-Z]{2}\d{1,2}\d{4,5}",
            text
        )

        candidates.extend(
            matches
        )

        # Try known state prefixes
        for state in STATE_CODES:

            if text.startswith(state):

                tail = text[2:]

                if 5 <= len(tail) <= 11:

                    possible = state + tail

                    if re.match(
                        r"^[A-Z]{2}\d",
                        possible
                    ):

                        candidates.append(
                            possible
                        )

    # Clean
    cleaned = []

    for candidate in candidates:

        candidate = normalize_text(
            candidate
        )

        if not (
            7 <= len(candidate) <= 13
        ):
            continue

        if candidate[:2] not in STATE_CODES:
            continue

        if not re.search(
            r"\d",
            candidate
        ):
            continue

        cleaned.append(
            candidate
        )

    if not cleaned:
        return None

    # Frequency voting
    counts = {}

    for candidate in cleaned:

        counts[candidate] = (
            counts.get(candidate, 0) + 1
        )

    best = sorted(
        counts.items(),
        key=lambda x: (
            x[1],
            len(x[0])
        ),
        reverse=True
    )

    return best[0][0]


# =========================================================
# COMPLETE IMAGE ANALYSIS
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    print(
        "Image size:",
        image.shape
    )

    all_outputs = []

    # First: possible plate regions
    regions = find_plate_regions(
        image
    )

    print(
        "Candidate regions:",
        len(regions)
    )

    for index, region in enumerate(
        regions
    ):

        print(
            f"OCR region {index + 1}"
        )

        outputs = perform_ocr(
            region
        )

        all_outputs.extend(
            outputs
        )

        plate = extract_plate(
            all_outputs
        )

        if plate:

            print(
                "PLATE FOUND:",
                plate
            )

            return build_result(
                plate
            )

    # Final full-image OCR
    outputs = perform_ocr(
        image
    )

    all_outputs.extend(
        outputs
    )

    plate = extract_plate(
        all_outputs
    )

    if plate:

        print(
            "PLATE FOUND:",
            plate
        )

        return build_result(
            plate
        )

    print(
        "No reliable plate found."
    )

    return None


# =========================================================
# RESULT
# =========================================================

def build_result(plate):

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

        rto_code = (
            state_code +
            match.group(2).zfill(2)
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
# /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE OCR BOT\n\n"
        "📸 Number plate ya vehicle ki clear photo "
        "bhejo.\n\n"
        "🔎 Main image se registration number read "
        "karne ki koshish karunga.\n\n"
        "🇮🇳 State/UT identify hoga.\n"
        "🏢 Known RTO reference bhi check hoga.\n\n"
        "⚠️ Unknown information guess nahi ki jayegi.\n"
        "⚠️ Owner details ya current vehicle location "
        "provide nahi ki jaati."
    )


# =========================================================
# /HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 BEST RESULT KE LIYE\n\n"
        "• Number plate ko close-up mein lo\n"
        "• Photo sharp rakho\n"
        "• Plate par light reflection kam ho\n"
        "• Plate tedhi ho to seedhi photo lo\n"
        "• Full vehicle photo bhi bhej sakte ho"
    )


# =========================================================
# PHOTO
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    status = await update.message.reply_text(
        "🔎 Image scan ho rahi hai...\n\n"
        "Plate locate + OCR chal raha hai."
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
            "Downloaded:",
            temp_path
        )

        result = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        if not result:

            await status.edit_text(
                "❌ Number plate scan nahi ho paya.\n\n"
                "📸 Please number plate ki close-up "
                "clear photo bhejo.\n\n"
                "Tip: Plate frame mein bada hona chahiye "
                "aur text readable hona chahiye."
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
            "\n⚠️ Registration area current vehicle "
            "location nahi hoti.\n"
            "⚠️ Owner information provide nahi ki jaati."
        )

        await status.edit_text(
            response
        )

    except Exception as e:

        print(
            "PHOTO ERROR:",
            repr(e)
        )

        try:

            await status.edit_text(
                "❌ Image process karte waqt error aaya.\n"
                "Please dobara clear photo bhejo."
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
        "Starting Telegram bot polling..."
    )

    await application.initialize()

    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

    print(
        "==================================="
    )

    print(
        "BOT IS RUNNING"
    )

    print(
        "Telegram polling: ACTIVE"
    )

    print(
        "OCR ENGINE: ACTIVE"
    )

    print(
        "==================================="
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


if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Bot stopped."
)
