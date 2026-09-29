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
# STATE CODES
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
        f"HTTP server running on port {PORT}"
    )

    return runner


# =========================================================
# LOAD IMAGE
# =========================================================

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise ValueError(
            "Unable to load image."
        )

    h, w = image.shape[:2]

    # Do not allow extremely large processing
    max_width = 1400

    if w > max_width:

        ratio = max_width / float(w)

        image = cv2.resize(
            image,
            (
                max_width,
                int(h * ratio)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================================================
# NORMALIZE OCR
# =========================================================

def normalize_text(text):

    text = text.upper()

    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# =========================================================
# OCR
# =========================================================

def ocr_image(image):

    results = []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    h, w = gray.shape

    # Keep OCR image reasonably sized
    if w > 1000:

        ratio = 1000 / float(w)

        gray = cv2.resize(
            gray,
            (
                1000,
                max(100, int(h * ratio))
            ),
            interpolation=cv2.INTER_AREA
        )

    # Upscale only once
    enlarged = cv2.resize(
        gray,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    # Contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        enlarged
    )

    # Threshold
    _, binary = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    images = [
        enhanced,
        binary
    ]

    # Only 2 PSM modes
    for img in images:

        for psm in (7, 8):

            try:

                text = pytesseract.image_to_string(
                    img,
                    config=(
                        f"--oem 3 --psm {psm} "
                        "-c tessedit_char_whitelist="
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    ),
                    lang="eng",
                    timeout=3
                )

                text = normalize_text(
                    text
                )

                if text:

                    print(
                        "OCR:",
                        text
                    )

                    results.append(
                        text
                    )

            except RuntimeError as e:

                print(
                    "OCR timeout/error:",
                    str(e)
                )

            except Exception as e:

                print(
                    "OCR error:",
                    str(e)
                )

    return results


# =========================================================
# PLATE CANDIDATE SEARCH
# =========================================================

def find_plate(texts):

    candidates = []

    for text in texts:

        # Normal Indian registration
        patterns = [

            r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",

            r"[A-Z]{2}\d{1,2}\d{4,5}",

        ]

        for pattern in patterns:

            matches = re.findall(
                pattern,
                text
            )

            candidates.extend(
                matches
            )

    # Only valid state prefixes
    valid = []

    for candidate in candidates:

        candidate = normalize_text(
            candidate
        )

        if len(candidate) < 7:
            continue

        if len(candidate) > 13:
            continue

        if candidate[:2] not in STATE_CODES:
            continue

        valid.append(
            candidate
        )

    if not valid:
        return None

    # Vote
    counter = {}

    for candidate in valid:

        counter[candidate] = (
            counter.get(candidate, 0) + 1
        )

    best = max(
        counter,
        key=counter.get
    )

    return best


# =========================================================
# PLATE REGION DETECTION
# =========================================================

def find_regions(image):

    regions = []

    h, w = image.shape[:2]

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
        (13, 5)
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

    candidates = []

    for contour in contours:

        x, y, cw, ch = cv2.boundingRect(
            contour
        )

        if cw < 80 or ch < 15:
            continue

        ratio = cw / float(ch)

        if 2.0 <= ratio <= 8.0:

            area = cw * ch

            candidates.append(
                (
                    area,
                    x,
                    y,
                    cw,
                    ch
                )
            )

    # Largest candidates first
    candidates.sort(
        reverse=True
    )

    # Only top 3 candidate regions
    for _, x, y, cw, ch in candidates[:3]:

        px = int(cw * 0.20)
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
# FULL ANALYSIS
# =========================================================

def analyse_vehicle(path):

    image = load_image(
        path
    )

    print(
        "Image:",
        image.shape
    )

    # -----------------------------------------------------
    # STEP 1: Possible plate regions
    # -----------------------------------------------------

    regions = find_regions(
        image
    )

    print(
        "Possible plate regions:",
        len(regions)
    )

    for index, region in enumerate(
        regions
    ):

        print(
            f"Scanning region {index + 1}"
        )

        texts = ocr_image(
            region
        )

        plate = find_plate(
            texts
        )

        if plate:

            print(
                "PLATE FOUND:",
                plate
            )

            return make_result(
                plate
            )

    # -----------------------------------------------------
    # STEP 2: Full image
    # -----------------------------------------------------

    print(
        "Scanning complete image..."
    )

    texts = ocr_image(
        image
    )

    plate = find_plate(
        texts
    )

    if plate:

        print(
            "PLATE FOUND:",
            plate
        )

        return make_result(
            plate
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
# START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE OCR BOT\n\n"
        "📸 Number plate ki clear photo bhejo.\n\n"
        "Main:\n"
        "🔢 Registration number read karunga\n"
        "🇮🇳 State/UT identify karunga\n"
        "🏢 Available RTO reference check karunga\n\n"
        "⚠️ Unknown information guess nahi ki jayegi."
    )


# =========================================================
# HELP
# =========================================================

async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 BEST PHOTO:\n\n"
        "• Number plate close-up\n"
        "• Text sharp ho\n"
        "• Blur kam ho\n"
        "• Reflection kam ho\n"
        "• Plate frame mein large ho\n\n"
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
        "Please wait."
    )

    temp_path = None

    try:

        # Highest resolution Telegram photo
        photo = update.message.photo[-1]

        file = await context.bot.get_file(
            photo.file_id
        )

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=False
        ) as temp:

            temp_path = temp.name

        await file.download_to_drive(
            temp_path
        )

        print(
            "Photo downloaded."
        )

        # Run CPU-heavy OCR separately
        result = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        if not result:

            await status.edit_text(
                "❌ Number plate scan nahi ho paya.\n\n"
                "📸 Number plate ka close-up bhejo.\n\n"
                "Example:\n"
                "Plate photo mein kam se kam "
                "aadhe frame ke aas-paas honi chahiye "
                "aur letters clearly visible hone chahiye."
            )

            return

        message = (
            "✅ NUMBER PLATE FOUND\n\n"
            f"🔢 Registration: {result['plate']}\n"
            f"🇮🇳 State/UT: {result['state']}\n"
        )

        if result["rto_code"]:

            message += (
                f"🏢 RTO Code: "
                f"{result['rto_code']}\n"
            )

        if result["rto_name"]:

            message += (
                f"📍 Known registration area: "
                f"{result['rto_name']}\n"
            )

        else:

            message += (
                "📍 Registration area: "
                "Not available in reference data\n"
            )

        message += (
            "\n⚠️ Registration area current "
            "vehicle location nahi hoti.\n"
            "⚠️ Owner details provide nahi ki jaati."
        )

        await status.edit_text(
            message
        )

    except Exception as e:

        print(
            "PHOTO HANDLER ERROR:",
            repr(e)
        )

        try:

            await status.edit_text(
                "❌ Photo process nahi ho saki.\n\n"
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
            "BOT_TOKEN is missing in Render Environment."
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
        "===================================="
    )
    print(
        "BOT IS RUNNING"
    )
    print(
        "TELEGRAM POLLING: ACTIVE"
    )
    print(
        "LIGHTWEIGHT OCR: ACTIVE"
    )
    print(
        "===================================="
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
