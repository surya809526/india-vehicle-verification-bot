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
# LIMITED LOCAL RTO REFERENCE
# Unknown codes are NOT guessed.
# =========================================================

RTO_CODES = {
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
    "MH21": "Jalna",
    "MH27": "Amravati",
    "MH30": "Akola",
    "MH31": "Nagpur",
}


# =========================================================
# HEALTH SERVER FOR RENDER
# =========================================================

async def health(request):
    return web.Response(
        text="Vehicle Verification Bot is running."
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

    print(f"HTTP health server running on port {PORT}")

    return runner


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(path):

    image = cv2.imread(path)

    if image is None:
        return []

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )

    enhanced = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(enlarged)

    _, threshold = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    return [
        gray,
        enlarged,
        enhanced,
        threshold,
        adaptive,
    ]


# =========================================================
# OCR
# =========================================================

def run_ocr(path):

    images = preprocess_image(path)

    results = []

    for image in images:

        for psm in (6, 7, 11, 13):

            try:

                text = pytesseract.image_to_string(
                    image,
                    config=f"--oem 3 --psm {psm}",
                    lang="eng"
                )

                if text:
                    results.append(text)

            except Exception as e:

                print("OCR error:", e)

    return "\n".join(results)


# =========================================================
# CLEAN OCR
# =========================================================

def normalize_ocr(text):

    text = text.upper()

    replacements = {
        " ": "",
        "\n": "",
        "\r": "",
        "-": "",
        ".": "",
        ":": "",
        "_": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )


# =========================================================
# PLATE DETECTION
# =========================================================

def detect_plate(raw_text):

    text = normalize_ocr(raw_text)

    candidates = []

    # Standard Indian registration
    patterns = [
        r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",
        r"[A-Z]{2}\d{1,2}\d{4}",
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        candidates.extend(matches)

    # Search around known state prefixes
    for state in STATE_CODES:

        pattern = state + r"\d{1,2}[A-Z0-9]{3,8}"

        matches = re.findall(
            pattern,
            text
        )

        candidates.extend(matches)

    # Remove duplicates
    candidates = list(
        dict.fromkeys(candidates)
    )

    # Keep reasonable registrations
    candidates = [
        x for x in candidates
        if 7 <= len(x) <= 13
    ]

    if not candidates:
        return None

    # Prefer candidate starting with a valid state code
    candidates.sort(
        key=lambda x: (
            x[:2] in STATE_CODES,
            len(x)
        ),
        reverse=True
    )

    return candidates[0]


# =========================================================
# ANALYSE
# =========================================================

def analyse_vehicle(path):

    raw_text = run_ocr(path)

    print("OCR RAW:", repr(raw_text))

    plate = detect_plate(raw_text)

    if not plate:
        return None

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

        number = match.group(2).zfill(2)

        rto_code = (
            state_code +
            number
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
# TELEGRAM COMMANDS
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE VERIFICATION BOT\n\n"
        "Number plate ki clear photo bhejo.\n\n"
        "🔎 Main photo se number plate read "
        "karne ki koshish karunga.\n\n"
        "🇮🇳 State/UT identify kiya jayega.\n"
        "📍 Available RTO reference check kiya jayega.\n\n"
        "⚠️ Main unknown information guess nahi karta.\n"
        "⚠️ Current vehicle location aur owner details "
        "photo se determine nahi ki ja sakti."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Number plate ki clear photo bhejo.\n\n"
        "Best result ke liye plate:\n"
        "• close-up ho\n"
        "• readable ho\n"
        "• zyada blur na ho\n"
        "• reflection kam ho"
    )


# =========================================================
# PHOTO HANDLER
# =========================================================

async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message or not update.message.photo:
        return

    status = await update.message.reply_text(
        "🔎 Number plate scan ho rahi hai...\n"
        "Please wait."
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
            "Downloaded image:",
            temp_path
        )

        # OCR CPU work ko Telegram event loop se alag run karo
        result = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        if not result:

            await status.edit_text(
                "❌ Number plate verify nahi ho saki.\n\n"
                "Please plate ki aur clear/close photo bhejo.\n\n"
                "⚠️ Bot guessed number nahi dega."
            )

            return

        response = (
            "🚗 VEHICLE ANALYSIS\n\n"
            f"🔢 Registration: {result['plate']}\n"
            f"🇮🇳 State/UT: {result['state']}\n"
        )

        if result["rto_code"]:
            response += (
                f"🏢 RTO Code: {result['rto_code']}\n"
            )

        if result["rto_name"]:

            response += (
                f"📍 Known registration area: "
                f"{result['rto_name']}\n"
            )

        else:

            response += (
                "📍 Registration area: "
                "Not verified in local reference\n"
            )

        response += (
            "\n"
            "⚠️ Registration area current vehicle "
            "location nahi hoti.\n"
            "👤 Owner details provide nahi ki ja rahi."
        )

        await status.edit_text(
            response
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
                os.remove(temp_path)
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

    # Render health server
    health_runner = await start_health_server()

    # Telegram application
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

    print("Starting Telegram bot polling...")

    await application.initialize()
    await application.start()

    await application.updater.start_polling(
        drop_pending_updates=True,
        allowed_updates=Update.ALL_TYPES
    )

    print("===================================")
    print("BOT IS RUNNING")
    print("Telegram polling: ACTIVE")
    print("===================================")

    try:

        # Keep service alive
        await asyncio.Event().wait()

    finally:

        print("Stopping bot...")

        await application.updater.stop()
        await application.stop()
        await application.shutdown()

        await health_runner.cleanup()


if __name__ == "__main__":

    try:
        asyncio.run(main())

    except KeyboardInterrupt:

        print("Bot stopped.")
