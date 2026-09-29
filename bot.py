import os
import re
import tempfile
import asyncio

import cv2
import numpy as np
import pytesseract

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
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "").rstrip("/")


# =========================
# INDIAN STATE CODES
# =========================

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
    "DD": "Daman and Diu",
}


# =========================
# LOCAL RTO MAPPING
# =========================

RTO_CODES = {

    # Delhi
    "DL01": "Mall Road",
    "DL02": "Indraprastha",
    "DL03": "Sheikh Sarai",
    "DL04": "Janakpuri",
    "DL05": "Loni Road",
    "DL06": "Sarai Kale Khan",
    "DL07": "Mayur Vihar",
    "DL08": "Wazirpur",
    "DL09": "Rohini",
    "DL10": "Kingway Camp",
    "DL11": "Rohini",
    "DL12": "Vasant Vihar",
    "DL13": "Surajmal Vihar",
    "DL14": "Burari",
    "DL15": "West Delhi",
    "DL16": "Ashok Vihar",
    "DL17": "Dwarka",

    # Uttar Pradesh
    "UP14": "Ghaziabad",
    "UP15": "Meerut",
    "UP16": "Gautam Buddh Nagar / Noida",
    "UP19": "Shamli",
    "UP20": "Bijnor",
    "UP21": "Moradabad",
    "UP22": "Rampur",
    "UP23": "Amroha",
    "UP24": "Badaun",
    "UP25": "Bareilly",
    "UP26": "Pilibhit",
    "UP27": "Shahjahanpur",
    "UP30": "Hardoi",
    "UP31": "Lakhimpur Kheri",
    "UP32": "Lucknow",
    "UP33": "Raebareli",
    "UP34": "Sitapur",
    "UP35": "Unnao",
    "UP36": "Barabanki",
    "UP37": "Fatehpur",
    "UP38": "Pratapgarh",
    "UP40": "Bahraich",
    "UP41": "Gonda",
    "UP42": "Ayodhya region",
    "UP43": "Sultanpur",
    "UP44": "Ambedkar Nagar",
    "UP45": "Shrawasti",
    "UP46": "Balrampur",
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
    "UP71": "Fatehpur",
    "UP72": "Kaushambi",
    "UP73": "Hamirpur",
    "UP74": "Mahoba",
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
    "UP90": "Banda",
    "UP91": "Chitrakoot",

    # Maharashtra
    "MH01": "Mumbai Central",
    "MH02": "Mumbai West",
    "MH03": "Mumbai East",
    "MH04": "Thane",
    "MH05": "Kalyan",
    "MH06": "Raigad / Pen",
    "MH07": "Sindhudurg",
    "MH08": "Ratnagiri",
    "MH09": "Kolhapur",
    "MH10": "Sangli",
    "MH11": "Satara",
    "MH12": "Pune",
    "MH13": "Solapur",
    "MH14": "Pimpri-Chinchwad",
    "MH15": "Nashik",
    "MH16": "Ahmednagar",
    "MH17": "Shrirampur",
    "MH19": "Jalgaon",
    "MH20": "Chhatrapati Sambhajinagar",
    "MH21": "Jalna",
    "MH22": "Parbhani",
    "MH23": "Beed",
    "MH24": "Latur",
    "MH25": "Dharashiv",
    "MH26": "Nanded",
    "MH27": "Amravati",
    "MH28": "Buldhana",
    "MH29": "Yavatmal",
    "MH30": "Akola",
    "MH31": "Nagpur",
    "MH32": "Wardha",
    "MH33": "Gadchiroli",
    "MH34": "Chandrapur",
    "MH35": "Gondia",
    "MH36": "Bhandara",
    "MH37": "Washim",
    "MH38": "Hingoli",
}


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):
    text = text.upper()
    text = text.replace("INDIA", "")
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


# =========================
# EXTRACT PLATE
# =========================

def extract_candidates(text):

    text = clean_text(text)

    candidates = set()

    patterns = [
        r"([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})",
        r"([A-Z]{2})(\d{1,2})(\d{4})",
    ]

    for pattern in patterns:

        matches = re.finditer(pattern, text)

        for match in matches:

            value = "".join(match.groups())

            if 7 <= len(value) <= 12:
                candidates.add(value)

    # Search using known state prefixes
    for state in STATE_CODES:

        pattern = state + r"\d{1,2}[A-Z0-9]{3,8}"

        for match in re.finditer(pattern, text):

            value = match.group(0)

            if 7 <= len(value) <= 13:
                candidates.add(value)

    return list(candidates)


# =========================
# IMAGE PROCESSING
# =========================

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

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(enlarged)

    _, otsu = cv2.threshold(
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

    blur = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    sharpen = cv2.addWeighted(
        enhanced,
        1.7,
        blur,
        -0.7,
        0
    )

    return [
        gray,
        enlarged,
        enhanced,
        otsu,
        adaptive,
        sharpen
    ]


# =========================
# OCR
# =========================

def read_plate(path):

    images = preprocess_image(path)

    all_text = []

    for image in images:

        for psm in [6, 7, 11, 13]:

            try:

                text = pytesseract.image_to_string(
                    image,
                    config=f"--oem 3 --psm {psm}"
                )

                if text:
                    all_text.append(text)

            except Exception as error:

                print("OCR error:", error)

    return "\n".join(all_text)


# =========================
# ANALYSIS
# =========================

def analyse_vehicle(path):

    raw_text = read_plate(path)

    candidates = extract_candidates(
        raw_text
    )

    if not candidates:
        return None

    scored = []

    for plate in candidates:

        score = 0

        state = plate[:2]

        if state in STATE_CODES:
            score += 100

        if re.fullmatch(
            r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}",
            plate
        ):
            score += 20

        scored.append(
            (score, plate)
        )

    scored.sort(
        reverse=True
    )

    plate = scored[0][1]

    state_code = plate[:2]

    state_name = STATE_CODES.get(
        state_code
    )

    rto_code = None
    rto_name = None

    match = re.match(
        r"^[A-Z]{2}(\d{1,2})",
        plate
    )

    if match:

        number = match.group(1).zfill(2)

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
        "state_code": state_code,
        "rto_code": rto_code,
        "rto_name": rto_name
    }


# =========================
# TELEGRAM
# =========================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🚗 INDIA VEHICLE VERIFICATION BOT\n\n"

        "Vehicle ki clear photo bhejo.\n\n"

        "🔎 Number plate OCR\n"
        "🇮🇳 State/UT identification\n"
        "📍 Known RTO mapping\n\n"

        "⚠️ Important:\n"
        "Registration area vehicle ki current location "
        "nahi hoti.\n\n"

        "👤 Owner name/address/phone number "
        "provide nahi kiya jayega.\n\n"

        "❌ Unknown information guess nahi ki jayegi."
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Vehicle ki photo bhejo.\n\n"
        "Number plate clear, close aur straight "
        "honi chahiye."
    )


async def photo_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    if not update.message.photo:
        return

    message = await update.message.reply_text(
        "🔎 Number plate scan ho rahi hai..."
    )

    temp_file = tempfile.NamedTemporaryFile(
        suffix=".jpg",
        delete=False
    )

    temp_path = temp_file.name

    temp_file.close()

    try:

        photo = update.message.photo[-1]

        telegram_file = (
            await context.bot.get_file(
                photo.file_id
            )
        )

        await telegram_file.download_to_drive(
            temp_path
        )

        result = await asyncio.to_thread(
            analyse_vehicle,
            temp_path
        )

        if not result:

            await message.edit_text(
                "❌ Number plate verify nahi ho saki.\n\n"

                "📸 Please:\n"
                "• closer photo bhejo\n"
                "• plate ko clear rakho\n"
                "• reflection avoid karo\n\n"

                "⚠️ Bot guess nahi karega."
            )

            return

        response = (
            "🚗 VEHICLE ANALYSIS\n\n"

            f"🔢 Registration: {result['plate']}\n"
            f"🇮🇳 State/UT: {result['state']}\n"
        )

        if result["rto_name"]:

            response += (
                f"📍 Registration area: "
                f"{result['rto_name']}\n"
                f"🏢 RTO Code: "
                f"{result['rto_code']}\n"
            )

        else:

            response += (
                f"🏢 RTO Code: "
                f"{result['rto_code']}\n"
                "📍 Local RTO area: "
                "Not verified in local database\n"
            )

        response += (
            "\n"
            "📌 Current location: NOT DETERMINED\n"
            "👤 Owner details: NOT PROVIDED\n\n"

            "ℹ️ Registration area ≠ current vehicle location."
        )

        await message.edit_text(
            response
        )

    except Exception as error:

        print(
            "Photo processing error:",
            repr(error)
        )

        await message.edit_text(
            "❌ Photo process nahi ho saki.\n"
            "Please ek clear photo dobara bhejo."
        )

    finally:

        try:
            os.remove(temp_path)
        except Exception:
            pass


async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "📸 Vehicle ki photo bhejo."
    )


# =========================
# MAIN
# =========================

def main():

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN environment variable missing."
        )

    if not RENDER_URL:

        raise RuntimeError(
            "RENDER_EXTERNAL_URL environment variable missing."
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

    webhook_url = (
        f"{RENDER_URL}/{BOT_TOKEN}"
    )

    print(
        "Starting Telegram webhook..."
    )

    application.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=BOT_TOKEN,
        webhook_url=webhook_url,
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
