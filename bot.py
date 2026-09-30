import asyncio
import os
import re
import cv2
import easyocr
from aiohttp import web
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters

# Initialize EasyOCR Reader (Hindi/English ke liye)
print("[INFO] Loading EasyOCR Model...")
reader = easyocr.Reader(['en'], gpu=False)
print("[INFO] EasyOCR Model Loaded!")

# 1. Render Health Check Server
async def health(request):
    return web.Response(text="India Vehicle OCR Bot running")

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    
    port = int(os.environ.get("PORT", 10000))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"[SERVER] Running on {port}")

# 2. RTO Database
RTO_DB = {
    "UP": "Uttar Pradesh", "DL": "Delhi", "MH": "Maharashtra", "KA": "Karnataka",
    "TN": "Tamil Nadu", "GJ": "Gujarat", "RJ": "Rajasthan", "MP": "Madhya Pradesh",
    "BR": "Bihar", "WB": "West Bengal", "HR": "Haryana", "PB": "Punjab"
}

RTO_DISTRICTS = {
    "UP78": "Kanpur, Uttar Pradesh",
    "UP32": "Lucknow, Uttar Pradesh",
    "UP16": "Gautam Buddh Nagar (Noida), Uttar Pradesh",
    "UP14": "Ghaziabad, Uttar Pradesh",
    "DL01": "Delhi (Civil Lines)",
    "MH01": "Mumbai Central, Maharashtra",
    "KA01": "Bangalore Central, Karnataka"
}

def get_rto_info(plate_text):
    clean_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    match = re.search(r'([A-Z]{2}\d{2}[A-Z]{0,3}\d{4})', clean_text)
    if match:
        full_plate = match.group(1)
        prefix = full_plate[:4]
        state_code = full_plate[:2]
        city = RTO_DISTRICTS.get(prefix, RTO_DB.get(state_code, "India"))
        return full_plate, city
        
    for code, location in RTO_DISTRICTS.items():
        if code in clean_text:
            return clean_text, location
            
    return clean_text, "India (General)"

# 3. EasyOCR Plate Extraction
def extract_plate_easyocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "", ""
    
    # EasyOCR direct image par bahut accha kaam karta hai
    results = reader.readtext(img)
    
    combined_text = ""
    for (bbox, text, prob) in results:
        if prob > 0.2: # Confidence threshold
            combined_text += " " + text
            
    plate, location = get_rto_info(combined_text)
    if len(plate) >= 6:
        return plate, location
        
    return "", ""

# 4. Telegram Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Namaste! Main India Vehicle OCR & RTO Bot hoon.\n\n"
        "🚗 Ab maine **EasyOCR** upgrade kar liya hai! Aap chahe jaisi bhi photo bhejein, yeh turant number plate aur RTO location bata dega."
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Photo scan ki ja rahi hai...")
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = "temp_plate.jpg"
    await file.download_to_drive(file_path)
    
    plate_text, location = extract_plate_easyocr(file_path)
    
    if plate_text:
        await msg.edit_text(
            f"✅ **Number Plate Detected!**\n\n"
            f"🚗 Plate: `{plate_text}`\n"
            f"📍 Location: **{location}**"
        )
    else:
        await msg.edit_text("❌ Number plate read nahi ho payi. Kripya thodi aur saaf photo bhejein.")
        
    if os.path.exists(file_path):
        os.remove(file_path)

# 5. Main Function
async def main():
    await start_web_server()
    
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        print("[ERROR] BOT_TOKEN missing!")
        return
        
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("[BOT] RUNNING WITH EASYOCR...")
    
    await application.initialize()
    await application.bot.delete_webhook(drop_pending_updates=True)
    await application.start()
    await application.updater.start_polling()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
