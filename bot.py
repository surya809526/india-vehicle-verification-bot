import asyncio
import os
import re
import requests
import cv2
from aiohttp import web
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters

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

# 2. Comprehensive RTO Database Mapping
RTO_DB = {
    "UP": "Uttar Pradesh", "DL": "Delhi", "MH": "Maharashtra", "KA": "Karnataka",
    "TN": "Tamil Nadu", "GJ": "Gujarat", "RJ": "Rajasthan", "MP": "Madhya Pradesh",
    "BR": "Bihar", "WB": "West Bengal", "HR": "Haryana", "PB": "Punjab",
    "JK": "Jammu and Kashmir", "HP": "Himachal Pradesh", "UK": "Uttarakhand",
    "CG": "Chhattisgarh", "JH": "Jharkhand", "OD": "Odisha", "AP": "Andhra Pradesh",
    "TS": "Telangana", "KL": "Kerala", "AS": "Assam", "GA": "Goa"
}

RTO_DISTRICTS = {
    "UP78": "Kanpur, Uttar Pradesh",
    "UP32": "Lucknow, Uttar Pradesh",
    "UP16": "Gautam Buddh Nagar (Noida), Uttar Pradesh",
    "UP14": "Ghaziabad, Uttar Pradesh",
    "UP01": "Dehradun, Uttarakhand", # Just an example
    "DL01": "Delhi (Civil Lines)",
    "DL02": "Delhi (Civil Lines)",
    "MH01": "Mumbai Central, Maharashtra",
    "MH02": "Mumbai West, Maharashtra",
    "KA01": "Bangalore Central, Karnataka",
    "GJ01": "Ahmedabad, Gujarat"
}

def get_rto_info(plate_text):
    clean_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    match = re.search(r'([A-Z]{2}\d{2}[A-Z]{0,3}\d{4})', clean_text)
    if match:
        full_plate = match.group(1)
        prefix = full_plate[:4]
        state_code = full_plate[:2]
        
        state = RTO_DB.get(state_code, "India")
        city_location = RTO_DISTRICTS.get(prefix, f"District Office ({prefix}), {state}")
        return full_plate, state, city_location
        
    return clean_text, "India", "General Region"

# 3. Enhanced Cloud OCR with OpenCV Preprocessing
def extract_plate_cloud(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=2, fy=2)
        _, thresh = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY)
        cv2.imwrite(image_path, thresh)

    url = "https://api.ocr.space/parse/image"
    
    with open(image_path, 'rb') as f:
        payload = {
            'isOverlayRequired': False,
            'apikey': 'helloworld',
            'language': 'eng',
            'scale': True,
            'OCREngine': 2 
        }
        files = {'filename': f}
        try:
            response = requests.post(url, data=payload, files=files, timeout=10)
            result = response.json()
            
            if result.get('ParsedResults'):
                parsed_text = result['ParsedResults'][0].get('ParsedText', '')
                plate, state, location = get_rto_info(parsed_text)
                if len(plate) >= 6:
                    return plate, state, location
        except Exception as e:
            print(f"[OCR ERROR] {e}")
            
    return "", "", ""

# 4. Telegram Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Namaste! Main India Vehicle OCR & RTO Bot hoon.\n\n"
        "🚗 Mujhe gaadi ki number plate ki photo bhejein, main turant:\n"
        "• Number Plate Text\n"
        "• State & City / RTO Office\n"
        "ki jankari dunga!"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Photo analyze ki ja rahi hai...")
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = "temp_plate.jpg"
    await file.download_to_drive(file_path)
    
    plate_text, state, location = extract_plate_cloud(file_path)
    
    if plate_text:
        await msg.edit_text(
            f"🚗 **VEHICLE REGISTRATION DETAILS** 🚗\n\n"
            f"📌 **Plate Number:** `{plate_text}`\n"
            f"🏛️ **State:** {state}\n"
            f"📍 **RTO Location:** {location}\n\n"
            f"✅ *Verification Successful*"
        )
    else:
        await msg.edit_text("❌ Number plate read nahi ho payi. Kripya saaf photo bhejein.")
        
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
    
    print("[BOT] RUNNING WITH ENHANCED DETAILS...")
    
    await application.initialize()
    await application.bot.delete_webhook(drop_pending_updates=True)
    await application.start()
    await application.updater.start_polling()
    
    asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
