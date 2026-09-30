import asyncio
import os
import re
import cv2
import pytesseract
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

# 2. Indian RTO State & City Database (Prefix Mapping)
RTO_DB = {
    "UP": "Uttar Pradesh",
    "DL": "Delhi",
    "MH": "Maharashtra",
    "KA": "Karnataka",
    "TN": "Tamil Nadu",
    "GJ": "Gujarat",
    "RJ": "Rajasthan",
    "MP": "Madhya Pradesh",
    "BR": "Bihar",
    "WB": "West Bengal",
    "HR": "Haryana",
    "PB": "Punjab",
    "JK": "Jammu and Kashmir",
    "HP": "Himachal Pradesh",
    "UK": "Uttarakhand",
    "CG": "Chhattisgarh",
    "JH": "Jharkhand",
    "OD": "Odisha",
    "AP": "Andhra Pradesh",
    "TS": "Telangana",
    "KL": "Kerala",
    "AS": "Assam",
    "GA": "Goa",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "SK": "Sikkim",
    "TR": "Tripura",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "LD": "Lakshadweep",
    "PY": "Puducherry",
    "LA": "Ladakh"
}

# Specific RTO District Mapping (Examples)
RTO_DISTRICTS = {
    "UP78": "Kanpur, Uttar Pradesh",
    "UP32": "Lucknow, Uttar Pradesh",
    "UP16": "Gautam Buddh Nagar (Noida), Uttar Pradesh",
    "UP14": "Ghaziabad, Uttar Pradesh",
    "DL01": "Delhi (Civil Lines)",
    "DL02": "Delhi (Civil Lines)",
    "MH01": "Mumbai Central, Maharashtra",
    "MH02": "Mumbai West, Maharashtra",
    "KA01": "Bangalore Central, Karnataka",
    "GJ01": "Ahmedabad, Gujarat"
}

def get_rto_info(plate_text):
    # Clean text: remove spaces and special characters
    clean_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
    
    # Match Indian Vehicle pattern (e.g., UP78HM7865)
    match = re.search(r'([A-Z]{2}\d{2}[A-Z]{0,3}\d{4})', clean_text)
    if match:
        full_plate = match.group(1)
        prefix = full_plate[:4] # e.g. UP78
        state_code = full_plate[:2] # e.g. UP
        
        city = RTO_DISTRICTS.get(prefix, RTO_DB.get(state_code, "India"))
        return full_plate, city
        
    # Agar exact match na ho toh state/prefix dhoondhein
    for code, location in RTO_DISTRICTS.items():
        if code in clean_text:
            return clean_text, location
            
    return clean_text, "Unknown Location"

# 3. Advanced Image Processing for Blurry & Two-Line Plates
def extract_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    # Resize to enhance small/blurry text
    height, width = img.shape[:2]
    scale = max(1, 1000 / width)
    img = cv2.resize(img, (int(width * scale), int(height * scale)))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Bilateral filter to remove noise while keeping edges sharp (Great for blur)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    
    # Adaptive Thresholding for varying lighting and two-line plates
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    
    # Try multiple PSM configs for single and multi-line plates
    configs = ['--psm 7', '--psm 8', '--psm 6', '--psm 3']
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(thresh, config=config, timeout=3)
            cleaned, location = get_rto_info(text)
            if len(cleaned) >= 6: # Valid plate length usually >= 8
                return cleaned, location
        except:
            continue
            
    return "", "Unknown"

# 4. Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Namaste! Main India Vehicle OCR & RTO Bot hoon.\n\n"
        "🚗 Mujhe kisi bhi gaadi ki number plate ki photo bhejein (chahe blurry ho ya two-line wali), main number detect karke RTO location bataunga!"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔍 Photo analyze ki ja rahi hai, blur/two-line plate process ho rahi hai...")
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = "temp_plate.jpg"
    await file.download_to_drive(file_path)
    
    plate_text, location = extract_plate(file_path)
    
    if plate_text and location != "Unknown":
        await msg.edit_text(
            f"✅ **Number Plate Detected!**\n\n"
            f"🚗 Plate: `{plate_text}`\n"
            f"📍 RTO Location / State: **{location}**"
        )
    elif plate_text:
        await msg.edit_text(
            f"⚠️ **Plate Found:** `{plate_text}`\n"
            f"📍 Location: India (Exact RTO mapping not found)"
        )
    else:
        await msg.edit_text("❌ Number plate read nahi ho payi. Kripya thodi aur clear ya nazdeek ki photo bhejein.")
        
    if os.path.exists(file_path):
        os.remove(file_path)

# 5. Main Function
async def main():
    await start_web_server()
    
    TOKEN = os.environ.get("BOT_TOKEN")
    if not TOKEN:
        print("[ERROR] BOT_TOKEN environment variable is missing!")
        return
        
    application = Application.builder().token(TOKEN).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("==========================================")
    print("INDIA VEHICLE OCR & RTO BOT - ADVANCED")
    print("[BOT] RUNNING & LISTENING...")
    print("==========================================")
    
    await application.initialize()
    await application.bot.delete_webhook(drop_pending_updates=True)
    await application.start()
    await application.updater.start_polling()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
