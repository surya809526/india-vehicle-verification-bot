import asyncio
import os
import cv2
import pytesseract
from aiohttp import web
from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, CommandHandler, filters

# 1. Render Health Check Server (Port bind karne ke liye)
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

# 2. Image Processing & OCR Function
def extract_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    
    height, width = img.shape[:2]
    if width > 800:
        img = cv2.resize(img, (800, int(height * (800 / width))))
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    try:
        text = pytesseract.image_to_string(thresh, config='--psm 7', timeout=5)
        return text.strip()
    except:
        return ""

# 3. Start Command Handler (Jab user /start bheje)
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Namaste! Main India Vehicle OCR Bot hoon.\n\n"
        "🚗 Kripya mujhe kisi bhi gaadi ki number plate ki saaf photo bhejein, main turant number plate detect karke uski details bataunga!"
    )

# 4. Telegram Photo Handler (Jab user photo bheje)
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔍 Photo mil gayi hai, number plate detect ki ja rahi hai...")
    
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    file_path = "temp_plate.jpg"
    await file.download_to_drive(file_path)
    
    # OCR run karein
    plate_text = extract_plate(file_path)
    
    if plate_text:
        await update.message.reply_text(f"🚗 Detected Plate Number: {plate_text}")
    else:
        await update.message.reply_text("❌ Number plate saaf detect nahi ho payi. Kripya saaf aur seedhi photo bhejein.")
        
    # Temporary file delete karein
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
    
    # Handlers Register Karein (/start aur Photo dono ke liye)
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    print("==========================================")
    print("INDIA VEHICLE OCR BOT")
    print("Universal plate detection: ON")
    print("Single-line OCR: ON")
    print("[BOT] RUNNING & LISTENING...")
    print("==========================================")
    
    await application.initialize()
    await application.bot.delete_webhook(drop_pending_updates=True)
    await application.start()
    await application.updater.start_polling()
    
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
