import asyncio
import os
from aiohttp import web
from telegram.ext import Application # Agar python-telegram-bot use kar rahe hain

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

async def main():
    # 1. Pehle web server start ho jo Render ka port pakad lega
    await start_web_server()
    
    # 2. Yahan apna Bot Token daalein (ya os.environ se uthayein)
    TOKEN = os.environ.get("BOT_TOKEN", "AAPKA_BOT_TOKEN_YAHAN")
    
    application = Application.builder().token(TOKEN).build()
    
    # (Yahan apne handlers add kar dena, jaise photo handler)
    
    print("[BOT] RUNNING & LISTENING...")
    
    # 3. Bot ko start karne ka sahi async tarika
    await application.initialize()
    await application.start()
    await application.updater.start_polling()
    
    # App ko chalaye rakhne ke liye
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
