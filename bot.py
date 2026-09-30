import asyncio
import os
from aiohttp import web

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

# Main function jisme web server aur bot dono ek sath chalenge
async def main():
    # 1. Pehle Web Server start ho jo Render ka port pakad le
    await start_web_server()
    
    # 2. Phir yahan aapka Telegram bot start hona chahiye (jaise application.run_polling() ya jo bhi aapka bot start method ho)
    # Example:
    # await application.initialize()
    # await application.start()
    # await application.updater.start_polling()
    
    # Infinite loop taaki app band na ho
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(main())
