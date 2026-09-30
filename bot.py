import os
from aiohttp import web

async def health(request):
    return web.Response(text="India Vehicle OCR Bot running")

async def start_server():
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    
    # Render ka diya hua PORT uthayega, agar nahi mila toh 10000 use karega
    port = int(os.environ.get("PORT", 10000))
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    print(f"Server started on port {port}")
