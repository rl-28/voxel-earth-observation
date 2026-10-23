import sys
import os
import json
import subprocess
import time
import urllib.request
import asyncio
import websockets
import base64
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9558

def start_chrome():
    user_data_dir = tempfile.mkdtemp(prefix="chrome_cdp_")
    cmd = [
        CHROME_PATH,
        f"--remote-debugging-port={PORT}",
        f"--user-data-dir={user_data_dir}",
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        "--window-size=1920,1080",
        "--hide-scrollbars",
        "http://127.0.0.1:8501/"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)
    return proc

async def send_cmd(ws, req_id, method, params=None):
    payload = {"id": req_id, "method": method}
    if params:
        payload["params"] = params
    await ws.send(json.dumps(payload))
    while True:
        raw = await ws.recv()
        msg = json.loads(raw)
        if msg.get("id") == req_id:
            return msg

async def eval_js(ws, req_id, expression):
    res = await send_cmd(ws, req_id, "Runtime.evaluate", {
        "expression": expression,
        "returnByValue": True
    })
    return res.get("result", {}).get("result", {}).get("value")

async def take_screenshot(ws, req_id, output_path, width=1920, height=1080):
    await send_cmd(ws, req_id * 10, "Emulation.setDeviceMetricsOverride", {
        "width": width,
        "height": height,
        "deviceScaleFactor": 1,
        "mobile": False
    })
    await asyncio.sleep(0.5)
    res = await send_cmd(ws, req_id * 10 + 1, "Page.captureScreenshot", {"format": "png"})
    data = res["result"]["data"]
    Path(output_path).write_bytes(base64.b64decode(data))
    print(f"Saved: {output_path} ({width}x{height})")

async def run_inspection():
    json_url = f"http://127.0.0.1:{PORT}/json"
    req = urllib.request.urlopen(json_url)
    tabs = json.loads(req.read().decode())
    pages = [t for t in tabs if t.get("type") == "page"]
    ws_url = pages[0]["webSocketDebuggerUrl"]
    
    out_dir = Path(r"d:\Voxel\voxel-earth-observation\results\screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pages_to_capture = [
        ("http://127.0.0.1:8501/?page=Overview", "page_1_overview"),
        ("http://127.0.0.1:8501/?page=Satellite+Analysis", "page_2_satellite_analysis"),
        ("http://127.0.0.1:8501/?page=Spectral+Band+Optimization", "page_3_spectral_optimization"),
        ("http://127.0.0.1:8501/?page=Model+Intelligence", "page_4_model_intelligence"),
        ("http://127.0.0.1:8501/?page=Research+%26+Evaluation", "page_5_research_evaluation"),
    ]
    
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        await send_cmd(ws, 1, "Page.enable")
        await send_cmd(ws, 2, "Runtime.enable")
        
        req_id = 100
        for url, file_prefix in pages_to_capture:
            print(f"\n--- Navigating to: {url} ---")
            await send_cmd(ws, req_id, "Page.navigate", {"url": url})
            req_id += 1
            
            # Wait for Streamlit page to hydrate
            print("Waiting 5s for page render...")
            await asyncio.sleep(5)
            
            # Check for errors in DOM
            dom_check = await eval_js(ws, req_id, """
                (() => {
                    const errors = Array.from(document.querySelectorAll('.stException, [data-testid="stException"]')).map(e => e.innerText);
                    return {
                        title: document.title,
                        heading: document.querySelector('h1, h2')?.innerText,
                        textLen: document.body.innerText.length,
                        hasErrors: errors.length > 0,
                        errors: errors
                    };
                })()
            """)
            print(f"DOM Check: {json.dumps(dom_check, indent=2)}")
            req_id += 1
            
            # 1920x1080 screenshot
            await take_screenshot(ws, req_id, str(out_dir / f"{file_prefix}_1920x1080.png"), 1920, 1080)
            req_id += 2
            
            # 1366x768 screenshot
            await take_screenshot(ws, req_id, str(out_dir / f"{file_prefix}_1366x768.png"), 1366, 768)
            req_id += 2

def main():
    proc = start_chrome()
    try:
        asyncio.run(run_inspection())
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
