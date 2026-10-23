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
PORT = 9559

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
        "http://127.0.0.1:8501/?page=Satellite+Analysis"
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

async def take_screenshot(ws, req_id, output_path, width=1920, height=1080, full_page=True):
    await send_cmd(ws, req_id * 10, "Emulation.setDeviceMetricsOverride", {
        "width": width,
        "height": height,
        "deviceScaleFactor": 1,
        "mobile": False
    })
    await asyncio.sleep(0.5)
    params = {"format": "png"}
    if full_page:
        params["captureBeyondViewport"] = True
    res = await send_cmd(ws, req_id * 10 + 1, "Page.captureScreenshot", params)
    data = res["result"]["data"]
    Path(output_path).write_bytes(base64.b64decode(data))
    print(f"Saved screenshot: {output_path} ({width}x{height}, full_page={full_page})")

async def run_live_tests():
    json_url = f"http://127.0.0.1:{PORT}/json"
    req = urllib.request.urlopen(json_url)
    tabs = json.loads(req.read().decode())
    pages = [t for t in tabs if t.get("type") == "page"]
    ws_url = pages[0]["webSocketDebuggerUrl"]
    
    out_dir = Path(r"d:\Voxel\voxel-earth-observation\results\screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    async with websockets.connect(ws_url, max_size=50_000_000) as ws:
        await send_cmd(ws, 1, "Page.enable")
        await send_cmd(ws, 2, "Runtime.enable")
        
        print("Waiting 6s for Satellite Analysis initial hydration...")
        await asyncio.sleep(6)
        
        # TEST QUERY 1: Loktak Lake, Manipur (Waterbody)
        print("\n==================================================")
        print("TEST 1: Loktak Lake, Manipur (Waterbody Task)")
        print("==================================================")
        
        # Ensure navigation to Satellite Analysis tab
        click_nav = await eval_js(ws, 8, """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const satTab = btns.find(b => b.innerText.includes('Satellite Analysis'));
                if (satTab) { satTab.click(); return 'Clicked Satellite Analysis Tab'; }
                return 'Tab not found';
            })()
        """)
        print(f"Nav tab status: {click_nav}")
        await asyncio.sleep(2)

        # Click Loktak Lake preset button
        click_loktak = await eval_js(ws, 9, """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const lokBtn = btns.find(b => b.innerText.includes('Loktak Lake'));
                if (lokBtn) { lokBtn.click(); return 'Clicked Loktak Preset'; }
                return 'Loktak preset button not found';
            })()
        """)
        print(f"Loktak preset status: {click_loktak}")
        await asyncio.sleep(3)

        # Click Run Analysis button
        click_run = await eval_js(ws, 10, """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const runBtn = btns.find(b => b.innerText.includes('Run Analysis'));
                if (runBtn) {
                    runBtn.click();
                    return 'Clicked Run Analysis';
                }
                return 'Run Analysis button not found';
            })()
        """)
        print(f"Click status: {click_run}")
        
        # Wait for pipeline execution to complete (geocoding, STAC tile, ViT inference)
        print("Waiting for pipeline execution (12s)...")
        await asyncio.sleep(12)
        
        # Check results in DOM
        results_dom = await eval_js(ws, 20, """
            (() => {
                const text = document.body.innerText;
                const hasImg = document.querySelectorAll('img').length;
                const badges = Array.from(document.querySelectorAll('[class*="badge"], [class*="hud"]')).map(b => b.innerText);
                const sceneEl = Array.from(document.querySelectorAll('*')).find(el => el.innerText && el.innerText.startsWith('Scene ID:'));
                const latencyEl = Array.from(document.querySelectorAll('[data-testid="stMetricLabel"]')).find(el => el.innerText.includes('Total Latency'))?.closest('[data-testid="stMetric"]');
                return {
                    imgCount: hasImg,
                    hasWater: text.includes('Permanent Waterbody') || text.includes('Waterbody') || text.includes('Water'),
                    sceneIdText: sceneEl ? sceneEl.innerText : 'Not found',
                    latencyText: latencyEl ? latencyEl.innerText : 'Not found',
                    badges: badges,
                    hasSavedMask: text.includes('Saved Task-Specific BWOA Mask'),
                    hasMeasuredPipeline: text.includes('Measured Pipeline') || text.includes('Sub-second Pipeline')
                };
            })()
        """)
        print(f"Test 1 DOM Check: {json.dumps(results_dom, indent=2)}")
        
        # Capture Test 1 at 1920x1080 and 1366x768
        await take_screenshot(ws, 30, str(out_dir / "live_query_1_loktak_waterbody_1920x1080.png"), 1920, 1080)
        await take_screenshot(ws, 31, str(out_dir / "live_query_1_loktak_waterbody_1366x768.png"), 1366, 768)
        
        # TEST QUERY 2: Jim Corbett National Park, Uttarakhand (Forest Cover)
        print("\n==================================================")
        print("TEST 2: Jim Corbett, Uttarakhand (Forest Cover Task)")
        print("==================================================")
        
        # Click "📍 Jim Corbett (Forest)" preset button
        click_corbett = await eval_js(ws, 40, """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const corbettBtn = btns.find(b => b.innerText.includes('Jim Corbett'));
                if (corbettBtn) {
                    corbettBtn.click();
                    return 'Clicked Jim Corbett Preset';
                }
                return 'Jim Corbett button not found';
            })()
        """)
        print(f"Preset click status: {click_corbett}")
        await asyncio.sleep(4)
        
        # Verify input value
        input_val = await eval_js(ws, 45, "document.querySelector('input[type=\"text\"]')?.value")
        print(f"Current query input in DOM: {input_val}")

        # Click Run Analysis button
        click_run2 = await eval_js(ws, 50, """
            (() => {
                const btns = Array.from(document.querySelectorAll('button'));
                const runBtn = btns.find(b => b.innerText.includes('Run Analysis'));
                if (runBtn) {
                    runBtn.click();
                    return 'Clicked Run Analysis';
                }
                return 'Run Analysis button not found';
            })()
        """)
        print(f"Run click status: {click_run2}")
        
        print("Waiting for pipeline execution (15s)...")
        await asyncio.sleep(15)
        
        results_dom2 = await eval_js(ws, 60, """
            (() => {
                const text = document.body.innerText;
                const hasImg = document.querySelectorAll('img').length;
                const badges = Array.from(document.querySelectorAll('[class*="badge"], [class*="hud"]')).map(b => b.innerText);
                const sceneEl = Array.from(document.querySelectorAll('*')).find(el => el.innerText && el.innerText.startsWith('Scene ID:'));
                const latencyEl = Array.from(document.querySelectorAll('[data-testid="stMetricLabel"]')).find(el => el.innerText.includes('Total Latency'))?.closest('[data-testid="stMetric"]');
                return {
                    imgCount: hasImg,
                    hasForest: text.includes('Dense Forest') || text.includes('Dense Canopy Forest') || text.includes('Forest Cover') || text.includes('Forest'),
                    sceneIdText: sceneEl ? sceneEl.innerText : 'Not found',
                    latencyText: latencyEl ? latencyEl.innerText : 'Not found',
                    badges: badges,
                    hasSavedMask: text.includes('Saved Task-Specific BWOA Mask'),
                    hasMeasuredPipeline: text.includes('Measured Pipeline') || text.includes('Sub-second Pipeline')
                };
            })()
        """)
        print(f"Test 2 DOM Check: {json.dumps(results_dom2, indent=2)}")
        
        # Capture Test 2 at 1920x1080 and 1366x768
        await take_screenshot(ws, 70, str(out_dir / "live_query_2_corbett_forest_1920x1080.png"), 1920, 1080)
        await take_screenshot(ws, 71, str(out_dir / "live_query_2_corbett_forest_1366x768.png"), 1366, 768)

def main():
    proc = start_chrome()
    try:
        asyncio.run(run_live_tests())
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
