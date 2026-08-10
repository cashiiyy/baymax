import httpx

try:
    print("Testing /ocr endpoint with mock image...")
    # Load 2.png
    with open("backend/static/2.png", "rb") as f:
        file_bytes = f.read()
    
    files = {"file": ("Sample-Patient-Medical-Record-Template-edit-online.png", file_bytes, "image/png")}
    data = {"lang": "eng"}
    
    # Send request
    r = httpx.post("http://127.0.0.1:8000/ocr", files=files, data=data, timeout=60.0)
    print("Status:", r.status_code)
    print(r.json())
except Exception as e:
    print("Error:", e)
