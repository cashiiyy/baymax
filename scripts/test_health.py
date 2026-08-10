import httpx

try:
    print("Testing /health...")
    r = httpx.get("http://127.0.0.1:8000/health", timeout=3.0)
    print("Health Status:", r.status_code)
    print(r.json())
    
    print("\nTesting /api/health...")
    r2 = httpx.get("http://127.0.0.1:8000/api/health", timeout=10.0)
    print("API Health Status:", r2.status_code)
    print(r2.json())
except Exception as e:
    print("Error:", e)
