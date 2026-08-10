import pytesseract
from PIL import Image
import sys

image_path = "K:/PROJECTS/BAYMAX/backend/static/2.png"

try:
    print(f"Opening image: {image_path}")
    img = Image.open(image_path)
    print("Calling pytesseract.image_to_string...")
    text = pytesseract.image_to_string(img, lang="eng")
    print("Extracted text successfully!")
    print(text[:200])
except Exception as e:
    print("Failed with exception:")
    print(e)
    print("Exception type:", type(e))
