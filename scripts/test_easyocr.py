import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

print("Generating image with text...")
img = Image.new('RGB', (400, 200), color = (255, 255, 255))
d = ImageDraw.Draw(img)
d.text((10,10), "Patient Name: John Doe", fill=(0,0,0))
d.text((10,40), "Diagnosis: Hypertension", fill=(0,0,0))

img_np = np.array(img)
img_bgr = img_np[:, :, ::-1]

print("Running easyocr...")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
results = reader.readtext(img_bgr, detail=0)
raw_text = "\n".join(results)

print("Extracted Text:")
print(raw_text)
