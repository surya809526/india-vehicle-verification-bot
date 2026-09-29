import cv2
import numpy as np
import pytesseract

def clean_and_read_plate(image_path):
    # 1. Raw image read karein
    img = cv2.imread(image_path)
    if img is None:
        return "Image load nahi ho payi!"

    # 2. Size chhota karein taaki timeout na ho (Fast processing ke liye)
    height, width = img.shape[:2]
    if width > 800:
        img = cv2.resize(img, (800, int(height * (800 / width))))

    # 3. Grayscale mein convert karein (Color hatayein)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. Noise remove karne ke liye Thoda Blur karein
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Thresholding (Black & White taaki number saaf dikhein)
    processed_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # 6. Tesseract OCR run karein (Timeout 5 seconds set kiya hai)
    try:
        custom_config = r'--oem 3 --psm 7'
        text = pytesseract.image_to_string(processed_img, config=custom_config, timeout=5)
        return text.strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"
