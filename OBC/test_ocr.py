import cv2
import pytesseract
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Create a simple test image with text
img = np.zeros((100, 300), dtype=np.uint8)
img.fill(255)  # White background
cv2.putText(img, "Hello OCR!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save the test image
cv2.imwrite("test_ocr.png", img)

try:
    # Try to read the text from the image
    text = pytesseract.image_to_string(img)
    print("OCR Result:", text.strip())
    print("OCR is working properly!")
except Exception as e:
    print(f"Error performing OCR: {e}") 