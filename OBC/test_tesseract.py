import pytesseract
import logging

# Set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract version: {version}")
    print("Tesseract is properly configured!")
except Exception as e:
    print(f"Error: {e}") 