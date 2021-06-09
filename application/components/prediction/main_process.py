import sys
import cv2
import numpy as np
import argparse
import imutils
import glob
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from ocr_utilities import *
from ocr_ine_process import *
from ocr_acta_process import *


def OCR_Main_Process(filePath, documentType):
    """
    1. Convertir a imagen
    2. Clasificar documento (INE o ACTA) y corroborar
    3. Llamar proceso de documento específico

    Input: a. Imagen (JPG, PNG) o PDF
    Output: 
    """

    # Check input paths, should be strings
    check_strings(filePath, documentType)

    # Check file type. Should be PNG, JPG or PDF
    fileType = get_file_type(filePath)

    # Check document type, should be INE or ACTA
    check_document_type(documentType)

    # Execute specific process depending on document type
    if documentType == "INE":

        ocr_ine_process(filePath)

    elif documentType == "ACTA":

        ocr_acta_process(filePath)

    # Program end
    print("[INFO] Program ended successfully.")
    
if __name__ == "__main__":
    OCR_Main_Process(filePath="scan_3-1.png", documentType="INE")

