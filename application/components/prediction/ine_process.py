import sys
import cv2
import numpy as np
import argparse
import imutils
import glob
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from application.components.prediction.ocr_utilities import *

def ine_process(filePath):

    # Establish Data Type
    documentType = "INE"

    # 1 - Open Image
    image = get_image(filePath)

    # 2 - Correct image orientation
    orientedImg = orient_image(image)

    # 3 - Clusterize image
    clusterizedImg = clusterize_image(orientedImg)

    # 4 - Apply OCR and get text
    dataDict = apply_ocr(clusterizedImg) # dataDict = extract_data(clusterizedImg)

    return dataDict
    
    
