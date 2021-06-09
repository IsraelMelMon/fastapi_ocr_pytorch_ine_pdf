import sys
import cv2
import numpy as np
import argparse
import imutils
import glob
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from utils.ocr_utilities import *

def acta_process(filePath):
    documentType = "ACTA"
    image = get_image(filePath)
    clusterizedImg = clusterize_image(image)