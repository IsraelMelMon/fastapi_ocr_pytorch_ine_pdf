import sys
import cv2
import numpy as np
import argparse
import imutils
import glob
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
# from ..pytorch_utilities.get_data import *
from .get_data import *

#pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

SUPPORTED_FILE_TYPES = [".jpg", ".png", ".pdf", ".jpeg"]

SUPPORTED_DOCUMENT_TYPES = ["INE", "ACTA"]

ERROR_DESCRIPTIONS_DICT = {
                            0: "No error description.",
                            50:"Provided document path and/or type are not strings.",
                            100:"File path could not be understood, no extension was found.",
                            101:"File extension not supported.",
                            111:"Document type is not supported.",
                            200:"Could not open image. Check path and file name."
                        }
def get_file_type(filePath):
    try:
        fileExtension = filePath.split(".")[-1]
        fileExtension = "." + fileExtension
    except:
        evaluate_error(errorFlag = 1, errorCode=100)
        fileExtension = None

    else:
        errorFlag=0 if fileExtension in SUPPORTED_FILE_TYPES else 1
        if errorFlag:
            evaluate_error(errorFlag = 1, errorCode=101)

        else:
            errorCode = 0
        
    report_state(f"Extension verified. File type is {fileExtension}")
    return fileExtension

def report_error(errorCode):
    if errorCode in ERROR_DESCRIPTIONS_DICT:
        errorDescription = ERROR_DESCRIPTIONS_DICT[errorCode]
    else:
        errorCode = "XXXX"
        errorDescription = "Unknown Error."
    print(f"[ERROR] Error code {errorCode}. {errorDescription} Shutting down program.")
    sys.exit()

def evaluate_error(errorFlag=0, errorCode=0):
    # print(f"[INFO] Error flag is {errorFlag}")
    # print(f"[INFO] Error code is {errorCode}")
    if errorFlag == 0:
        "[INFO] No error"
    else:
        report_error(errorCode)
    
def check_document_type(documentType):
    if documentType in SUPPORTED_DOCUMENT_TYPES:
        report_state(f"Document type is supported: {documentType}")
    else:
        evaluate_error(errorFlag = 1, errorCode=111)

def check_strings(filePath, documentType):
    # Check if input file path and document type are strings
    if isinstance(filePath, str) and isinstance(documentType, str):
        report_state(f"Input format is correct (strings)")
    else:
        evaluate_error(errorFlag = 1, errorCode=50)

def get_image(filePath):
    try:
        image = cv2.imread(filePath,1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        image = None
        evaluate_error(errorFlag = 1, errorCode=200)
    else:
        report_state(f"Image opened correctly.")
    return image

def report_state(message):
    print(f"[INFO] {message}")


def clusterize_image(image):
    report_state(f"Clusterizing...")
    # convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    #print(pixel_values.shape)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    # show the image
    # merged = cv2.merge([G, G, G])
    cv2.imwrite("result/output.png", segmented_image)
    # Notify that the process was successful
    report_state(f"Image clusterized successfully. Saved as 'result/output.jpg'.")
    # report_state(f"Showing resulting image:")
    # plt.imshow(segmented_image)
    # plt.show()
    report_state(f"Image closed.")
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 2
    masked_image[labels == cluster] = [0, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    # show the image
    # plt.imshow(masked_image)
    # plt.show()
    return segmented_image

def orient_image(image):
    degrees = 0
    if degrees != 0:
        report_state(f"Original image was rotated {degrees}°. Correction applied.")
    else:
        report_state(f"No significant image rotation detected.")
    return image

def apply_ocr(image):
    trainedModel = 'models/craft_mlt_25k.pth'
    textThreshold = 0.7
    lowText = 0.4
    linkThreshold = 0.4
    cudaFlag = False
    canvasSize = 1280
    magRatio = 1.5
    polyFlag = False
    showTimeFlag = False
    testFolder = "test"
    refineFlag = False
    refinerModel = 'models/craft_refiner_CTW1500.pth'

    # Apply OCR to get text data
    dataDict = get_data(trainedModel,
    textThreshold,
    lowText,
    linkThreshold,
    cudaFlag,
    canvasSize,
    magRatio,
    polyFlag,
    showTimeFlag,
    testFolder,
    refineFlag,
    refinerModel)

    return dataDict
    
def get_sections(image):
    pass

def extract_data(image):
    nameCoord = (432, 206, 576, 181)
    birthDateCoord = (439, 383, 663, 190)
    genderCoord = (1095, 206, 303, 99) 
    homeCoord = (1218, 296, 149, 60) 
    electorKeyCoord = (445, 557, 647, 66)
    curpCoord = (444, 610, 497, 66)
    registryDateCoord = (975, 607, 393, 72)
    dataCoordLst = [nameCoord,birthDateCoord,genderCoord,homeCoord,electorKeyCoord,curpCoord,registryDateCoord]

    dataNameLst = ["name","birth","sex","address", "key","curp","registry"]

    img = image
    resizedImg = cv2.resize(img, (1400,850))
    dataImgLst = []
    dataDict = {}
    report_state(f"Detecting sections and applying Tesseract for character recognition...")
    for i,r in enumerate(dataCoordLst):
        imCrop = resizedImg[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        #cv2.imshow("Image", imCrop)
        cv2.imwrite(f"imCrop_{i}.png", imCrop)
        cv2.waitKey(0)
        ocrImage = Image.fromarray(np.uint8(imCrop)).convert('RGB')
        ocrImage = Image.fromarray(imCrop.astype('uint8'), 'RGB')
        if i == 6:
            text = pytesseract.image_to_string(ocrImage, lang = "eng")
        else:
            text = pytesseract.image_to_string(ocrImage, lang = "eng")
        dataDict[dataNameLst[i]] = text
    report_state(f"OCR process finished, displaying results:")
    for k,v in dataDict.items():
        print(k,v)
    print(dataDict)
    return dataDict

def encode_image(image):
    pass