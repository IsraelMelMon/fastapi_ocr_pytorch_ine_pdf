# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from application.components.prediction import imgproc
import pytesseract
from application.components.prediction.cluster import clustering
import re
import math

#pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
# borrowed from https://github.com/lengstrom/fast-style-transfer/blob/master/src/utils.py
def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files = []
    mask_files = []
    gt_files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            elif ext == '.bmp':
                mask_files.append(os.path.join(dirpath, file))
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            elif ext == '.zip':
                continue
    # img_files.sort()
    # mask_files.sort()
    # gt_files.sort()
    return img_files, mask_files, gt_files

def distance(point, point2):
    x1, y1 = point
    x2, y2 = point2
    distance = math.sqrt((x1-x2)**2+(y1-y2)**2)
    return distance

def saveResult(img_file, img, boxes, dirname='./result/', verticals=None, texts=None):
        """ save text detection result one by one
        Args:
            img_file (str): image file name
            img (array): raw image context
            boxes (array): array of result file
                Shape: [num_detections, 4] for BB output / [num_detections, 4] for QUAD output
        Return:
            None
        """
        img = np.array(img)
        #print("image shape:", img.shape)
        height = img.shape[0]
        width = img.shape[1]
        #img = clustering(img)
        alpha = 1.25 # Contrast control (1.0-3.0)
        beta = 0# Brightness control (0-100)

        img= cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        # make result file list
        filename, file_ext = os.path.splitext(os.path.basename(img_file))

        # result directory
        res_file = dirname + "res_" + filename + '.txt'
        res_img_file = dirname + "res_" + filename + '.jpg'

        if not os.path.isdir(dirname):
            os.mkdir(dirname)

        # ignore top bboxes
        boxes = boxes[2:]
        # enlist top left corner of bboxes
        top_l_points = []
        textd = []
        #texts = [i for i, _ in enumerate(boxes)]
        with open(res_file, 'w') as f:
            for i, box in enumerate(boxes):
                poly = np.array(box).astype(np.int32).reshape((-1))
                strResult = ','.join([str(p) for p in poly]) + '\r\n'
                f.write(strResult)
                #### these points contain edges of boxes or polygons, dependin
                ## on argument of SaveResult
                poly = poly.reshape(-1, 2)

                #### these points contain edges of boxes ##
                #x, y, w, h = poly[0][0 ], poly[0][1], poly[2][0], poly[2][1]
                # draw first point of box
                #cv2.circle(img, (poly[0][0], poly[0][1]), 6, (255,0,0), 2)
                # x, y = tuple(poly[0])
                # w, h = tuple(poly[2])
                y, x = tuple(poly[0])
                h, w = tuple(poly[2])
                #print(f"Coordinates are x {x}, y {y}, w {w}, h {h}")
                img_copy = img.copy()

                cropped_boxes = img_copy[int(min(x,w)-4):int(max(x,w)+4), int(min(y,h)-4):int(max(y,h)+4)]
                #print("cropped boxes: ",cropped_boxes )
                
                #print("min and max (w,x), min and max (y,h)",
                if cropped_boxes is not None:
                    cv2.imwrite("saved_{}_box.png".format(i), cropped_boxes)
                   
                    dilated_img = cv2.dilate(cropped_boxes[:,:,1], np.ones((33,33 ), np.uint8))
                    #bg_img = cv2.GaussianBlur(dilated_img, (9,9),0)
                    bg_img = cv2.medianBlur(dilated_img, 11)

                    #--- finding absolute difference to preserve edges ---
                    diff_img = 255 - cv2.absdiff(cropped_boxes[:,:,1], bg_img)

                    #--- normalizing between 0 to 255 ---
                    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    #--- Otsu threshold ---
                    #th = cv2.adaptiveThreshold(norm_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
                    cropped_boxes = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                    
                    #cropped_boxes = clustering(cropped_boxes)
                    

                    text = pytesseract.image_to_string(cropped_boxes, lang="spa", config='--psm 6')
                    #print("text by crop box {}".format(i), text)
                    top_l_points.append(tuple(poly[0]))

                    text_w_n_blanks = text.strip() 
                    textd.append(text_w_n_blanks)

                    # Check where DoB is
                    check_dob = re.search("[0-9]{1,2}(/)[0-9]{1,2}(/)[0-9]{4}", 
                        text_w_n_blanks)
                    if check_dob:
                        x_b, y_b = tuple(poly[0])
                        DoB_point = tuple(poly[0])
                        DoB_text = text

                        #print("DoB point: ", poly[0])
                        print("DoB: ", DoB_text)


                    # Check where curp is
                    check_curp = re.search("[a-zA-Z]{4}[0-9]{6}[a-zA-Z]{4}", text_w_n_blanks)
                    if check_curp:
                        curp = text.split(" ")[-1]
                        print("curp: ", curp)

                    # Check where clave de elector is
                    check_clave = re.search("[a-zA-Z]{6}[0-9]{8}[a-zA-Z]{1}", text_w_n_blanks)
                    if check_clave:
                        clave = text.split(" ")[-1]
                        print("clave: ", clave)

                    # Check where registro is 
                    check_registro= re.search("[0-9]{4}( )[0-9]{2}", text_w_n_blanks)
                    if check_registro:
                        registro1 = text.split(" ")[-2:-1][0]
                        registro2 = text.split(" ")[-1]
                        registro = registro1+" "+registro2
                        print("registro: ", registro1, registro2)
                    
                    # Check emisión and vigencia
                    #check_vig_emision= re.search("[0-9]{4}( )[a-zA-Z]{8}( )[0-9]{4}", 
                    #    text_w_n_blanks)

                    vig = text_w_n_blanks.split(" ")[-1]
                    emi = text_w_n_blanks.split(" ")[0]
                    check_vig= re.search("[0-9]{4}", vig)
                    check_emi= re.search("[0-9]{4}", emi)

                    if check_vig and check_emi:
                        print("vigencia: ", vig)
                        print("emisión: ", emi)
                        vigencia = vig
                        emisión = emi
                    
                    
                    # check if sexo
                    if "sex" in text_w_n_blanks.lower():
                        sexo = text.split(" ")[-1]
                        print("sexo check", sexo)
                        if "m" in sexo.lower():
                            sexo = "M"
                            print("sexo: ",sexo)
                        else:
                            sexo = "H"
                            print("sexo: ",sexo)
                    #print("sexo: ", sexo)

                    # check if municipio
                    if "munici" in text_w_n_blanks.lower():
                        municipio = text.split(" ")[-1]
                        print("municipio: ", municipio)
                    
                    if "esta" in text_w_n_blanks.lower():
                        estado = text.split(" ")[-1]
                        print("estado: ", estado)

                    #print("debug", text_w_n_blanks)

                    # all text is lowercase
                    text_w_n_blanks = text_w_n_blanks.lower()


                    #print(text_w_n_blanks)
                
                
            print(DoB_point, DoB_text)
            name_dicts = dict(zip(textd, top_l_points))
            #print("name_dicts: ", name_dicts)
            #print("DoB_point:", DoB_point)
            
            for k, v in name_dicts.copy().items():
                if v == tuple(DoB_point):
                    #print(k ,"value deleted")
                    del name_dicts[k]

            
            top_l_points.remove(tuple(DoB_point))
            ## gets the nearest y coordinate initial bounding point

            name_dicts0= {k:tuple(map(lambda i, j: 
                i - j, v, tuple(DoB_point))) for k, v in name_dicts.items() }
            #print(name_dicts0)

            for x,y in top_l_points:
                if y < y_b+(0.015*height) and y > y_b-(0.015*height) :
                # if y < y_b+15 and y > y_b-15 :
                    NamePoint = x,y

            #print(NamePoint)    
            distances_list = []
            for point in top_l_points:
                distances_list.append(distance(point, NamePoint))

            #print( distances_list)

            for k, v in name_dicts.copy().items():  #  (for Python 2.x)
                if v == NamePoint:
                    PrimerApellido = k
                    #print("Primer apellido", k)

            name_dicts2= {k:tuple(map(lambda i, j: 
                i - j, v, NamePoint)) for k, v in name_dicts.items() }
            #print(name_dicts2)
            
            
            dist_dict = {k:distance((0,0),v) for k,v  in name_dicts2.items()}
            #print(dist_dict)
            
            sorted_dist_dict = {k: v for k, v in sorted(dist_dict.items(),
                key=lambda item: item[1])}
            #print(sorted_dist_dict)
            

            ## get the next two items (they are in ordered by the tuple)
            ## and should be the next two bounding boxes
            names_list= list(sorted_dist_dict.keys())[:5]
            names_list = [name for name in names_list 
                if "DOMICI" not in name]
            names_list = [name for name in names_list 
                if "NOM" not in name]
            names_list = [name for name in names_list 
                if "CREDENCIAL" not in name]
            
            Domicilio_list= list(sorted_dist_dict.keys())[5:10]
            #print(Domicilio_list)
            Domicilio_list = [name for name in Domicilio_list 
                if "DOMICI" not in name]
            Domicilio_list = [name for name in Domicilio_list 
                if "MÉXICO" not in name]
            Domicilio_list = [name for name in Domicilio_list 
                if "CREDENCIAL" not in name]
            Domicilio_list = [name for name in Domicilio_list 
                if "ELECTOR" not in name]
            Domicilio_list = [name for name in Domicilio_list 
                if "CLAVE" not in name]
            Domicilio_list = [name for name in Domicilio_list 
                if "cur" not in name]

            domicilio_list_str = ' '.join([str(elem) for elem in Domicilio_list])


            #print("names_list: ",names_list)

            names_list_str = ' '.join([str(elem) for elem in names_list])
            print()
            print("Nombre completo: ", names_list_str)
            print("Domicilio completo: ", domicilio_list_str)
            #print("Fecha de nacimiento:", DoB_text)

        # Save result image
        
        cv2.imwrite(res_img_file, img)
        return {"nombre": names_list_str, "fecha_de_nacimiento":DoB_text.strip(),
            "sexo": sexo, "domicilio":domicilio_list_str,"clave_de_elector": clave.strip(),
            "CURP": curp.strip(), "registro":registro.strip(), "numero_de_emisión":emisión,
            "estado": estado.strip(), "municipio": municipio.strip(), "vigencia":vigencia}
            #,  "seccion": seccion}


