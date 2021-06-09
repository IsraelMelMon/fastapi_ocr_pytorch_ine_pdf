import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import cv2
import numpy as np
from PIL import Image
from application.components import predict, read_imagefile
from application.schema import Symptom
from application.components.prediction import symptom_check
import os

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>
<h2>Try INE type D data extraction api - demo</h2>
<br>by Israel Melendez Montoya, Hernan Martinez Sanchez and Moni"""

app = FastAPI(title='Pytorch FastAPI INE extraction', description=app_desc)


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")


@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    print(type(image))
    image.save("test2.png")
    #pic = cv2.imwrite("test/input_test.png", image)
    #image = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    #print(image.shape)
    #image = Image.fromarray(image, 'RGB')
    #image.save('test.png')
    #cv2.imwrite("test/output.png",image)
    print(os.getcwd)
    image_path = "/Users/israel/tensorflow-fastapi-starter-pack/test2.png"
    prediction = predict(image_path)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app, debug=True, reload=True)#, port=5050)
