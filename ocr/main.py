from io import BytesIO
from text_extraction import grouping_image
from PIL import Image
import numpy as np

# grouping_image('./test_image.jpeg')

from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}
# 이미지로부터 텍스트 추
@app.post("/convert")
async def convert(file: UploadFile):
    image = np.array(Image.open(BytesIO(await file.read())))
    grouping_image(image)

    return {}
