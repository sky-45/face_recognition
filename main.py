from fastapi import FastAPI, File, UploadFile
from root import model_functions
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
db = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = model_functions.read_imagefile(await file.read())
    face = model_functions.process_image_face(image)
    result = model_functions.predict_face(face)
    return result