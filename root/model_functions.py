import numpy as np
import tensorflow as tf
import PIL
import io

model = tf.keras.models.load_model("root/mask_detector.model")
#model.summary()
labels = ["Face with Mask", "Face without Mask"]
# TODO: Segmentar rostros en una imagen y devolver un array con los rostros

def read_imagefile(file) -> PIL.Image.Image:
    image = PIL.Image.open(io.BytesIO(file))
    return image

def process_image_face(image):
    #image = PIL.Image.open("face_with_mask.jpg")
    image = image.resize((224, 224))
    face = np.asarray(image)
    face = tf.keras.applications.mobilenet.preprocess_input(face)
    face = np.expand_dims(face, axis=0)
    return face
def predict_face(face):
    prediction = model.predict(face)[0]
    label_idx = np.argmax(prediction)
    label = labels[label_idx]
    result = {"result":label,
              "accuracy": round(prediction[label_idx]*100,2)}
    return result
