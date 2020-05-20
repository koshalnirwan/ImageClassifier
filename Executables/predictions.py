#from final_model import *
import cv2
import keras

CATEGORY = ["Cat","Dog"]

def prepare(filepath):
    IMG_SIZE = 70
    
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = keras.models.load_model("myCNN.model")


prediction = model.predict([prepare('dog.jpg')])    # should be list even if predicting for 1 item
print(CATEGORY[int(prediction[0][0])])