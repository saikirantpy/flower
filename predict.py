import tensorflow as tf
import numpy as np
from PIL import Image
import glob

model = tf.keras.models.load_model("flower_model")
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Pick first rose image automatically
img_path = glob.glob("flowers/rose/*.jpg")[0]
img = Image.open(img_path).resize((224, 224))

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print("🌸 Image:", img_path)
print("🌸 Predicted flower:", class_names[np.argmax(pred)])
