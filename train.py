import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
import numpy as np

# --------------------
# CONFIG
# --------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

# --------------------
# DATASET
# --------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "flowers",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Flower classes:", class_names)

# --------------------
# CLASS WEIGHTS (CRITICAL)
# --------------------
counts = np.zeros(num_classes)
for _, labels in train_ds:
    for l in labels:
        counts[l] += 1

total = np.sum(counts)
class_weights = {i: total / (num_classes * counts[i]) for i in range(num_classes)}
print("Class weights:", class_weights)

# --------------------
# PERFORMANCE
# --------------------
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# --------------------
# MODEL
# --------------------
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation="softmax")
])

# --------------------
# TRAIN
# --------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# --------------------
# SAVE
# --------------------
model.save("flower_model")
print("✅ FINAL flower model saved")
