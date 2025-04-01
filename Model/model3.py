import os
import tensorflow as tf
import datetime
import time
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Check GPU Availability
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    device = "GPU"
    print("Using GPU for training")
else:
    device = "CPU"
    print("GPU not found, using CPU for training")

# Define dataset path
DATASET_PATH = "C:\\Users\\LIKITH S G\\Desktop\\Sign Language Detection\\Data"
SELECTED_CLASSES = [
    "Hello_Augmented", "Yes_Augmented", "No_Augmented", "ILoveYou_Augmented",
    "Okay_Augmented", "Please_Augmented", "ThankYou_Augmented"
]

# Parameters
IMG_SIZE = (224, 224)  # MobileNetV3 requires 224x224 input
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.005
VALIDATION_SPLIT = 0.3  # 70% train, 30% test

# Logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)

# Data Loading
train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES,
    class_mode='categorical',
    subset='training'
)

test_generator = test_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=SELECTED_CLASSES,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained MobileNetV3
base_model = MobileNetV3Large(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze initial layers to retain pre-trained features
for layer in base_model.layers:
    layer.trainable = False

# Custom Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Instead of Flatten() to avoid shape mismatch
    Dropout(0.5),
    Dense(len(SELECTED_CLASSES), activation='softmax')
])

# Compile Model
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
start_time = time.time()
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[tensorboard_callback]
)
total_time = time.time() - start_time

# Save Model
MODEL_DIR = "C:\\Users\\LIKITH S G\\Desktop\\Sign Language Detection\\Model"
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "mobilenetv3_sign_language_model.keras"))

# Print Metrics
train_loss, train_acc = model.evaluate(train_generator)
test_loss, test_acc = model.evaluate(test_generator)

print(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
print(f"Total Training Time: {total_time:.2f} seconds")

# Save Logs
log_file = os.path.join(MODEL_DIR, "training_log.txt")
with open(log_file, "w") as f:
    f.write(f"Device Used: {device}\n")
    f.write(f"Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}\n")
    f.write(f"Total Training Time: {total_time:.2f} seconds\n")
    f.write(f"Logs stored at: {log_dir}\n")
