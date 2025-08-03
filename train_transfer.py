from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os

# Constants
img_size = (224, 224)
batch_size = 16
epochs = 15  # Increase epochs for better learning

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

train = datagen.flow_from_directory(
    'Dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val = datagen.flow_from_directory(
    'Dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load Pretrained ResNet50 Base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base initially

# Custom Layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(train.num_classes, activation='softmax')
])

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = callbacks.EarlyStopping(patience=4, restore_best_weights=True)

# Train Phase 1
history = model.fit(
    train,
    validation_data=val,
    epochs=epochs,
    callbacks=[early_stop]
)

# Fine-tune the base model
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

# Train Phase 2 (fine-tuning)
history_finetune = model.fit(
    train,
    validation_data=val,
    epochs=5,
    callbacks=[early_stop]
)

# Save the final model
os.makedirs("models", exist_ok=True)
model.save("models/resnet_fish_model.h5")
print("✅ Final ResNet model saved!")
