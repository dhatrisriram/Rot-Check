import os
import tensorflow as tf

# Dataset directories
DATASET_DIR = r'Dataset'

# Image Data Generator for training
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,      # Increased from 20 for more variety
    width_shift_range=0.3,   # Increased from 0.2
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2], # Added to help with lighting differences
    horizontal_flip=True,
    vertical_flip=True,      # Added because carrots can be in any orientation
    fill_mode='nearest',
    validation_split=0.2     # Reserved 20% of data for testing
)

train_generator = train_data_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_data_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Set as validation data
)

# Build the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with validation monitoring
model.fit(
    train_generator,
    validation_data=validation_generator, # Use the new validation generator
    epochs=15,
    class_weight={0: 1.0, 1: 4.0}         # Handling class imbalance
)

# Save the model
model.save('carrot_classifier_model.h5')
print("Model training complete and saved as carrot_classifier_model.h5")


