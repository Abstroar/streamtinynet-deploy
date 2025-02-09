import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, Flatten, Dense, SeparableConv2D, Input
from tensorflow.keras.models import Model
import numpy as np



def build_spatial_extractor(input_shape):
    inputs = Input(shape=input_shape)
    x = SeparableConv2D(16, (3, 3), padding="same")(inputs)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = SeparableConv2D(32, (3, 3), padding="same")(x)
    x = ReLU()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    model = Model(inputs, x, name="SpatialFeatureExtractor")
    return model


# Temporal Processing (h)
def build_temporal_processor(input_shape):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation="relu")(inputs)
    x = Dense(32, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)  # Assuming 10 classes
    model = Model(inputs, x, name="TemporalProcessor")
    return model


def build_streamtinynet(frame_shape, sequence_length):
    spatial_extractor = build_spatial_extractor(frame_shape)

    # Process multiple frames
    sequence_input = Input(shape=(sequence_length, *frame_shape))
    extracted_features = tf.keras.layers.TimeDistributed(spatial_extractor)(sequence_input)
    combined_features = Flatten()(extracted_features)

    temporal_processor = build_temporal_processor(combined_features.shape[1:])
    outputs = temporal_processor(combined_features)

    model = Model(sequence_input, outputs, name="StreamTinyNet")
    return model


# Example Usage
frame_shape = (64, 64, 3)  # Small resolution for TinyML
sequence_length = 5  # Using 5 frames per sequence
model = build_streamtinynet(frame_shape, sequence_length)
model.summary()

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open("streamtinynet.tflite", "wb") as f:
    f.write(tflite_model)
