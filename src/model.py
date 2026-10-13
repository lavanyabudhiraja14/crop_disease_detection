"""
Model construction utilities.
Builds a MobileNetV2-based transfer-learning classifier.
"""

import tensorflow as tf


def build_model(input_shape=(224, 224, 3), num_classes=1, learning_rate=0.001):
    """
    Builds and compiles a MobileNetV2 transfer-learning model.

    The model includes:
      - Data augmentation layers (active only during training)
      - MobileNetV2 preprocess_input (scales pixels to [-1, 1])
      - Frozen MobileNetV2 feature extractor
      - GlobalAveragePooling → Dropout → Dense(softmax)

    Image inputs should be float32 in [0, 255].
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    # Data augmentation (applied only during training)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ], name='data_augmentation')

    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
