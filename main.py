from pathlib import Path
from typing import Tuple

import keras_cv
import keras_tuner as kt
import splitfolders
import tensorflow as tf
from keras import layers
from keras.applications import VGG16, VGG19, ConvNeXtXLarge, EfficientNetV2B0, InceptionV3, ResNet50V2
from matplotlib import pyplot as plt
from tensorflow import keras

IMG_WIDTH = 224
IMG_HEIGHT = 224
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)
NUM_CLASSES = 6
EPOCHS = 20


def split_data() -> None:
    # https://pypi.org/project/split-folders/

    """Splits the given dataset in train, validation, test folders (60:20:20 ratio), inside a output folder.
    Using split-folders library."""

    splitfolders.ratio(
        Path(__file__).resolve().parent / "REI-Dataset_", output="outputs", seed=42, ratio=(0.6, 0.2, 0.2)
    )


def prefetch_augment_data(randaug: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Fetch and augment the train, validation, test sets in TensorFlow dataset format.
    Perform caching for improved performance.

    Args:
        randaug (bool, optional): Enable RandAug augmentations NOTE: Reduces accuracy considerably. Defaults to False.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: train, validation, test sets
                                                                    in TensorFlow dataset format.
    """

    # https://keras.io/examples/vision/image_classification_from_scratch/

    # Paths to the training and testing datasets.
    # Assumes the data is already split into train and test folders using split_data() method.
    train_path = Path(__file__).resolve().parent / "output/train"
    val_path = Path(__file__).resolve().parent / "output/val"
    test_path = Path(__file__).resolve().parent / "output/test"

    # Load training dataset as TensorFlow datasets.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        seed=42,
        batch_size=32,
    )

    # Load validation dataset as TensorFlow dataset.
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_path,
        seed=42,
        batch_size=32,
    )

    # Load testing dataset as TensorFlow dataset.
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        seed=42,
        batch_size=32,
    )

    # Augmentations
    # Resizes images into a uniform dimension and rescales pixel values from [0, 255] to [0, 1] range.
    resize_and_rescale = keras.Sequential([layers.Resizing(IMG_HEIGHT, IMG_WIDTH), layers.Rescaling(1.0 / 255)])

    # Perform random image augmentations
    # https://www.tensorflow.org/tutorials/images/data_augmentation
    # https://keras.io/api/layers/preprocessing_layers/image_augmentation/

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(factor=0.5),
            layers.RandomCrop(180, 180),
            layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
            layers.GaussianNoise(stddev=0.2),
            layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            layers.RandomContrast(factor=0.4),
            layers.RandomBrightness(factor=0.4),
        ]
    )

    # Image Histogram equalization
    eql = keras_cv.layers.Equalization(value_range=(0, 22), bins=20)

    # If specified, use RandAugment
    # https://keras.io/examples/vision/randaugment/
    if randaug is True:
        rand_augment = keras_cv.layers.RandAugment(value_range=(0, 1), augmentations_per_image=3, magnitude=0.5)
        train_ds = train_ds.map(lambda x, y: (rand_augment(x, training=True), y))

    # Apply the data augmentation pipeline to the training dataset.
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (eql(x, training=True), y))

    # Apply only the resize and rescale layer to the validation and testing datasets.
    val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y))
    test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), y))

    # https://www.tensorflow.org/tutorials/images/classification#configure_the_dataset_for_performance
    # https://www.tensorflow.org/guide/data_performance
    # Cache and prefetch the training, validation, and testing datasets to speed up and optimise future operations.
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


def lenet_5_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise LeNet-5 CNN model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """

    # https://towardsdatascience.com/understanding-and-implementing-lenet-5-cnn-architecture-deep-learning-a2d531ebc342

    lenet_5_model = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                6, kernel_size=5, strides=1, activation="tanh", input_shape=INPUT_SHAPE, padding="same"
            ),  # C1
            keras.layers.AveragePooling2D(),  # S2
            keras.layers.Conv2D(16, kernel_size=5, strides=1, activation="tanh", padding="valid"),  # C3
            keras.layers.AveragePooling2D(),  # S4
            keras.layers.Conv2D(120, kernel_size=5, strides=1, activation="tanh", padding="valid"),  # C5
            keras.layers.Flatten(),  # Flatten
            keras.layers.Dense(84, activation="tanh"),  # F6
            keras.layers.Dense(NUM_CLASSES, activation="softmax"),  # Output layer
        ]
    )
    # Compile the model
    lenet_5_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping callback
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, mode="max", verbose=1)

    # Train the model with early stopping
    lenet_5_model_history = lenet_5_model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("LeNet 5 accuracy on Test data:")
    lenet_5_model.evaluate(test_ds)
    return lenet_5_model_history


def vgg16_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise VGG16 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """

    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    vgg16_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("VGG16 accuracy on Test data:")
    model.evaluate(test_ds)
    return vgg16_model_history


def vgg19_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise VGG19 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """

    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = VGG19(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    vgg19_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("VGG19 accuracy on Test data:")
    model.evaluate(test_ds)
    return vgg19_model_history


def resnet50v2_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise VGG16 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """
    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    resnet50v2_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("ResNet50V2 accuracy on Test data:")
    model.evaluate(test_ds)
    return resnet50v2_model_history


def inceptionv3_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise VGG16 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """
    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    inceptionv3_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("InceptionV3 accuracy on Test data:")
    model.evaluate(test_ds)
    return inceptionv3_model_history


def convnextxlarge_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise VGG16 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """
    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = ConvNeXtXLarge(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    convnextxlarge_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("ConvNeXtXLarge accuracy on Test data:")
    model.evaluate(test_ds)
    return convnextxlarge_model_history


def efficientnetv2b0_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise EfficientNetV2B0 model with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """
    # https://keras.io/guides/transfer_learning/#build-a-model

    # Load the pre-trained model
    base_model = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Setting up the input layer according to the input shape of data
    inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

    # Passing input layer to custom classification layers
    x = base_model(inputs)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    # Setting up the output layer
    outputs = keras.layers.Dense(6, activation="softmax")(x)

    # Create the final model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    # Define early stopping
    callback = keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", verbose=1, patience=4)

    # Train the model with early stopping
    efficientnetv2b0_model_history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[callback])
    print("EfficientNetV2B0 accuracy on Test data:")
    model.evaluate(test_ds)
    return efficientnetv2b0_model_history


def resnet50v2_tuner_model(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> tf.keras.callbacks.History:
    """Initialise ResNet50V2 model, and perform automated hyparameter tuning with callbacks

    Args:
        train_ds (tf.data.Dataset): train set
        val_ds (tf.data.Dataset): validation set
        test_ds (tf.data.Dataset): test set

    Returns:
        tf.keras.callbacks.History: History variable for metrics
    """
    # https://keras.io/guides/transfer_learning/#build-a-model
    # https://www.tensorflow.org/tutorials/keras/keras_tuner

    def model_builder(hp: kt.engine.hyperparameters.HyperParameters) -> keras.Sequential:
        """Build Model for hyperparameter tuning, with defined parameter set

        Args:
            hp (kt.engine.hyperparameters.HyperParameters): Keras tuner hypermater object

        Returns:
            keras.Sequential: Keras model
        """
        # Load the pre-trained ResNet50V2 model
        base_model = ResNet50V2(weights="imagenet", include_top=False, input_shape=INPUT_SHAPE)

        # Freeze the pre-trained layers
        for layer in base_model.layers:
            layer.trainable = False

        # Setting up the input layer according to the input shape of data
        inputs = keras.layers.Input(shape=INPUT_SHAPE, name="input_shape")

        # Passing input layer to custom classification layers
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x)
        hp_units = hp.Int("units", min_value=32, max_value=512, step=32)
        x = keras.layers.Dense(units=hp_units, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(0.5)(x)

        # Setting up the output layer
        outputs = keras.layers.Dense(6, activation="softmax")(x)

        # Create the final model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-4, 1e-5])

        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model

    tuner = kt.Hyperband(
        model_builder,
        objective="val_accuracy",
        max_epochs=10,
    )
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4)
    tuner.search(train_ds, validation_data=val_ds, epochs=20, callbacks=[callback])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best number of units: ", (best_hps.get("units")))
    print("Best learning rate for the optimizer: ", (best_hps.get("learning_rate")))

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(train_ds, validation_data=val_ds, epochs=15)

    val_acc_per_epoch = history.history["val_accuracy"]
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print("Best epoch: %d" % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    resnet50v2_tuner_model_history = hypermodel.fit(train_ds, epochs=best_epoch, validation_data=val_ds)

    print("Hyperparameter tuned ResNet50V2 accuracy on Test data:")
    hypermodel.evaluate(test_ds)
    return resnet50v2_tuner_model_history


def plot_acc_loss_graph(history: tf.keras.callbacks.History, file_name: str) -> None:
    """Plot and save the accuracy and loss graphs for given model history.

    Args:
        history (tf.keras.callbacks.History): History variable for metrics
        file_name (str): Name of file to saved
    """

    # Plot train and val accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.savefig(Path(__file__).resolve().parent / "figs" / (file_name + "acc.png"))


if __name__ == "__main__":

    # Split data in train, val, test sets
    split_data()

    # Fecth and Augment data in TensorFlow data format
    train_ds, val_ds, test_ds = prefetch_augment_data()

    # Evaluate all models
    lenet5_model_history = lenet_5_model(train_ds, val_ds, test_ds)
    vgg16_model_history = vgg16_model(train_ds, val_ds, test_ds)
    vgg19_model_history = vgg16_model(train_ds, val_ds, test_ds)
    resnet50v2_model_history = resnet50v2_model(train_ds, val_ds, test_ds)
    inceptionv3_model_history = inceptionv3_model(train_ds, val_ds, test_ds)
    convnextxlarge_model_history = convnextxlarge_model(train_ds, val_ds, test_ds)
    efficientnetv2b0_model_history = efficientnetv2b0_model(train_ds, val_ds, test_ds)
    resnet50v2_tuner_model_history = resnet50v2_tuner_model(train_ds, val_ds, test_ds)

    # Plot accuracy graphs for train and val accuracies for all models
    plot_acc_loss_graph(lenet5_model_history, "lenet")
    plot_acc_loss_graph(vgg16_model_history, "vgg16")
    plot_acc_loss_graph(vgg19_model_history, "vgg19")
    plot_acc_loss_graph(resnet50v2_model_history, "rn50")
    plot_acc_loss_graph(inceptionv3_model_history, "inc")
    plot_acc_loss_graph(convnextxlarge_model_history, "xtx")
    plot_acc_loss_graph(resnet50v2_tuner_model_history, "rn50hp")
