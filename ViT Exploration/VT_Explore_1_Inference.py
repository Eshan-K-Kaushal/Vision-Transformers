img_size = 72
patch_size = 18
num_patches = (img_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
num_classes = 100
input_shape = (32, 32, 3)

str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
              'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
              'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
              'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
              'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail',
              'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
              'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train',
              'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
              'woman', 'worm']
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import load_model
from keras.preprocessing import *
from keras.datasets import cifar100
import cv2
import matplotlib.pyplot as plt
import numpy as np
dict_labels = {}
for i in range(0, len(str_labels)):
    dict_labels[i] = str_labels[i]
print(dict_labels)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
i = 24
label = y_train[i][0]
print(dict_labels[label])
print(label)
img = x_train[i]
img = img.reshape(32, 32, 3)
print(img.shape, type(img))
#plt.imshow(image.asype("uint8"))
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()





# MODEL
learning_rate = 0.001
weight_decay = 0.0001

batch_size = 32
num_epochs = 4
image_size = 72 # upsample the image using the functions that will come further
patch_size = 18 # size of a patch - we are taking 25 by 25 patches
num_patches = (image_size // patch_size) ** 2 # to find the number of patches and this gives a number to all the patches and this will help with the positional encoding
projection_dim = 64 # the patches from an image will be projected on to these dims so that feature vectors can be created
num_heads = 4 # no. of attention heads (how many times a attention will be calculated) and in the end the weighted average is taken
transformer_units = [
    projection_dim * 2,
    projection_dim

] # size of transformer layers

transformer_layers = 8
mlp_head_units = [2048, 1024] # size of the hidden layer

data_augmentation = keras.Sequential(
    [
        tf.keras.layers.Normalization(),
        tf.keras.layers.Resizing(image_size, image_size),
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(factor=0.02),
        tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)

data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection": self.projection,
            "position_embedding": self.position_embedding
        })
        return config

# encoded is the final vector for every patch that has the (embedding+positional_embeddings)

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "checkpoint/"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    # history = model.fit(
    #     x=x_train,
    #     y=y_train,
    #     batch_size=batch_size,
    #     epochs=num_epochs,
    #     validation_split=0.1,
    #     callbacks=[checkpoint_callback],
    # )
    #model.save('Saved_CiFAR_ViT.h5', history)
    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    # return history

checkpoint_filepath = "checkpoint/"

def inference(model, img):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )
    model.load_weights(checkpoint_filepath)
    pred = np.argmax(model.predict(img))
    print('predicted index: ', pred)
    #label = y_train[pred][0]
    #print('label from y', label)
    return pred, dict_labels[label]

test_img = x_train[24]
test_img = test_img.reshape(32, 32, 3)
cv2.imshow('test_img', test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(test_img.shape)
resized_image = tf.image.resize(
    tf.convert_to_tensor([test_img]), size=(32, 32)
)
print(resized_image.shape)

vit_classifier = create_vit_classifier()
inference_engine = inference(vit_classifier, resized_image)
print(inference_engine[0], inference_engine[1])
#print(history)

#test_img = x_train[np.random.choice(range(x_train.shape[0]))]
#patches = Patches(patch_size)(resized_image)

