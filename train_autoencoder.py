from signal import SIGINT, getsignal, signal
import keras
from keras import layers, callbacks
import cv2
from matplotlib.pylab import Axes
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dataset import load_dataset
from methods import Label, LabeledDataset, Split
from train import stop

stop = False

def test(split: Split):
    autoencoder: keras.Model = keras.saving.load_model("models/autoencoder-best.keras")
    encoder = keras.Model(autoencoder.input, autoencoder.layers[-16].output)
    predictions = encoder.predict(split.train.data)
    decoded = autoencoder.predict(split.train.data)
    encoded = np.vstack([np.reshape(image, (1,16,16)) for image in predictions])
    _, ax = plt.subplots(3,3, constrained_layout=True)
    for i, label in enumerate(Label):
        ax[0][i].imshow(split.train[label][0])
        ax[1][i].imshow(encoded[split.train._label_ranges[label].start])
        ax[2][i].imshow(decoded[split.train._label_ranges[label].start])
        ax[0][i].set_title(label.name)
    ax[0][0].set_ylabel("Original")
    ax[1][0].set_ylabel("Encoded")
    ax[2][0].set_ylabel("Decoded")
    plt.show()
    
def encode_dataset(dataset: LabeledDataset):
    autoencoder = keras.saving.load_model("models/autoencoder-best.keras")
    encoder = keras.Model(autoencoder.input, autoencoder.layers[-16].output)
    return encoder.predict(dataset.data).reshape((-1,256))

def main():
    original_handler = getsignal(SIGINT)
    # First Ctrl+C will exit loop after ending current iterations, second one will use default handler
    def stop_handler(_, _2):
        global stop
        stop = True
        signal(SIGINT, original_handler)
    signal(SIGINT, stop_handler)
    plt.show(block=False)
    split = load_dataset(recreate=True)[1]
    test(split)

class TerminateOnFlag(callbacks.Callback):
    """Callback that terminates training when flag=1 is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        global stop
        if stop == True:
            self.model.stop_training = True

def draw_images(original_dataset: LabeledDataset, decoded_images, axes):
    rng = np.random.default_rng(seed=2222)
    label_ranges = original_dataset._label_ranges
    random_indices = { label: rng.choice(np.arange(label_ranges[label].start, label_ranges[label].end), 5) for label in Label }
    for row, label in enumerate(Label):
        for col, rng_index in enumerate(random_indices[label]):
            ax = axes[row][col]
            ax[0].clear()
            ax[1].clear()
            ax[0].set_axis_off()
            ax[1].set_axis_off()
            ax[0].set_title(label.name)
            ax[0].imshow(original_dataset[rng_index])
            ax[1].imshow(decoded_images[rng_index])
    plt.show(block=False)
    plt.pause(1)
    return
    for label in Label:
            fig = plt.figure()
            fig.suptitle(label.name)
            for i in range(1, n):
                ax = plt.subplot(1, n, i + 1)
                plt.imshow(encoded_imgs[split.val._label_ranges[label].start + i].reshape((16, -1)).T)
                #plt.gray()
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

def train_autoencoder(split: Split):
    # This is our input image
    # resized = np.empty(shape=(len(split.train.data), 256, 256), dtype=np.float64)
    # for i, image in enumerate(split.train.data):
    #     resized[i] = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    # split.train.data = resized
    # resized = np.empty(shape=(len(split.val.data), 256, 256), dtype=np.float64)
    # for i, image in enumerate(split.val.data):
    #     resized[i] = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    # split.val.data = resized

    shape = (*split.train.data[0].shape,1)
    input_img = keras.Input(shape=shape)

    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Flatten()(x)

    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    fig, axes = plt.subplots(3,5)
    fig.set_size_inches(15,15)
    plt.axis("off")
    for row in range(3):
        for col in range(5):
            ax: Axes = axes[row][col]
            divider = make_axes_locatable(ax)
            axes[row][col] = [ax, divider.append_axes("bottom", size="100%", pad=0.1)]
            
    print(encoded)
    fig.suptitle(f"Encoded layer shape: {encoded.shape}")
    
    i = 0
    while not stop:
        autoencoder.fit(split.train.data, split.train.data,
                    epochs=5 + i,
                    initial_epoch=i,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(split.val.data, split.val.data),
                    callbacks=[callbacks.TensorBoard(log_dir='/tmp/autoencoder'), TerminateOnFlag()])
        i += 5
        decoded_imgs = autoencoder.predict(split.val.data)
        #encoder = keras.Model(input_img, encoded)
        #encoded_imgs = encoder.predict(split.val.data)

        fig.subplots_adjust(hspace=0.4)
        draw_images(split.val, decoded_imgs, axes)

    autoencoder.save("models/autoencoder.keras")
        

if __name__ == "__main__":
    main()
    plt.show()
