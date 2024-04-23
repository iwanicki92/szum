from datetime import datetime
from itertools import repeat
from pathlib import Path
import time
import joblib
from matplotlib.pylab import Axes
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from signal import SIGINT, signal, getsignal

from dataset import load_dataset
from methods import Label, Split

stop = False

def plot_model(ax1, ax2, scores, iteration):
    ax1.clear()
    ax2.clear()
    for score_legend, score in scores.items():
        if 'loss' in score_legend:
            ax2.plot(range(1, iteration + 2), score, label=score_legend, color="red" if 'train' in score_legend else "green")
        else:
            ax1.plot(range(1, iteration + 2), score, label=score_legend)
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax1.set_label("accuracy")
    ax2.set_label("loss")
    plt.pause(0.2)

def train(split: Split):
    print("Training")
    model_path = Path("models") / str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    model_path.mkdir(parents=True)
    statistics_path = model_path / 'stats.csv'
    statistics_path.write_text("iteration,train loss,val loss,train accuracy,val accuracy,total time\n", encoding="utf-8")
    for set_str, set in zip(split._fields, split):
        shape = set.data.shape
        set.data = set.data.reshape(len(set), -1)
        print(f"Reshaping {set_str} set from {shape} to {set.data.shape}")
    
    scores = {'train loss': [], 'val loss': [], 'train accuracy': [], 'val accuracy': []}
    iterations = 10000
    hidden_layers = (100,100)
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=5000, max_iter=10)

    train_y = [
        y
        for label in Label
        for y in repeat(label, len(split.train[label]))
        ]
    
    val_y = [
        y
        for label in Label
        for y in repeat(label, len(split.val[label]))
        ]

    ax1: Axes
    ax2: Axes
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    best_loss: float = float('inf')
    train_start = time.time()

    for iteration in range(iterations):
        if stop:
            break
        classifier.warm_start = iteration > 0
        start = time.time()
        classifier.fit(split.train, train_y)
        print(f"Iteration time: {time.time() - start}s")

        y_pred_train = classifier.predict(split.train)
        y_pred_prob_train = classifier.predict_proba(split.train)
        y_pred_val = classifier.predict(split.val)
        y_pred_prob_val = classifier.predict_proba(split.val)
        scores['train loss'].append(metrics.log_loss(train_y, y_pred_prob_train))
        scores['val loss'].append(metrics.log_loss(val_y, y_pred_prob_val))
        scores['train accuracy'].append(metrics.accuracy_score(train_y, y_pred_train))
        scores['val accuracy'].append(metrics.accuracy_score(val_y, y_pred_val))

        plot_model(ax1, ax2, scores, iteration)
        if scores['val loss'][-1] < best_loss:
            joblib.dump(classifier, Path(model_path) / "best")
            best_loss = scores['val loss'][-1]
            print(f"{best_loss=}")
        with statistics_path.open("a", encoding="utf-8") as file:
            stats = ','.join([
                str(iteration), str(scores['train loss'][-1]), str(scores['val loss'][-1]), 
                str(scores['train accuracy'][-1]), str(scores['val accuracy'][-1]), str(time.time() - train_start)
            ])
            file.write(f"{stats}\n")
    
    print("Training complete")

def main():
    splits = load_dataset(recreate=True)
    split_1 = splits[1]
    del splits
    import gc
    gc.collect()

    train(split_1)

import keras
from keras import layers, regularizers, callbacks
import cv2

class TerminateOnFlag(callbacks.Callback):
    """Callback that terminates training when flag=1 is encountered.
    """

    def on_batch_end(self, batch, logs=None):
        if stop == True:    
            self.model.stop_training = True

def train_autoencoder():
    split_1 = load_dataset()[1]

    # This is the size of our encoded representations

    # This is our input image
    resized = np.empty(shape=(len(split_1.train.data), 256, 256), dtype=np.float64)
    for i, image in enumerate(split_1.train.data):
        resized[i] = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    split_1.train.data = resized
    resized = np.empty(shape=(len(split_1.val.data), 256, 256), dtype=np.float64)
    for i, image in enumerate(split_1.val.data):
        resized[i] = cv2.resize(image, (256,256), interpolation=cv2.INTER_AREA)
    split_1.val.data = resized

    shape = (*split_1.train.data[0].shape,1)
    print(shape)
    input_img = keras.Input(shape=shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = layers.Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = keras.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.summary()

    autoencoder.fit(split_1.train.data, split_1.train.data,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_data=(split_1.val.data, split_1.val.data),
                callbacks=[callbacks.TensorBoard(log_dir='/tmp/autoencoder'), TerminateOnFlag()])

    decoded_imgs = autoencoder.predict(split_1.val.data)
    encoder = keras.Model(input_img, encoded)
    encoded_imgs = encoder.predict(split_1.val.data)

    n = 10  # How many digits we will display
    for label in Label:
        plt.figure(figsize=shape[:-1])
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(split_1.val[split_1.val._label_ranges[label].start + i])
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[split_1.val._label_ranges[label].start + i])
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show(block=False)
    for label in Label:
        fig = plt.figure()
        fig.suptitle(label.name)
        for i in range(1, n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(encoded_imgs[split_1.val._label_ranges[label].start + i].reshape((16, -1)).T)
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

if __name__ == "__main__":
    original_handler = getsignal(SIGINT)
    # First Ctrl+C will exit loop after ending current iterations, second one will use default handler
    def stop_handler(_, _2):
        global stop
        stop = True
        signal(SIGINT, original_handler)
    signal(SIGINT, stop_handler)
    plt.show(block=False)
    train_autoencoder()
    #main()
    plt.show()