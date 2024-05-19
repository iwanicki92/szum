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
from methods import Label, LabeledDataset, Split

stop = False

def show3(dataset, reshape: None | tuple = None):
    _, ax = plt.subplots(1,3, constrained_layout=True)
    for i, label in enumerate(Label):
        if reshape is None:
            img = dataset[label][0]
        else:
            print(dataset[label][0].shape)
            img = np.reshape(dataset[label][0], reshape)
        ax[i].imshow(img)
        ax[i].set_title(label.name)
    plt.show()

def plot_model(ax1, ax2, scores, iteration):
    ax1.clear()
    ax2.clear()
    for score_legend, score in scores.items():
        if 'loss' in score_legend:
            ax2.plot(range(1, iteration + 2), score, label=score_legend, color="red" if 'train' in score_legend else "green")
        else:
            ax1.plot(range(1, iteration + 2), score, label=score_legend)
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    ax1.set_label("accuracy")
    ax2.set_label("loss")
    plt.pause(0.2)

def train(split: Split):
    print("Training")
    from train_autoencoder import encode_dataset
    training_encoded = encode_dataset(split.train)
    val_encoded = encode_dataset(split.val)
    model_dir = str(datetime.now().strftime("%Y-%m-%d %H_%M_%S"))
    model_path = Path("models") / model_dir
    model_path.mkdir(parents=True)
    symlink = Path(Path("models") / "last")
    symlink.unlink(True)
    symlink.symlink_to(model_dir, True)
    statistics_path = model_path / 'stats.csv'
    statistics_path.write_text("iteration,train loss,val loss,train accuracy,val accuracy,total time\n", encoding="utf-8")
    for set_str, set in zip(split._fields, split):
        shape = set.data.shape
        set.data = set.data.reshape(len(set), -1)
        print(f"Reshaping {set_str} set from {shape} to {set.data.shape}")
    
    scores = {'train loss': [], 'val loss': [], 'train accuracy': [], 'val accuracy': []}
    iterations = 10000
    hidden_layers = (10)
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=5000, max_iter=1)

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
    fig.suptitle(f"hidden layers={hidden_layers}")
    best_loss: float = float('inf')
    best_accuracy: float = 0
    train_start = time.time()

    for iteration in range(iterations):
        if stop:
            break
        classifier.warm_start = iteration > 0
        start = time.time()
        classifier.fit(training_encoded, train_y)
        print(f"Iteration time: {time.time() - start}s")

        y_pred_train = classifier.predict(training_encoded)
        y_pred_prob_train = classifier.predict_proba(training_encoded)
        y_pred_val = classifier.predict(val_encoded)
        y_pred_prob_val = classifier.predict_proba(val_encoded)
        scores['train loss'].append(metrics.log_loss(train_y, y_pred_prob_train))
        scores['val loss'].append(metrics.log_loss(val_y, y_pred_prob_val))
        scores['train accuracy'].append(metrics.accuracy_score(train_y, y_pred_train))
        scores['val accuracy'].append(metrics.accuracy_score(val_y, y_pred_val))

        plot_model(ax1, ax2, scores, iteration)
        if scores['val accuracy'][-1] > best_accuracy:
            joblib.dump(classifier, Path(model_path) / "best")
            best_accuracy = scores['val accuracy'][-1]
            print(f"{best_accuracy=}")
        with statistics_path.open("a", encoding="utf-8") as file:
            stats = ','.join([
                str(iteration), str(scores['train loss'][-1]), str(scores['val loss'][-1]), 
                str(scores['train accuracy'][-1]), str(scores['val accuracy'][-1]), str(time.time() - train_start)
            ])
            file.write(f"{stats}\n")
    
    print("Training complete")

def main():
    splits = load_dataset(recreate=True)

    def get_split(index):
        nonlocal splits
        split = splits[index]
        del splits
        import gc
        gc.collect()
        return split

    split = get_split(1)
    show3(split.train)

    train(split)

def train_autoencoder(split):
    from train_autoencoder import train_autoencoder as train_autoenc
    train_autoenc(split)

if __name__ == "__main__":
    original_handler = getsignal(SIGINT)
    # First Ctrl+C will exit loop after ending current iterations, second one will use default handler
    def stop_handler(_, _2):
        global stop
        stop = True
        signal(SIGINT, original_handler)
    signal(SIGINT, stop_handler)
    plt.show(block=False)
    main()
    plt.show()
