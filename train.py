from itertools import repeat
from matplotlib.pylab import Axes
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from signal import SIGINT, signal, getsignal

from dataset import load_dataset
from methods import Label, Split

stop = False

def train(split: Split):
    print("Training")
    for set_str, set in zip(split._fields, split):
        shape = set.data.shape
        set.data = set.data.reshape(len(set), -1)
        print(f"Reshaping {set_str} set from {shape} to {set.data.shape}")
    
    scores = {'loss': [], 'train_accuracy': [], 'val_accuracy': []}
    iterations = 30
    hidden_layers = (4,4)
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=1, max_iter=1)

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

    ax: Axes
    fig, ax = plt.subplots()

    for iter in range(iterations):
        if stop:
            break
        classifier.warm_start = iter > 0
        classifier.fit(split.train, train_y)
        scores['loss'].append(classifier.loss_)
        y_pred = classifier.predict(split.train)
        y_pred_val = classifier.predict(split.val)
        scores['train_accuracy'].append(metrics.accuracy_score(train_y, y_pred))
        scores['val_accuracy'].append(metrics.accuracy_score(val_y, y_pred_val))
        ax.clear()
        for score_legend, score in scores.items():
            ax.plot(range(1, iter + 2), score, label=score_legend)
            plt.draw()
        ax.legend(loc="upper right")
        plt.pause(0.25)

def main():
    original_handler = getsignal(SIGINT)
    # First Ctrl+C will exit loop after ending current iterations, second one will use default handler
    def stop_handler(_, _2):
        global stop
        stop = True
        signal(SIGINT, original_handler)
    signal(SIGINT, stop_handler)

    splits = load_dataset()

    train(splits[1])


if __name__ == "__main__":
    plt.show(block=False)
    main()
    plt.show()