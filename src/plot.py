import matplotlib.pyplot as plt
import numpy
import numpy as np

path = "./outputs/"


def plot_FHE(acc_test: list[float], wdr: list[float], id: str):
    plt.figure(figsize=(10, 5), dpi=400)
    plt.plot(list(range(len(acc_test))), acc_test, label="Test Accuracy", color="blue")
    plt.plot(
        np.array(list(range(len(wdr)))) + 0.25,
        wdr,
        label="Dynamic Watermark",
        color="black",
        marker="v",
        markersize=3,
        linestyle="dashed",
    )
    plt.xlabel("Round")
    plt.xlim(0, len(acc_test))
    plt.yticks(list(np.arange(0, len(acc_test), 5)))
    plt.ylim(0 - 0.01, 1 + 0.01)
    plt.grid(True)
    plt.yticks(list(np.arange(0, 1.05, 0.05)))
    plt.legend()
    plt.savefig(
        path
        + "FHE"
        + id
        + ".pdf"
    )
    plt.close()

    np.savez(
        path
        + "FHE" + "_" + id,
        acc_test,
        wdr,
    )


def plot_pruning_attack(pruning_rates: np.array, test_accuracy: np.array, wdr_dynamic: np.array):
    accuracy_mean = np.mean(test_accuracy, axis=0)
    accuracy_std = np.std(test_accuracy, axis=0)

    wdr_mean = np.mean(wdr_dynamic, axis=0)
    wdr_std = np.std(wdr_dynamic, axis=0)

    plt.figure(figsize=(5, 4), dpi=400)

    plt.plot(pruning_rates, accuracy_mean, c="blue", label="Test")
    plt.fill_between(pruning_rates, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, color="blue",
                     alpha=0.25)

    plt.plot(pruning_rates, wdr_mean, c="black", label="Dynamic Watermark", marker="v",
             markersize=3,
             linestyle="dashed")
    plt.fill_between(pruning_rates, wdr_mean - wdr_std, wdr_mean + wdr_std, color="black", alpha=0.25)

    plt.plot(pruning_rates, 0.47 * np.ones(len(pruning_rates)), c="red", label="$\delta$", linestyle='dashed')

    plt.xlabel("Pruning rate")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0 - 0.1, 1 + 0.1)
    plt.xticks(pruning_rates)
    plt.yticks(pruning_rates)
    plt.grid(True)
    plt.legend()
    plt.savefig(
        path
        + "pruning.pdf"
    )
    plt.close()
