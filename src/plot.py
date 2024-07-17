import matplotlib.pyplot as plt
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
    plt.savefig(path + "FHE" + id + ".pdf")
    plt.close()

    np.savez(
        path + "FHE" + "_" + id,
        acc_test,
        wdr,
    )


def plot_FHE_overwriting(
    acc_test: list[float], wdr_org: list[float], wdr_new: list[float], id: str
):
    accuracy_mean = np.mean(acc_test, axis=0)
    accuracy_std = np.std(acc_test, axis=0)

    wdr_org_mean = np.mean(wdr_org, axis=0)
    wdr_org_std = np.std(wdr_org, axis=0)

    wdr_new_mean = np.mean(wdr_new, axis=0)
    wdr_new_std = np.std(wdr_new, axis=0)
    # %%
    fig = plt.figure(figsize=(4, 3), dpi=400)
    plt.axes(facecolor="#f7f7f7")
    epoch = np.arange(0, len(acc_test) + 1, 1)
    plt.plot(epoch, accuracy_mean, c="blue", label="Test Set")
    plt.fill_between(
        epoch,
        accuracy_mean - accuracy_std,
        accuracy_mean + accuracy_std,
        color="blue",
        alpha=0.25,
    )

    plt.plot(epoch, wdr_org_mean, c="green", label="Original Watermark", markersize=5)
    plt.fill_between(
        epoch,
        wdr_org_mean - wdr_org_std,
        wdr_org_mean + wdr_org_std,
        color="green",
        alpha=0.25,
    )

    plt.plot(epoch, wdr_new_mean, c="red", label="Attacker Watermark", markersize=5)
    plt.fill_between(
        epoch,
        wdr_new_mean - wdr_new_std,
        wdr_new_mean + wdr_new_std,
        color="red",
        alpha=0.25,
    )

    plt.plot(
        epoch,
        0.47 * np.ones(len(epoch)),
        c="black",
        label="Treshold",
        linestyle="dashed",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xlim(0, len(acc_test))
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, len(acc_test) + 1, 10))
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "fine-tuning.pdf")
    plt.close()

    np.savez(path + "FHE_overwriting" + "_" + id, acc_test, wdr_org, wdr_new)


def plot_pruning_attack(
    pruning_rates: np.array, test_accuracy: np.array, wdr_dynamic: np.array
):
    accuracy_mean = np.mean(test_accuracy, axis=0)
    accuracy_std = np.std(test_accuracy, axis=0)

    wdr_mean = np.mean(wdr_dynamic, axis=0)
    wdr_std = np.std(wdr_dynamic, axis=0)

    plt.figure(figsize=(4, 3), dpi=400)
    plt.axes(facecolor="#f7f7f7")

    plt.plot(pruning_rates, accuracy_mean, c="blue", label="Test Set")
    plt.fill_between(
        pruning_rates,
        accuracy_mean - accuracy_std,
        accuracy_mean + accuracy_std,
        color="blue",
        alpha=0.25,
    )

    plt.plot(pruning_rates, wdr_mean, c="green", label="Watermark")
    plt.fill_between(
        pruning_rates, wdr_mean - wdr_std, wdr_mean + wdr_std, color="green", alpha=0.25
    )

    plt.plot(
        pruning_rates,
        0.47 * np.ones(len(pruning_rates)),
        c="black",
        label="Threshold",
        linestyle="dashed",
    )

    plt.xlabel("Pruning rate")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0 - 0.1, 1 + 0.1)
    plt.xticks(np.arange(0, 1.01, 0.2))
    plt.yticks(np.arange(0, 1.01, 0.2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path + "pruning.pdf")
    plt.close()
