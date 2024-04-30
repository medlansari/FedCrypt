import matplotlib.pyplot as plt
import numpy as np

path = "./outputs/"

def plot_FHE(acc_test : list[float], wdr : list[float], id : str):
    plt.figure(figsize=(10,5), dpi=400)
    plt.plot(list(range(len(acc_test))), acc_test, label="Test Accuracy", color="blue")
    plt.plot(
        np.array(list(range(len(wdr))))+0.25,
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