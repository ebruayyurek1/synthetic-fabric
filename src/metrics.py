from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


def compute_all_errors(true_centers: np.ndarray, predicted_centers: np.ndarray, matching_threshold_px: float = 5) -> np.ndarray:
    """
    Compute all errors based on the true and extracted centers

    @param true_centers: N true centers formatted as (x, y) coordinates
    @param predicted_centers: M true centers in (x, y) coordinates
    @param matching_threshold_px: max pixel distance between true and predicted centers to allow for a match
    @return: numpy array with pairwise distance of matched centers
    """
    # Pair true centers with predicted centers, or match with None if they cannot be matched
    matches, pairwise_distances = _match_true_with_predicted(true_centers, predicted_centers, matching_threshold_px)

    # Compute precision metrics for matched pairs
    error = list()
    for i, j in matches:
        if i is not None and j is not None:
            error.append(pairwise_distances[i, j])
    return np.array(error)


def compute_metrics(true_centers: np.ndarray, predicted_centers: np.ndarray, matching_threshold_px: float = 5) -> dict[str, float]:
    """
    Compute metrics based on the true and extracted centers

    @param true_centers: N true centers formatted as (x, y) coordinates
    @param predicted_centers: M true centers in (x, y) coordinates
    @param matching_threshold_px: max pixel distance between true and predicted centers to allow for a match
    @return: dictionary of metrics with their name and value
    """
    # Pair true centers with predicted centers, or match with None if they cannot be matched
    matches, pairwise_distances = _match_true_with_predicted(true_centers, predicted_centers, matching_threshold_px)

    # Compute precision metrics for matched pairs
    error = list()
    for i, j in matches:
        if i is not None and j is not None:
            error.append(pairwise_distances[i, j])
    error = np.array(error)

    ma_err, std_err, min_err, max_err, rms_err = -1, -1, -1, -1, -1
    if len(error):
        ma_err, std_err, min_err, max_err = float(error.mean()), float(error.std()), float(error.min()), float(error.max())
        rms_err = float(np.sqrt(np.square(error).mean()))

    # Compute recall metrics for all pairs
    recall = float(min(len(error), len(true_centers)) / max(len(error), len(true_centers)))

    # Derive a precision score comparing the error to the max possible error
    precision_rmse = 1 - rms_err / matching_threshold_px
    h_mean = 2 * (recall * precision_rmse) / (recall + precision_rmse)

    return {
        "mae": ma_err,
        "rmse": rms_err,
        "min_err": min_err,
        "max_err": max_err,
        "std_err": std_err,
        "completeness": recall,
        "fitness": h_mean
    }


def _match_true_with_predicted(true_centers: np.ndarray, predicted_centers: np.ndarray, matching_threshold_px: float) \
        -> tuple[list[tuple[Optional[int], Optional[int]]], np.ndarray]:
    # Pair true centers with predicted centers, or match with None if they cannot be matched
    pairwise_distances = cdist(true_centers, predicted_centers, "euclidean")  # N x M

    matches: list[tuple[Optional[int], Optional[int]]] = list()
    pred_matched_indexes: set[int] = set()
    for i in range(len(true_centers)):
        below_threshold_indexes = np.argwhere(pairwise_distances[i] < matching_threshold_px).squeeze(1)
        # # Alternative 1: match the closest center if more than 1 can be matched
        # # --------------------------------------------------------------------------------------
        # below_threshold_indexes = np.setdiff1d(below_threshold_indexes, np.array(list(pred_matched_indexes)))
        # below_threshold_indexes = below_threshold_indexes[np.argsort(pairwise_distances[i, below_threshold_indexes])]  # sort ascending
        # if len(below_threshold_indexes) > 0:
        # #     --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------
        # Alternative 2: match two centers only if they can be unambiguously matched
        if len(below_threshold_indexes) == 1 and below_threshold_indexes[0] not in pred_matched_indexes:
            # --------------------------------------------------------------------------------------
            matched_index = int(below_threshold_indexes[0])
            matches.append((i, matched_index))
            pred_matched_indexes.add(matched_index)
        else:
            matches.append((i, None))
    for j in range(len(predicted_centers)):
        if j not in pred_matched_indexes:
            matches.append((None, j))

    return matches, pairwise_distances


def auc_curve_metrics(true_centers: np.ndarray, predicted_centers: np.ndarray, save_path: Optional[Path] = None) -> dict[str, float]:
    """
    Plot metrics over several matching threshold values for better comparisons.

    @param true_centers: N true centers formatted as (x, y) coordinates
    @param predicted_centers: M predicted centers in (x, y) coordinates
    @param save_path: an optional path to file to dump the image. If None plot is shown
    """
    thresholds = np.array([1, 3, 5])

    tracked_metrics: list[list[float]] = [[], [], [], []]
    for threshold in thresholds:
        metrics = compute_metrics(true_centers, predicted_centers, threshold)
        tracked_metrics[0].append(metrics["mae"])
        tracked_metrics[1].append(metrics["rmse"])
        tracked_metrics[2].append(metrics["recall"])
        tracked_metrics[3].append(metrics["h_mean"])

    fig, axs = plt.subplots(figsize=(15, 12), nrows=2, ncols=2)
    axs[0, 0].plot(thresholds, tracked_metrics[0], "x--m", label="MAE")
    axs[0, 1].plot(thresholds, tracked_metrics[1], "x--g", label="RMSE")
    axs[1, 0].plot(thresholds, tracked_metrics[2], "x--b", label="Recall")
    axs[1, 1].plot(thresholds, tracked_metrics[3], "x--y", label="H-AVG")
    axs[0, 0].set_ylabel("MAE")
    axs[0, 1].set_ylabel("RMSE")
    axs[1, 0].set_ylabel("Recall")
    axs[1, 1].set_ylabel("H-AVG")
    axs[1, 0].set_ylim(-0.1, 1.1)
    axs[1, 1].set_ylim(-0.1, 1.1)
    for ax in axs.flatten():
        ax.set_xlabel("Pixel Threshold")
        ax.set_xticks(thresholds)
        ax.set_xlim(0, thresholds[-1] + 1)
        # ax.set_ylabel(ax.get_label())
    fig.suptitle(f"Metrics at pixel thresholds [{int(thresholds.min())}, {int(thresholds.max())}]")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.15, hspace=0.15)

    if save_path:
        save_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(save_path, dpi=400)
    else:
        # plt.show()
        pass

    plt.close()

    # fig, axs = plt.subplots(figsize=(10, 8))
    # fig.suptitle("PRMSE-REC curve")
    # axs.plot(tracked_metrics[0], tracked_metrics[2], "x--k", label="P-R")
    # axs.set_xlim(-0.1, 1.1)
    # axs.set_ylim(-0.1, 1.1)
    # plt.show()

    # Compute the integral of the curve with precision and recall
    # auc_pr = float(sp.integrate.simpson(tracked_metrics[1], x=tracked_metrics[2]))
    # TODO: decide which threshold to use
    mean_f1 = float(np.mean(tracked_metrics[3]))
    return dict(mean_h=mean_f1, mean_mae=float(np.mean(tracked_metrics[0])),
                mean_rmse=float(np.mean(tracked_metrics[1])), mean_recall=float(np.mean(tracked_metrics[2])))


if __name__ == '__main__':
    np.random.seed(42)
    d = np.random.randint(0, 200, (600, 2))
    e = np.random.randint(0, 200, (1000, 2))
    mm = compute_metrics(d, e)
    print(mm)

    p = auc_curve_metrics(d, e)
    print(p)
