"""Read a finished run's loss curves and flag the common ways training goes wrong."""

from dataclasses import asdict, dataclass


@dataclass
class Warning:
    code: str  # diverged | no_improvement | overfitting | memorised
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


def diagnose(train: list[tuple[int, float]], val: list[tuple[int, float]]) -> list[Warning]:
    """`train`/`val` are (iteration, loss) pairs in iteration order."""
    out: list[Warning] = []
    if len(train) >= 3:
        first = train[0][1]
        best_so_far = first
        for it, loss in train[1:]:
            # A jump well above both the starting loss and the best seen so far.
            if loss > 1.5 * best_so_far and loss > first:
                out.append(
                    Warning(
                        "diverged",
                        f"Training loss spiked to {loss:.2f} at iteration {it} (best before was {best_so_far:.2f}). "
                        "The learning rate is probably too high; try halving it or adding warmup.",
                    )
                )
                break
            best_so_far = min(best_so_far, loss)
        end = train[-1][1]
        if end >= first * 0.95 and not any(w.code == "diverged" for w in out):
            out.append(
                Warning(
                    "no_improvement",
                    f"Training loss didn't improve ({first:.2f} → {end:.2f}). Check the data mapping, "
                    "or raise the learning rate / number of iterations.",
                )
            )
    if len(val) >= 2:
        start, end = val[0][1], val[-1][1]
        best_it, best = min(val, key=lambda p: p[1])
        if end < 0.05:
            out.append(
                Warning(
                    "memorised",
                    f"Validation loss fell to {end:.3f}. That usually means validation examples overlap the "
                    "training data (near-duplicates) or the dataset is tiny and repetitive, so it can't show "
                    "whether the model generalises.",
                )
            )
        elif len(val) >= 3 and end > best * 1.1 and end > 0.1 and train and train[-1][1] < best:
            out.append(
                Warning(
                    "overfitting",
                    f"Validation loss bottomed out at {best:.3f} (iteration {best_it}) and rose to {end:.3f} while "
                    "training loss kept falling. Use fewer iterations, a lower learning rate or more data.",
                )
            )
        elif end > start * 1.05 and not any(w.code == "diverged" for w in out):
            out.append(Warning("val_worse", f"Validation loss got worse ({start:.3f} → {end:.3f})."))
    return out
