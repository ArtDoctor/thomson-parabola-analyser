from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt


def show_or_save(save_path: Path | None, **savefig_kwargs: object) -> None:
    if save_path is not None:
        tmp_path = save_path.with_suffix(".tmp.png")
        plt.savefig(tmp_path, dpi=150, bbox_inches="tight", **savefig_kwargs)
        plt.close()
        tmp_path.replace(save_path)
    elif matplotlib.is_interactive():
        plt.show()
    else:
        plt.close()
