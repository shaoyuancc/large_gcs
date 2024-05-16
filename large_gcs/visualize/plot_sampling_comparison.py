from functools import cached_property
from pathlib import Path
from typing import List, NamedTuple, Optional

import matplotlib.pyplot as plt
from omegaconf import OmegaConf, open_dict

from large_gcs.algorithms.search_algorithm import AlgMetrics
from large_gcs.graph.graph import ShortestPathSolution
from large_gcs.utils.utils import use_type_1_fonts_in_plots


class SingleSamplingRunData(NamedTuple):
    num_samples: int
    cost: float
    wall_clock_time: float
    num_paths_expanded: int


class SamplingRunData:
    def __init__(self, run_data: List[SingleSamplingRunData]) -> None:
        self.data = run_data

    @classmethod
    def load_from_folder(cls, data_dir: Path) -> "SamplingRunData":
        run_data = []

        # Iterate through the runs and generate figures
        for path in data_dir.iterdir():
            # We only care about the subfolders (which contain data for a run)
            if not path.is_dir():
                continue

            # Load the config file for this run
            config_path = path / "config.yaml"
            if not config_path.is_file():
                raise RuntimeError(
                    f"The config file {
                        config_path}"
                    "does not exist."
                )

            cfg: DictConfig = OmegaConf.load(config_path)

            sol_files = list(path.glob("*_solution.pkl"))
            if not len(sol_files) == 1:
                raise RuntimeError(
                    f"Found more than one solution file in {path}."
                    f"This is not expected, so something is likely wrong."
                )
            sol_file = sol_files[0]
            sol = ShortestPathSolution.load(sol_file)

            metric_files = list(path.glob("*_metrics.json"))
            if not len(metric_files) == 1:
                raise RuntimeError(
                    f"Found more than one metric file in {path}."
                    f"This is not expected, so something is likely wrong."
                )
            metric_file = metric_files[0]
            metrics = AlgMetrics.load(metric_file)

            num_samples = cfg.domination_checker.num_samples_per_vertex

            data = SingleSamplingRunData(
                num_samples, sol.cost, metrics.time_wall_clock, 0
            )
            run_data.append(data)

        # sort data based on num_samples
        run_data = sorted(run_data, key=lambda d: d.num_samples)

        return cls(run_data)

    @cached_property
    def num_samples(self) -> List[int]:
        return [d.num_samples for d in self.data]

    @cached_property
    def solve_times(self) -> List[float]:
        return [d.wall_clock_time for d in self.data]

    @cached_property
    def costs(self) -> List[float]:
        return [d.cost for d in self.data]

    @cached_property
    def num_paths_expanded(self) -> List[int]:
        return [d.num_paths_expanded for d in self.data]

    def make_plot(self, output_path: Optional[Path] = None) -> None:
        use_type_1_fonts_in_plots()

        fig_height = 4
        num_plots = 3
        fig, axs = plt.subplots(
            1, num_plots, figsize=(fig_height * num_plots, fig_height)
        )

        for ax in axs:
            ax.set_xlabel("Number of samples")

        axs[0].plot(self.num_samples, self.solve_times)
        axs[0].set_title("Solve time [s]")

        axs[1].plot(self.num_samples, self.num_paths_expanded)
        axs[1].set_title("Number of paths expanded")

        axs[2].plot(self.num_samples, self.costs)
        axs[2].set_title("Cost")

        if output_path:
            fig.savefig(output_path)
            plt.close()
        else:
            plt.show()
