import argparse
from pathlib import Path

from large_gcs.visualize.multirun_data import MultirunData, SingleRunData


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load data and configuration from a specified directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to the directory containing data and config file",
        required=True,
    )
    parser.add_argument(
        "--ah_dir",
        type=str,
        help="Path to the directory containing data and config file for the AH-containment comparison",
        default=None,
    )

    # Parse the arguments
    args = parser.parse_args()

    data_dir = Path(args.dir)

    # Validate the directory
    if not data_dir.is_dir():
        print(f"The directory {data_dir} does not exist.")
        return

    run_data = MultirunData.load_from_folder(data_dir)

    if args.ah_dir is not None:
        ah_comparison_data_dir = Path(args.ah_dir)
        ah_run_data = MultirunData.load_from_folder(ah_comparison_data_dir)
        ah_run_data.save(ah_comparison_data_dir / "aggregated_run_data.json")

        if (
            not len(ah_run_data.data) == 1
            or not type(ah_run_data.data[0]) is SingleRunData
        ):
            raise RuntimeError(
                "AH-containment data should contain exactly one run, where\
                num_samples_per_vertex should not be set in the config."
            )
        ah_data = ah_run_data.data[0]

        run_data.make_sampling_comparison_plot(ah_data)
    else:
        run_data.make_sampling_comparison_plot()


if __name__ == "__main__":
    main()
