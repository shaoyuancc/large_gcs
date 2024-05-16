import argparse
from pathlib import Path

from large_gcs.visualize.plot_sampling_comparison import SamplingRunData


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load data and configuration from a specified directory."
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Path to the directory containing data and config file",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the directory path
    data_dir = Path(args.dir)

    # Validate the directory
    if not data_dir.is_dir():
        print(f"The directory {data_dir} does not exist.")
        return

    run_data = SamplingRunData.load_from_folder(data_dir)

    run_data.make_plot()


if __name__ == "__main__":
    main()
