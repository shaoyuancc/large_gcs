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
        help="Path to the directory containing multirun data and config file",
        required=True,
    )

    # Parse the arguments
    args = parser.parse_args()

    data_dir = Path(args.dir)

    # Validate the directory
    if not data_dir.is_dir():
        print(f"The directory {data_dir} does not exist.")
        return

    run_data = MultirunData.load_from_folder(data_dir)

    run_data.save(data_dir / "aggregated_run_data.json")


if __name__ == "__main__":
    main()
