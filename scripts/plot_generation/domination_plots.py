import argparse

import matplotlib.pyplot as plt
import numpy as np

from large_gcs.utils.utils import use_type_1_fonts_in_plots
from large_gcs.visualize.plot_sampling_comparison import SamplingRunData, SingleRunData

parser = argparse.ArgumentParser(description="Figure to generate")
parser.add_argument(
    "--num",
    type=int,
    help="Figure number",
    required=True,
)


# Parse the arguments
args = parser.parse_args()
figure_idx = args.num

if figure_idx == 1:
    # Coefficients for the quadratic equations
    a, b, c = 1, -3, 2
    d, e, f = -1, 2, 1

    # Generate x values
    x = np.linspace(-10, 10, 400)

    # Calculate y values for both quadratic equations
    y1 = a * x**2 + b * x + c
    y2 = d * x**2 + e * x + f

    # Create the plot
    use_type_1_fonts_in_plots()

    plt.figure(figsize=(4, 4))
    plt.plot(x, y1, label=r"$y_1 = x^2 - 3x + 2$")
    plt.plot(x, y2, label=r"$y_2 = -x^2 + 2x + 1$")
    plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and bottom axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Labels and legend
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    interval_start, interval_end = -2, 2
    # ax.axvline(x=interval_start, color="red", linestyle="--")
    # ax.axvline(x=interval_end, color="red", linestyle="--")
    # ax.fill_betweenx(
    #     y=np.linspace(-10, 10, 400),
    #     x1=interval_start,
    #     x2=interval_end,
    #     color="red",
    #     alpha=0.1,
    # )

    # Add brackets to indicate the interval
    plt.text(
        interval_start,
        0,
        r"$\left[ \right.$",
        color="black",
        fontsize=20,
        ha="right",
        va="center",
    )
    plt.text(
        interval_end,
        0,
        r"$\left. \right]$",
        color="black",
        fontsize=20,
        ha="left",
        va="center",
    )
    # Create the arrows
    arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

    # X-axis arrow
    ax.annotate("", xy=(1.8, 0), xytext=(-1.8, 0), arrowprops=arrowprops)

    # Y-axis arrow
    ax.annotate("", xy=(0, 1.8), xytext=(0, -1.8), arrowprops=arrowprops)

    # Plot the dashed line
    x_start = 1.0
    x_end = 10
    y_start = 0.0
    y_end = 3.0
    ax.plot(
        [x_start, x_end], [y_start, y_end], linestyle="--", color="black", linewidth=2
    )

    plt.legend()
    plt.show()

else:
    # Coefficients for the quadratic equations
    a, b, c = 1, -3, 2
    d, e, f = -1, 2, 1

    # Generate x values
    x = np.linspace(-10, 10, 400)

    # Calculate y values for both quadratic equations
    y1 = a * x**2 + b * x + c
    y2 = d * x**2 + e * x + f

    # Create the plot
    use_type_1_fonts_in_plots()

    plt.figure(figsize=(4, 4))
    plt.plot(x, y1, label=r"$y_1 = x^2 - 3x + 2$")
    plt.plot(x, y2, label=r"$y_2 = -x^2 + 2x + 1$")
    plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and bottom axes only
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")

    # Labels and legend
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    interval_start, interval_end = -2, 2
    # ax.axvline(x=interval_start, color="red", linestyle="--")
    # ax.axvline(x=interval_end, color="red", linestyle="--")
    # ax.fill_betweenx(
    #     y=np.linspace(-10, 10, 400),
    #     x1=interval_start,
    #     x2=interval_end,
    #     color="red",
    #     alpha=0.1,
    # )

    # Add brackets to indicate the interval
    plt.text(
        interval_start,
        0,
        r"$\left[ \right.$",
        color="black",
        fontsize=20,
        ha="right",
        va="center",
    )
    plt.text(
        interval_end,
        0,
        r"$\left. \right]$",
        color="black",
        fontsize=20,
        ha="left",
        va="center",
    )
    # Create the arrows
    arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

    # X-axis arrow
    ax.annotate("", xy=(1.8, 0), xytext=(-1.8, 0), arrowprops=arrowprops)

    # Y-axis arrow
    ax.annotate("", xy=(0, 1.8), xytext=(0, -1.8), arrowprops=arrowprops)

    # Plot the dashed line
    x_start = 1.0
    x_end = 10
    y_start = 0.0
    y_end = 3.0
    ax.plot(
        [x_start, x_end], [y_start, y_end], linestyle="--", color="black", linewidth=2
    )

    plt.legend()
    plt.show()
