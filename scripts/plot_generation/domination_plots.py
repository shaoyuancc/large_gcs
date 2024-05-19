import argparse

import matplotlib.pyplot as plt
import numpy as np
from pydrake.symbolic import Monomial, Polynomial, Variable

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


def _make_values(a, b, c, shift, x_vals) -> np.ndarray:
    x = Variable("x")

    shifted_x = x + shift
    g = Polynomial(a * shifted_x**2 + b * shifted_x + c)

    g_coeffs = [
        coeff.Evaluate() for coeff in list(g.monomial_to_coefficient_map().values())
    ]

    g1, g2, g3 = g_coeffs  # 1, x, x**2

    # Calculate y values for both quadratic equations
    g_vals = g3 * x_vals**2 + g2 * x_vals + g1
    return g_vals


if figure_idx == 1:

    # Generate x values
    x_vals = np.linspace(0, 10, 400)

    # Create the plot
    use_type_1_fonts_in_plots()

    g_vals = _make_values(7, 1, 7, shift=-3, x_vals=x_vals)
    g_tilde_vals = _make_values(7, 1, 7, shift=-7, x_vals=x_vals)

    plt.figure(figsize=(4, 2))
    plt.plot(x_vals, g_vals, label=r"$g(v, x)$", zorder=99, color="b")
    plt.plot(x_vals, g_tilde_vals, label=r"$\tilde{g}(v', x)$", zorder=99, color="pink")

    # min_vals = np.array(
    #     [np.min([g, g_tilde]) for g, g_tilde in zip(g_vals, g_tilde_vals)]
    # )
    # plt.plot(
    #     x_vals,
    #     min_vals,
    #     label=r"$\text{min}(g, \tilde{g}}$",
    #     zorder=0,
    #     linewidth=3,
    #     color="g",
    # )

    # plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and bottom axes only
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    #
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Labels and legend
    plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    # interval_start, interval_end = 1, 8
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
    # plt.text(
    #     interval_start,
    #     0,
    #     r"$\left[ \right.$",
    #     color="black",
    #     fontsize=20,
    #     ha="right",
    #     va="center",
    # )
    # plt.text(
    #     interval_end,
    #     0,
    #     r"$\left. \right]$",
    #     color="black",
    #     fontsize=20,
    #     ha="left",
    #     va="center",
    # )
    # # Create the arrows
    # arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

    # # X-axis arrow
    # ax.annotate("", xy=(1.8, 0), xytext=(-1.8, 0), arrowprops=arrowprops)
    #
    # # Y-axis arrow
    # ax.annotate("", xy=(0, 1.8), xytext=(0, -1.8), arrowprops=arrowprops)

    # Plot the dashed line
    # x_start = 1.0
    # x_end = 10
    # y_start = 0.0
    # y_end = 3.0
    # ax.plot(
    #     [x_start, x_end], [y_start, y_end], linestyle="--", color="black", linewidth=2
    # )

    plt.ylim([0, 50])

    # plt.legend(
    #     loc="lower right",
    #     fontsize=14,
    # )
    plt.show()

if figure_idx == 2:

    # Generate x values
    x_vals = np.linspace(0, 6, 400)

    # Create the plot
    use_type_1_fonts_in_plots()

    g_vals = _make_values(7, 1, 7, shift=-3, x_vals=x_vals)
    g_tilde_vals = _make_values(20, 1, 20, shift=-5, x_vals=x_vals)

    plt.figure(figsize=(4, 2))
    plt.plot(x_vals, g_vals, label=r"$g(v, x)$", zorder=99, color="b")
    plt.plot(x_vals, g_tilde_vals, label=r"$\tilde{g}(v', x)$", zorder=99, color="pink")

    # min_vals = np.array(
    #     [np.min([g, g_tilde]) for g, g_tilde in zip(g_vals, g_tilde_vals)]
    # )
    # plt.plot(
    #     x_vals,
    #     min_vals,
    #     label=r"$\text{min}(g, \tilde{g}}$",
    #     zorder=0,
    #     linewidth=3,
    #     color="g",
    # )

    # plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and bottom axes only
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    #
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Labels and legend
    plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    # interval_start, interval_end = 1, 8
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
    # plt.text(
    #     interval_start,
    #     0,
    #     r"$\left[ \right.$",
    #     color="black",
    #     fontsize=20,
    #     ha="right",
    #     va="center",
    # )
    # plt.text(
    #     interval_end,
    #     0,
    #     r"$\left. \right]$",
    #     color="black",
    #     fontsize=20,
    #     ha="left",
    #     va="center",
    # )
    # # Create the arrows
    # arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

    # # X-axis arrow
    # ax.annotate("", xy=(1.8, 0), xytext=(-1.8, 0), arrowprops=arrowprops)
    #
    # # Y-axis arrow
    # ax.annotate("", xy=(0, 1.8), xytext=(0, -1.8), arrowprops=arrowprops)

    # Plot the dashed line
    # x_start = 1.0
    # x_end = 10
    # y_start = 0.0
    # y_end = 3.0
    # ax.plot(
    #     [x_start, x_end], [y_start, y_end], linestyle="--", color="black", linewidth=2
    # )

    plt.ylim([0, 50])

    # plt.legend(
    #     loc="lower right",
    #     fontsize=14,
    # )
    plt.show()

if figure_idx == 3:

    # Generate x values
    x_vals = np.linspace(0, 10, 400)

    # Create the plot
    use_type_1_fonts_in_plots()

    g_vals_1 = _make_values(5, 1, 7, shift=-3, x_vals=x_vals)
    g_vals_2 = _make_values(4, 1, 6, shift=-5, x_vals=x_vals)
    g_vals_3 = _make_values(3, 1, 8, shift=-8, x_vals=x_vals)
    g_tilde_vals = _make_values(4, 1, 5, shift=-6.5, x_vals=x_vals)

    plt.figure(figsize=(4, 2))
    plt.plot(x_vals, g_vals_1, label=r"$g(v, x)$", zorder=99, color="b")
    plt.plot(x_vals, g_vals_2, label=r"$g(v, x)$", zorder=99, color="b")
    plt.plot(x_vals, g_vals_3, label=r"$g(v, x)$", zorder=99, color="b")
    plt.plot(x_vals, g_tilde_vals, label=r"$\tilde{g}(v', x)$", zorder=99, color="pink")

    min_vals = np.array(
        [
            np.min([g1, g2, g3, g_tilde])
            for g1, g2, g3, g_tilde in zip(g_vals_1, g_vals_2, g_vals_3, g_tilde_vals)
        ]
    )
    plt.plot(
        x_vals,
        min_vals - 0.5,
        label=r"$\text{min}(g, \tilde{g}}$",
        zorder=0,
        linewidth=3,
        color="g",
    )

    # plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Show ticks in the left and bottom axes only
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    #
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Labels and legend
    plt.xlabel(r"$x$")
    # plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    # interval_start, interval_end = 1, 8
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
    # plt.text(
    #     interval_start,
    #     0,
    #     r"$\left[ \right.$",
    #     color="black",
    #     fontsize=20,
    #     ha="right",
    #     va="center",
    # )
    # plt.text(
    #     interval_end,
    #     0,
    #     r"$\left. \right]$",
    #     color="black",
    #     fontsize=20,
    #     ha="left",
    #     va="center",
    # )
    # # Create the arrows
    # arrowprops = dict(arrowstyle="->", linewidth=1.5, color="black")

    # # X-axis arrow
    # ax.annotate("", xy=(1.8, 0), xytext=(-1.8, 0), arrowprops=arrowprops)
    #
    # # Y-axis arrow
    # ax.annotate("", xy=(0, 1.8), xytext=(0, -1.8), arrowprops=arrowprops)

    # Plot the dashed line
    # x_start = 1.0
    # x_end = 10
    # y_start = 0.0
    # y_end = 3.0
    # ax.plot(
    #     [x_start, x_end], [y_start, y_end], linestyle="--", color="black", linewidth=2
    # )

    plt.ylim([0, 30])

    # plt.legend(
    #     loc="lower right",
    #     fontsize=14,
    # )
    plt.show()

else:
    # Coefficients for the quadratic equations
    a, b, c = 1, -3, 2
    d, e, f = -1, 2, 1

    # Generate x values
    x_vals = np.linspace(-10, 10, 400)

    # Calculate y values for both quadratic equations
    g_vals = a * x_vals**2 + b * x_vals + c
    g_tilde_vals = d * x_vals**2 + e * x_vals + f

    # Create the plot
    use_type_1_fonts_in_plots()

    plt.figure(figsize=(4, 4))
    plt.plot(x_vals, g_vals, label=r"$y_1 = x^2 - 3x + 2$")
    plt.plot(x_vals, g_tilde_vals, label=r"$y_2 = -x^2 + 2x + 1$")
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
