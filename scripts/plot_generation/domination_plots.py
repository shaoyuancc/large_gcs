from dataclasses import dataclass
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from pydrake.symbolic import Monomial, Polynomial, Variable

from large_gcs.utils.utils import use_type_1_fonts_in_plots
from large_gcs.visualize.colors import (
    BLUE,
    DEEPPINK2,
    DODGERBLUE3,
    GREEN,
    GREEN2,
    GREEN3,
    GREEN4,
    LIGHTSEAGREEN,
    PURPLE,
)
from large_gcs.visualize.plot_sampling_comparison import SamplingRunData, SingleRunData


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


def _plot_vline(x, y, color):
    plt.plot(
        [x, x],
        [y, 999],
        linestyle="--",
        linewidth=1,
        zorder=99,
        color=color,
    )


@dataclass
class Curve:
    x_vals: np.ndarray
    y_vals: np.ndarray
    color: np.ndarray
    name: Optional[str] = None

    def plot(self) -> None:
        plt.plot(self.x_vals, self.y_vals, label=self.name, zorder=99, color=self.color)

        # indicate start and end
        _plot_vline(self.x_vals[0], self.y_vals[0], self.color)
        _plot_vline(self.x_vals[-1], self.y_vals[-1], self.color)

    @classmethod
    def make_quadratic(
        cls,
        a,
        b,
        c,
        shift,
        x_min,
        x_max,
        with_dash: bool = False,
        no_name: bool = False,
    ) -> "Curve":
        x_vals = np.linspace(x_min, x_max, 100)
        if with_dash:
            color = G_NEXT_COLOR
            name = r"$\tilde{g}(v',x)$"
        else:
            color = G_COLOR
            name = r"$\tilde{g}(v,x)$"

        if no_name:
            name = None
        return cls(x_vals, _make_values(a, b, c, shift, x_vals), color, name)

    def at(self, x: float) -> float:
        if x < min(self.x_vals) or x > max(self.x_vals):
            return np.inf
        idx_before = np.where(self.x_vals < x)[0][-1]
        idx_after = np.where(self.x_vals > x)[0][0]
        y_mean = (self.y_vals[idx_before] + self.y_vals[idx_after]) / 2
        return y_mean  # type: ignore


def plot_min(curves: List[Curve], x_vals, offset=1) -> None:
    min_vals = np.array([np.min([c.at(x) for c in curves]) for x in x_vals])
    plt.plot(
        x_vals, min_vals - offset, color=MIN_COLOR, linewidth=2.0, label=r"Minimum"
    )


#
# parser = argparse.ArgumentParser(description="Figure to generate")
# parser.add_argument(
#     "--num",
#     type=int,
#     help="Figure number",
#     required=True,
# )
#
#
# # Parse the arguments
# args = parser.parse_args()
# figure_idx = args.num


G_COLOR = DODGERBLUE3.diffuse()
G_NEXT_COLOR = DEEPPINK2.diffuse()
MIN_COLOR = GREEN2.diffuse()


def make_plot(figure_idx, filename=None):
    use_type_1_fonts_in_plots()
    fig = plt.figure(figsize=(3, 2))

    if figure_idx == 0:

        # Generate x values
        x_vals = np.linspace(0, 10, 800)

        g = Curve.make_quadratic(
            6,
            1,
            7,
            shift=-5,
            x_min=2.5,
            x_max=7,
        )
        g_next = Curve.make_quadratic(
            2,
            1,
            5,
            shift=-5,
            x_min=2,
            x_max=7.5,
            with_dash=True,
        )

        g.plot()
        g_next.plot()

        plot_min([g, g_next], x_vals)
        y_max = 50

        plt.legend(fontsize=14, loc="center left")

        ax = plt.gca()

        # Create a new figure for the legend
        fig_legend = plt.figure(figsize=(1.5, 1))

        # Add the legend to the new figure
        fig_legend.legend(*ax.get_legend_handles_labels(), loc="center")

        # Remove axes from the legend figure
        plt.axis("off")

        # Save the legend figure
        fig_legend.savefig("domination_legend.pdf")

    elif figure_idx == 1:

        # Generate x values
        x_vals = np.linspace(1, 9, 800)

        g = Curve.make_quadratic(
            5,
            1,
            7,
            shift=-4,
            x_min=2,
            x_max=6,
        )
        g_next = Curve.make_quadratic(
            5,
            1,
            7,
            shift=-6.5,
            x_min=5,
            x_max=8,
            with_dash=True,
        )

        g.plot()
        g_next.plot()

        plot_min([g, g_next], x_vals)
        y_max = 50

    elif figure_idx == 2:

        # Generate x values
        x_vals = np.linspace(0, 6, 800)

        g_next = Curve.make_quadratic(
            20,
            1,
            12,
            shift=-4,
            x_min=3.5,
            x_max=4.5,
            with_dash=True,
        )

        # idx = np.where(g_next.x_vals > 5)[0][0]
        # asymptote = -1 / (g_next.x_vals - 5)
        # asymptote[idx:] = np.inf
        # g_next.y_vals += asymptote
        g_next.plot()

        g = Curve.make_quadratic(7, 1, 7, shift=-3, x_min=1, x_max=5, with_dash=False)
        g.plot()

        plot_min([g, g_next], x_vals)

        y_max = 50

    elif figure_idx == 3:

        # Generate x values
        x_vals = np.linspace(0, 10, 800)

        g_1 = Curve.make_quadratic(8, 1, 7, shift=-3, x_min=1, x_max=4)
        g_2 = Curve.make_quadratic(4, 1, 6, shift=-5, x_min=3, x_max=6, no_name=True)
        g_3 = Curve.make_quadratic(3, 1, 8, shift=-8, x_min=5.5, x_max=9, no_name=True)
        g_next = Curve.make_quadratic(
            4, 1, 5, shift=-6.5, x_min=5, x_max=8, with_dash=True
        )

        g_1.plot()
        g_2.plot()
        g_3.plot()
        g_next.plot()

        plot_min([g_1, g_2, g_3, g_next], x_vals, offset=0.5)

        y_max = 30

    elif figure_idx == 4:

        # Generate x values
        x_vals = np.linspace(0, 10, 800)

        g_1 = Curve.make_quadratic(5, 1, 7, shift=-3, x_min=1, x_max=4)
        g_2 = Curve.make_quadratic(4, 1, 6, shift=-5, x_min=3, x_max=7)
        g_3 = Curve.make_quadratic(3, 1, 8, shift=-7, x_min=5.5, x_max=9)
        g_next = Curve.make_quadratic(
            1.5, 1, 13, shift=-7, x_min=5, x_max=8, with_dash=True
        )

        g_1.plot()
        g_2.plot()
        g_3.plot()
        g_next.plot()

        plot_min([g_1, g_2, g_3, g_next], x_vals, offset=0.5)

        y_max = 30

    else:  # test figure
        # Coefficients for the quadratic equations
        a, b, c = 1, -3, 2
        d, e, f = -1, 2, 1

        # Generate x values
        x_vals = np.linspace(-10, 10, 800)

        # Calculate y values for both quadratic equations
        g_vals = a * x_vals**2 + b * x_vals + c
        g_next_vals = d * x_vals**2 + e * x_vals + f

        plt.figure(figsize=(4, 4))
        plt.plot(x_vals, g_vals, label=r"$y_1 = x^2 - 3x + 2$")
        plt.plot(x_vals, g_next_vals, label=r"$y_2 = -x^2 + 2x + 1$")
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
        plt.xlabel(r"$x$", fontsize=16)
        plt.ylabel(r"$y$")

        # Indicate an interval along the x-axis with brackets
        interval_start, interval_end = -2, 2
        # ax.axvline(x=interval_start, color="red", linestyle="--")
        # ax.axvline(x=interval_end, color="red", linestyle="--")
        # ax.fill_betweenx(
        #     y=np.linspace(-10, 10, 800),
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
            [x_start, x_end],
            [y_start, y_end],
            linestyle="--",
            color="black",
            linewidth=2,
        )

    # plt.title("Plot of Two Quadratic Functions")
    # Customize the plot to look like a math textbook
    ax = plt.gca()

    # Move left y-axis and bottom x-axis to the center, passing through (0,0)
    # ax.spines["left"].set_position("zero")
    # ax.spines["bottom"].set_position("zero")

    # Eliminate top and right axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    # Remove x and y axis numbers
    ax.set_xticks([])
    ax.set_yticks([])

    # Show ticks in the left and bottom axes only
    # ax.xaxis.set_ticks_position("bottom")
    # ax.yaxis.set_ticks_position("left")
    #
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # Labels and legend
    plt.xlabel(r"$x$", fontsize=16)
    # plt.ylabel(r"$y$")

    # Indicate an interval along the x-axis with brackets
    # interval_start, interval_end = 1, 8
    # ax.axvline(x=interval_start, color="red", linestyle="--")
    # ax.axvline(x=interval_end, color="red", linestyle="--")
    # ax.fill_betweenx(
    #     y=np.linspace(-10, 10, 800),
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

    plt.ylim([0, y_max])
    plt.xlim([min(x_vals), max(x_vals)])

    plt.tight_layout()

    if filename:
        fig.savefig(filename + ".pdf")
        plt.close()
    else:
        plt.show()


names = "a", "d", "c", "e", "f"
for i, name in zip([0, 1, 2, 3, 4], names):
    make_plot(i, f"domination_{name}")
