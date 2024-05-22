import itertools
import pickle
from dataclasses import fields
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def copy_pastable_str_from_np_array(arr):
    # Print the array in a copy-pastable format
    arr_str = np.array2string(
        arr, max_line_width=np.inf, threshold=np.inf, separator=","
    )
    arr_str = arr_str.replace("\n", "").replace(" ", "")
    return f"np.array({arr_str})"


def is_pickleable(obj):
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError):
        return False
    return True


def dict_to_dataclass(klass, dikt):
    field_names = {f.name for f in fields(klass) if f.init}
    filtered_arg_dict = {k: v for k, v in dikt.items() if k in field_names}
    return klass(**filtered_arg_dict)


def split_numbers_into_sublists(N: int, M: int) -> List[List[int]]:
    # Create a list of numbers from 0 to N (excluding N)
    numbers = list(range(N))

    # Calculate the size of each sublist
    k, m = divmod(len(numbers), M)

    # Use list comprehension and itertools to split the list into M sublists
    sublists = [
        numbers[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(M)
    ]

    return sublists


def use_type_1_fonts_in_plots() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["ps.useafm"] = True
    plt.rcParams["pdf.use14corefonts"] = True
    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.serif"] = "Computer Modern Roman"
