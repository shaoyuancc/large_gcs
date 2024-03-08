import numpy as np


def copy_pastable_str_from_np_array(arr):
    # Print the array in a copy-pastable format
    arr_str = np.array2string(
        arr, max_line_width=np.inf, threshold=np.inf, separator=","
    )
    arr_str = arr_str.replace("\n", "").replace(" ", "")
    return f"np.array({arr_str})"
