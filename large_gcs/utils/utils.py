import pickle
from dataclasses import fields

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
