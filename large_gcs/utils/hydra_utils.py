import importlib


def get_function_from_string(function_string):
    """Get function from string.

    Args:
        function_string (str): String representation of function.
    Returns:
        function: Function.
    """
    module_name, function_name = function_string.rsplit(".", 1)
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function
