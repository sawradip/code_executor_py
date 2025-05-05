# from code_executor_py import RemoteExecutorServer

# res = RemoteExecutorServer(host="0.0.0.0", port=8099)
# res.run()

from code_executor_py import RemoteExecutor


executor = RemoteExecutor("http://localhost:8099")

# Test function code
func_code = """
import toml
from yaml import safe_load
import requests as req
# import pathlib.Path as pth
from itertools import chain, combinations
import tomli, pytz, click

def add_numbers(a: int, b: int) -> int:
    return a + b
    """

# Create executable function
add_numbers = executor.create_executable(func_code)

# Test with positional arguments
result1 = add_numbers(5, 3)
assert result1 == 8, f"Expected 8, got {result1}"

# Test with keyword arguments
result2 = add_numbers(a=2, b=7)
assert result2 == 9, f"Expected 9, got {result2}"

print("All remote executor tests passed!")
