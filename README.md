# code-executor-py 

[![PyPI version](https://img.shields.io/pypi/v/code-executor-py.svg)](https://pypi.org/project/code-executor-py/)
[![Downloads](https://img.shields.io/pypi/dm/code-executor-py.svg)](https://pypi.org/project/code-executor-py/)
[![License](https://img.shields.io/github/license/sawradip/code_executor_py.svg)](https://github.com/sawradip/code_executor_py/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/sawradip/code_executor_py.svg)](https://github.com/sawradip/code_executor_py/stargazers)

A powerful and flexible Python module for secure code execution in isolated environments. Execute Python code safely in virtual environments locally or in remote sandbox containers.

## Features

- **VenvExecutor**: Execute code in isolated virtual environments
  - Automatic dependency detection and installation
  - Smart package name resolution
  - Handles import errors intelligently

- **RemoteExecutor**: Run code in remote sandbox containers
  - Client-server architecture
  - Secure data serialization
  - Network-isolated execution
 
- **Scoped Environment Variables**
  - Pass per-function environment values during `create_executable`
  - Variables are available only to that execution context

- **Smart Package Management**
  - Automatically resolves import statements to package names
  - Maps common aliases to correct package names (e.g., 'sklearn' → 'scikit-learn')
  - Dynamically installs missing dependencies

## Installation

```bash
pip install code-executor-py
```

## Quick Start

### Local Execution with VenvExecutor

```python
import pandas as pd
from code_executor_py import VenvExecutor

# Define your function
func_code = """
from sklearn.preprocessing import StandardScaler
import pandas as pd

def process_data(data_df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data_df), columns=data_df.columns)
"""

# Create executor and compile function
executor = VenvExecutor()
process_data = executor.create_executable(
    func_code,
    env_vars={"DATA_API_KEY": "abc123"}
)

# Execute function with data
test_data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
result = process_data(data_df=test_data)
print(result)
```

### Remote Execution (Server)

```python
from code_executor_py import RemoteExecutorServer

# Start a server on port 8099
server = RemoteExecutorServer(host="0.0.0.0", port=8099)
server.run()
```

### Remote Execution (Client)

```python
from code_executor_py import RemoteExecutor

# Connect to the remote server
executor = RemoteExecutor("http://localhost:8099")

# Define your function
func_code = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""

# Create executable function
add_numbers = executor.create_executable(
    func_code,
    env_vars={"REMOTE_SECRET": "secret-value"}
)

# Execute remotely
result = add_numbers(5, 3)
print(result)  # Output: 8
```

## Advanced Usage

### Using with LLMs to Auto-Install Dependencies

```python
from code_executor_py import VenvExecutor
from langchain_openai import ChatOpenAI

# Create executor with LLM for smart package resolution
executor = VenvExecutor(
    llm=ChatOpenAI(temperature=0),
    debug_mode=True
)

# The executor can now ask the LLM for the correct package name
# when encountering unknown imports
```

### Scoped Environment Variables

```python
from code_executor_py import VenvExecutor

executor = VenvExecutor()

secret_function = executor.create_executable(
    """
import os


def get_secret():
    return os.environ.get("MY_SECRET")
""",
    function_name="get_secret",
    env_vars={"MY_SECRET": "top-secret"}
)

print(secret_function())  # -> "top-secret"
```

Environment variables provided in `env_vars` are injected only into the spawned
subprocess and never touch the parent process.

### Custom Base Packages

```python
from code_executor_py import VenvExecutor

# Create executor with custom base packages
executor = VenvExecutor(
    venv_path="./custom_venv",
    base_packages=["numpy", "pandas", "matplotlib", "scipy"]
)
```

## How It Works

1. **Code Analysis**: Your function code is analyzed to extract imports
2. **Environment Preparation**: Dependencies are automatically installed in isolated environments
3. **Secure Execution**: Code runs in a separate process with proper error handling
4. **Result Serialization**: Results are properly serialized and returned to your main program

## Security Benefits

- **Isolation**: Code executes in separate environments, protecting your main application
- **Dependency Management**: Avoid conflicts with your application's dependencies
- **Resource Control**: Limit the resources available to executed code (especially with remote execution)
- **Network Isolation**: Remote execution provides complete network isolation

## Use Cases

- **AI/ML Pipelines**: Safely execute generated or user-provided code
- **Data Processing**: Run data transformation scripts in isolation
- **Teaching/Education**: Create safe execution environments for student code
- **Microservices**: Execute code remotely for resource-intensive operations

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## Author

Sawradip Saha - [GitHub](https://github.com/sawradip) - [Email](mailto:sawradip0@gmail.com)