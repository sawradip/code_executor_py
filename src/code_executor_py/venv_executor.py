import os
import re
import ast
import glob
import venv
import pickle
import tempfile
import subprocess
import contextlib
from pathlib import Path
from typing import Any, Optional, Union, List
from langchain_core.messages import HumanMessage


class VenvExecutor:
    PACKAGE_MAPPING = {
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'cv2': 'opencv-python',
        'tf': 'tensorflow',
        'px': 'plotly_express',
        'plt': 'matplotlib',
    }

    def __init__(self,
                 llm=None,
                 debug_mode=False,
                 venv_path: Optional[Union[str, Path]] = None,
                 base_packages: Optional[list[str]] = None):
        self.venv_dir = Path(venv_path or './.venv')
        self.base_packages = base_packages or ['numpy', 'pandas']
        self.llm = llm
        self.debug_mode = debug_mode

        self.python_path = self.venv_dir / (
            'Scripts' if os.name == 'nt' else 'bin'
            ) / ('python' + ('.exe' if os.name == 'nt' else ''))

        if not self.venv_dir.exists():
            self._create_venv()

    def _create_venv(self):
        """Create virtual environment and install base packages."""
        venv.create(self.venv_dir, with_pip=True)

        # Ensure pip is installed
        subprocess.run(
            [str(self.python_path), '-m', 'ensurepip', '--upgrade'],
            check=True,
            capture_output=True
        )

        if self.base_packages:
            self.install_packages(*self.base_packages)

    def install_packages(self, *packages: str):
        """Install packages in the virtual environment."""

        for package in packages:
            print(f"Installing - {package}")
            subprocess.run(
                [str(self.python_path), '-m', 'pip', 'install', package],
                check=True, capture_output=True
            )

    def _extract_imports(self, code: str) -> List[str]:
        """Extract all import statements using regex.
        
        Handles cases like:
        - import x
        - from x import y 
        - import x as z
        - import x.y as z
        - from x.y import z
        - import x, y, z
        - from x import (y, z)
        """
        import_patterns = [
            r'^import\s+([\w.]+(?:\s*,\s*[\w.]+)*)',  # import x or import x, y, z
            r'^from\s+([\w.]+)\s+import\s+(?:\([\s\w,]+\)|[\s\w,]+)'  # from x import y or from x import (y, z)
        ]
        
        imports = set()
        for pattern in import_patterns:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                # Handle comma-separated imports
                for imp in match.group(1).split(','):
                    # Get base package name before any dots
                    base_pkg = imp.strip().split('.')[0]
                    imports.add(base_pkg)
                    
        return list(imports)

    def _get_package_name(self, import_name: str) -> str:
        """Get correct package name for pip installation."""
        if self.llm and import_name not in self.PACKAGE_MAPPING:
            prompt = f"What is the correct pip install package name for Python import '{import_name}'? Reply with ONLY the package name, nothing else."
            response = self.llm.invoke([HumanMessage(content=prompt)])
            suggested_package = response.content.strip()
            self.PACKAGE_MAPPING[import_name] = suggested_package
            return suggested_package
        return self.PACKAGE_MAPPING.get(import_name, import_name)

    def _handle_import_error(self, error_msg: str) -> bool:
        """Handle import errors by installing missing packages."""
        match = re.search(r"No module named '([\w.]+)'", error_msg)
        if match:
            missing_module = match.group(1).split('.')[0]
            package_name = self._get_package_name(missing_module)
            try:
                print(f"Attempting to install missing package: {package_name}")
                self.install_packages(package_name)
                return True
            except Exception as e:
                print(f"Failed to install package {package_name}: {str(e)}")
                return False
        return False

    def _get_uninstalled_deps(self, package_names):
        # Get the site-packages directory path
        if os.name == 'nt':  # Windows
            site_packages = os.path.join(self.venv_dir, 'Lib', 'site-packages')
        else:  # Unix/Linux/Mac
            # Find the Python version directory (like lib/python3.8)
            lib_dir = os.path.join(self.venv_dir, 'lib')
            if not os.path.exists(lib_dir):
                return package_names
                
            python_dirs = [d for d in os.listdir(lib_dir) if d.startswith('python')]
            if not python_dirs:
                return package_names
                
            site_packages = os.path.join(lib_dir, python_dirs[0], 'site-packages')

        builtin_modules = ['itertools', 'os', 'sys', 're', 'math', 'datetime']
        # if package_name in builtin_modules:
        #     return False  # These are built-in, not venv-installed 

        uninstalled = []
        
        for package_name in package_names:
            if package_name in builtin_modules:
                continue
            # Check various package installation patterns
            package_pypi_name = self._get_package_name(package_name)

            package_dir = os.path.join(site_packages, package_name)
            package_egg = os.path.join(site_packages, f"{package_pypi_name}.egg-info")
            package_dist = os.path.join(site_packages,
                                      f"{package_pypi_name.replace('-', '_')}-*-info")
            
            # If none of the patterns match, package is not installed
            if not (
                os.path.exists(package_dir) or
                os.path.exists(package_egg) or
                glob.glob(package_dist)
            ):
                uninstalled.append(package_pypi_name)
                
        return uninstalled

    def create_executable(self, function_code: str, function_name: Optional[str] = None) -> callable:
        """Create an executable function that runs in the venv."""

        all_dependencies = self._extract_imports(function_code)

        # The names have been converted to PyPi versions
        uninstalled_dependencies = self._get_uninstalled_deps(all_dependencies)

        self.install_packages(*uninstalled_dependencies)

        if not function_name:
            # Find the function that isn't called by others (except in __main__)
            tree = ast.parse(function_code)
            function_nodes = {node.name: node for node in tree.body
                                if isinstance(node, ast.FunctionDef)}

            # Collect function calls within function bodies only
            function_calls = set()
            for func_node in function_nodes.values():
                for node in ast.walk(func_node):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        function_calls.add(node.func.id)

            uncalled = [name for name in function_nodes if name not in function_calls]
            if not uncalled:
                raise ValueError("Could not determine top-level function")
            function_name = uncalled[-1]


        def execute_in_venv(*args, **kwargs) -> Any:
            with (
                contextlib.nullcontext(tempfile.mkdtemp())
                if self.debug_mode
                else tempfile.TemporaryDirectory()
            ) as temp_dir:
                # with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                input_path = temp_dir / 'input.pkl'
                output_path = temp_dir / 'output.pkl'
                error_path = temp_dir / 'error.pkl'
                
                input_data = [args, kwargs]

                if input_data is not None:
                    with open(input_path, 'wb') as f:
                        pickle.dump(input_data, f)

                # Extract imports and function definitions
                tree = ast.parse(function_code)
                imports = []
                functions = []
                
                for node in tree.body:
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        imports.append(ast.unparse(node))
                    elif isinstance(node, ast.FunctionDef):
                        functions.append(ast.unparse(node))

                # Add proper indentation
                indented_imports = ['    ' + imp for imp in imports]
                indented_functions = ['    ' + line for func in functions
                                    for line in func.split('\n')]

                script = (
                    "import pickle\n"
                    "import sys\n"
                    "import traceback\n\n"
                    "try:\n"
                    f"{chr(10).join(indented_imports)}\n\n"
                    f"{chr(10).join(indented_functions)}\n\n"
                    f"    input_data = {input_data is not None}\n"
                    f"    if input_data:\n"
                    f"        with open(r'{input_path}', 'rb') as f:\n"
                    f"            input_data = pickle.load(f)\n\n"
                    f"            args, kwargs = input_data\n\n"
                    f"    result = {function_name}(*args, **kwargs) if input_data else {function_name}()\n"
                    f"    with open(r'{output_path}', 'wb') as f:\n"
                    "        pickle.dump(result, f)\n"
                    "except Exception as e:\n"
                    "    error_info = {\n"
                    "        'type': type(e).__name__,\n"
                    "        'message': str(e),\n"
                    "        'traceback': traceback.format_exc()\n"
                    "    }\n"
                    f"    with open(r'{str(error_path)}', 'wb') as f:\n"
                    "        pickle.dump(error_info, f, protocol=4)\n"
                    "    sys.exit(1)\n"
                )
                script_path = temp_dir / 'script.py'
                with open(script_path, 'w') as f:
                    f.write(script)
                
                trial_counter = 0
                while trial_counter <= len(all_dependencies):
                    trial_counter += 1
                    process = subprocess.run(
                        [str(self.python_path), str(script_path)],
                        capture_output=True,
                        text=True
                    )

                    if process.returncode == 0:
                        with open(output_path, 'rb') as f:
                            return pickle.load(f)
                        
                    else:
                        print(error_path)
                        if error_path.exists():
                            with open(error_path, 'rb') as f:
                                error_info = pickle.load(f)
                            
                            if (error_info['type'] in ['ModuleNotFoundError', 'ImportError']):
                                if self._handle_import_error(error_info['message']) and (trial_counter == len(all_dependencies)):
                                    continue
                                else:
                                    error_type = eval(error_info['type'])
                                    error = error_type(error_info['message'])
                                    error.original_traceback = error_info['traceback']
                                    raise error
                            else:
                                error_type = eval(error_info['type'])
                                error = error_type(error_info['message'])
                                error.original_traceback = error_info['traceback']
                                raise error
                        
                        raise RuntimeError(f"Execution failed: {process.stderr}")
                    

        def wrapper(*args, **kwargs):
            return execute_in_venv(*args, **kwargs)

        return wrapper