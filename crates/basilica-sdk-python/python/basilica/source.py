"""
Source Code Packaging Utilities

This module provides utilities for packaging Python source code for deployment.
It handles reading source files and generating the appropriate container commands.

The SourcePackager abstracts the complexity of:
- Reading source from files or inline strings
- Extracting source from Python functions via inspect
- Detecting the source type (file path vs inline code)
- Generating proper container commands with heredocs
- Escaping special characters

Example:
    >>> from basilica.source import SourcePackager
    >>> packager = SourcePackager("path/to/app.py")
    >>> command = packager.build_command()
    >>> # Returns: ["bash", "-c", "python - <<'PYCODE'\\n...code...\\nPYCODE\\n"]

    >>> # Or from a function:
    >>> def my_app():
    ...     print("Hello!")
    >>> packager = SourcePackager.from_function(my_app)
"""

import inspect
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from .exceptions import SourceError


class SourcePackager:
    """
    Packages Python source code for container deployment.

    This class handles the complexity of preparing Python source code to run
    inside a container. It supports both file paths and inline code strings.

    Attributes:
        source: The original source (file path or code string)
        code: The actual Python code content
        is_file: Whether the source was loaded from a file

    Example:
        From a file:
        >>> packager = SourcePackager("app.py")
        >>> print(packager.code)  # File contents

        From inline code:
        >>> packager = SourcePackager("print('Hello, World!')")
        >>> print(packager.code)  # "print('Hello, World!')"
    """

    # Default Python image for deployments
    DEFAULT_IMAGE = "python:3.11-slim"

    # Common pip packages to pre-install for web apps
    WEB_PACKAGES = ["fastapi", "uvicorn", "pydantic"]

    def __init__(self, source: Union[str, Path]):
        """
        Initialize the source packager.

        Args:
            source: Either a file path to a Python file, or inline Python code.
                    If the string looks like a file path (ends with .py or is an
                    existing file), it will be read. Otherwise, treated as code.

        Raises:
            SourceError: If the file doesn't exist or can't be read
        """
        self.source = str(source)
        self.is_file = False
        self.code = ""

        # Detect if source is a file path or inline code
        if self._is_file_path(self.source):
            self.code = self._read_file(self.source)
            self.is_file = True
        else:
            self.code = self.source
            self.is_file = False

        if not self.code.strip():
            raise SourceError("Source code is empty", source_path=self.source if self.is_file else None)

    def _is_file_path(self, source: str) -> bool:
        """
        Determine if the source string is a file path.

        A string is considered a file path if:
        1. It ends with .py, OR
        2. It's an existing file path

        Returns:
            True if the source appears to be a file path
        """
        # Check if it looks like a Python file
        if source.endswith(".py"):
            return True

        # Check if it's an existing file
        expanded = os.path.expanduser(source)
        if os.path.isfile(expanded):
            return True

        return False

    def _read_file(self, path: str) -> str:
        """
        Read source code from a file.

        Args:
            path: Path to the Python file

        Returns:
            The file contents as a string

        Raises:
            SourceError: If the file doesn't exist or can't be read
        """
        expanded_path = os.path.expanduser(path)

        if not os.path.exists(expanded_path):
            raise SourceError(f"Source file '{path}' not found", source_path=path)

        try:
            with open(expanded_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError as e:
            raise SourceError(f"Failed to read source file '{path}': {e}", source_path=path)

    def detect_framework(self) -> Optional[str]:
        """
        Detect the web framework used in the source code.

        Inspects the source code for common framework imports.

        Returns:
            The detected framework name: "fastapi", "flask", "django", or None
        """
        code_lower = self.code.lower()

        if "from fastapi" in code_lower or "import fastapi" in code_lower:
            return "fastapi"
        if "from flask" in code_lower or "import flask" in code_lower:
            return "flask"
        if "from django" in code_lower or "import django" in code_lower:
            return "django"

        return None

    def detect_entry_point(self) -> Optional[str]:
        """
        Detect the application entry point.

        Looks for common patterns like:
        - uvicorn.run(app, ...)
        - app.run(...)
        - if __name__ == "__main__":

        Returns:
            A hint about how to run the app, or None if unclear
        """
        if "uvicorn.run" in self.code:
            return "uvicorn"
        if "app.run" in self.code:
            return "app.run"
        if 'if __name__ == "__main__"' in self.code or "if __name__ == '__main__'" in self.code:
            return "main"

        return None

    def build_command(
        self,
        pip_packages: Optional[List[str]] = None,
        python_args: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Build the container command to run the Python source.

        Generates a bash command that:
        1. Optionally installs pip packages
        2. Executes the Python code using a heredoc

        Args:
            pip_packages: Additional pip packages to install before running.
                         If None, auto-detects based on framework.
            python_args: Additional arguments to pass to python command.

        Returns:
            A command list suitable for container execution.
            Example: ["bash", "-c", "pip install ... && python - <<'PYCODE'\\n...\\nPYCODE"]

        Example:
            >>> packager = SourcePackager("print('hello')")
            >>> cmd = packager.build_command()
            >>> # ["bash", "-c", "python - <<'PYCODE'\\nprint('hello')\\nPYCODE\\n"]
        """
        parts = []

        # Auto-detect packages if not specified
        packages = pip_packages
        if packages is None:
            framework = self.detect_framework()
            if framework == "fastapi":
                packages = self.WEB_PACKAGES

        # Add pip install if packages are specified
        if packages:
            packages_str = " ".join(packages)
            parts.append(f"pip install -q {packages_str}")

        # Build the python command with heredoc
        # Use python3 for broader compatibility (some images don't have python symlink)
        python_cmd = "python3"
        if python_args:
            python_cmd += " " + " ".join(python_args)

        # Use heredoc to pass the code, avoiding shell escaping issues
        # The single quotes around PYCODE prevent variable expansion
        heredoc = f"{python_cmd} - <<'PYCODE'\n{self.code}\nPYCODE\n"
        parts.append(heredoc)

        # Join with && so pip install failure stops execution
        full_command = " && ".join(parts)

        return ["bash", "-c", full_command]

    def build_uvicorn_command(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        pip_packages: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Build a command to run a FastAPI/Starlette app with uvicorn.

        This is a convenience method for FastAPI apps that writes the source
        to a file and runs it with uvicorn.

        Args:
            host: Host to bind to (default: 0.0.0.0)
            port: Port to bind to (default: 8000)
            pip_packages: Additional pip packages to install

        Returns:
            A command list for running with uvicorn

        Example:
            >>> packager = SourcePackager("fastapi_app.py")
            >>> cmd = packager.build_uvicorn_command(port=8000)
        """
        packages = list(pip_packages or [])

        # Ensure we have uvicorn and fastapi
        for pkg in ["fastapi", "uvicorn", "pydantic"]:
            if pkg not in packages:
                packages.append(pkg)

        packages_str = " ".join(packages)

        # Write code to file and run with uvicorn
        # This approach is more reliable than heredoc for complex apps
        # Use python3 for broader compatibility
        script = (
            f"pip install -q {packages_str} && "
            f"cat > /tmp/app.py <<'PYCODE'\n{self.code}\nPYCODE\n && "
            f"python3 /tmp/app.py"
        )

        return ["bash", "-c", script]

    @staticmethod
    def from_string(code: str) -> "SourcePackager":
        """
        Create a SourcePackager from an inline code string.

        This explicitly treats the input as code, not a file path.
        Use this when your code might look like a file path.

        Args:
            code: Python source code as a string

        Returns:
            A SourcePackager instance

        Example:
            >>> packager = SourcePackager.from_string("app.py = 'not a file'")
        """
        packager = object.__new__(SourcePackager)
        packager.source = code
        packager.code = code
        packager.is_file = False

        if not code.strip():
            raise SourceError("Source code is empty")

        return packager

    @staticmethod
    def from_file(path: Union[str, Path]) -> "SourcePackager":
        """
        Create a SourcePackager from a file path.

        This explicitly treats the input as a file path.
        Raises an error if the file doesn't exist.

        Args:
            path: Path to a Python source file

        Returns:
            A SourcePackager instance

        Raises:
            SourceError: If the file doesn't exist

        Example:
            >>> packager = SourcePackager.from_file("~/projects/app.py")
        """
        path_str = str(path)
        expanded = os.path.expanduser(path_str)

        if not os.path.isfile(expanded):
            raise SourceError(f"Source file '{path_str}' not found", source_path=path_str)

        return SourcePackager(path_str)

    def __repr__(self) -> str:
        source_type = "file" if self.is_file else "inline"
        source_preview = self.source[:50] + "..." if len(self.source) > 50 else self.source
        return f"SourcePackager({source_type}: {source_preview!r})"

    @staticmethod
    def from_function(func: Callable, call: bool = True) -> "SourcePackager":
        """
        Create a SourcePackager from a Python function.

        Uses inspect.getsource() to extract the function's source code.
        Optionally appends a call to the function at the end.

        Args:
            func: A Python function to extract source from
            call: If True, append a call to the function (default: True)

        Returns:
            A SourcePackager instance with the function's source

        Raises:
            SourceError: If the function's source cannot be retrieved

        Example:
            >>> def my_server():
            ...     from http.server import HTTPServer, BaseHTTPRequestHandler
            ...     HTTPServer(('', 8000), BaseHTTPRequestHandler).serve_forever()
            >>> packager = SourcePackager.from_function(my_server)
            >>> # Source includes the function definition + call

        Note:
            inspect.getsource() requires the function to be defined in a file,
            not in an interactive session or exec/eval context.
        """
        if not callable(func):
            raise SourceError(f"Expected a callable, got {type(func).__name__}")

        try:
            source = inspect.getsource(func)
        except (OSError, TypeError) as e:
            raise SourceError(
                f"Cannot get source for {func.__name__}: {e}. "
                "Ensure the function is defined in a file, not interactively."
            )

        # Dedent the source in case it was defined inside another scope
        import textwrap
        source = textwrap.dedent(source)

        # Append function call if requested
        if call:
            source += f"\n\n{func.__name__}()"

        packager = object.__new__(SourcePackager)
        packager.source = f"<function {func.__name__}>"
        packager.code = source
        packager.is_file = False

        return packager
