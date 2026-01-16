# Basilica Sandbox - Python SDK

**Secure, Isolated Code Execution for AI Agents and Untrusted Code**

[![PyPI](https://img.shields.io/pypi/v/basilica-sdk)](https://pypi.org/project/basilica-sdk/)

Basilica Sandbox provides secure, isolated environments for executing arbitrary code. Built on Firecracker microVMs and gVisor, sandboxes offer VM-level isolation while maintaining container-like performance—perfect for AI agents, code evaluation systems, and interactive development environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Creating Sandboxes](#creating-sandboxes)
5. [Code Execution](#code-execution)
6. [File Operations](#file-operations)
7. [Git Operations](#git-operations)
8. [Language Server Protocol (LSP)](#language-server-protocol-lsp)
9. [Snapshots](#snapshots)
10. [Security & Isolation](#security--isolation)
11. [GPU Support](#gpu-support)
12. [Error Handling](#error-handling)
13. [Configuration](#configuration)
14. [Best Practices](#best-practices)
15. [API Reference](#api-reference)
16. [Examples](#examples)

---

## Quick Start

```python
from basilica import python_sandbox

# One-liner with automatic cleanup
with python_sandbox() as sandbox:
    result = sandbox.run("print('Hello from a secure sandbox!')")
    print(result.stdout)  # Hello from a secure sandbox!
```

That's it! The sandbox is created, your code runs in complete isolation, and the environment is automatically cleaned up.

---

## Installation

```bash
pip install basilica-sdk
```

**Requirements:**
- Python 3.10+
- Basilica API token (get one at [basilica.ai](https://basilica.ai) or via `basilica tokens create`)

**Set your API token:**
```bash
export BASILICA_API_TOKEN="basilica_..."
```

---

## Core Concepts

### What is a Sandbox?

A sandbox is an isolated execution environment that provides:

| Feature | Description |
|---------|-------------|
| **Code Execution** | Run Python, JavaScript, Bash, Go, Rust, and more |
| **File System** | Isolated `/workspace` directory for your files |
| **Network Isolation** | Configurable network access (none, egress-only, or full) |
| **Resource Limits** | CPU, memory, and time limits prevent runaway processes |
| **VM-Level Isolation** | Firecracker microVMs provide hardware-level security |
| **Snapshots** | Save and restore sandbox state |

### Sandbox States

```
Creating → Initializing → Ready ⇄ Executing → Terminating → Terminated
                            ↓
                       Snapshotting
                            ↓
                          Failed
```

| State | Description |
|-------|-------------|
| `Creating` | Sandbox resources being provisioned |
| `Initializing` | Runtime environment starting up |
| `Ready` | Ready for code execution |
| `Executing` | Currently running code |
| `Snapshotting` | Creating a snapshot |
| `Terminating` | Shutting down |
| `Terminated` | Sandbox deleted |
| `Failed` | Error occurred |

---

## Creating Sandboxes

### Factory Functions (Recommended)

For common use cases, factory functions provide sensible defaults:

```python
from basilica import python_sandbox, javascript_sandbox, js_sandbox

# Python sandbox (default: container runtime)
with python_sandbox() as sb:
    sb.run("import sys; print(sys.version)")

# JavaScript/Node.js sandbox
with javascript_sandbox() as sb:
    sb.run("console.log(process.version)")

# Alias for JavaScript
with js_sandbox() as sb:
    sb.run("console.log('Hello!')")
```

### Full Control with `Sandbox.create()`

For advanced configuration:

```python
from basilica import Sandbox, NetworkIsolation

sandbox = Sandbox.create(
    # Language & Runtime
    language="python",           # python, javascript, bash, go, rust, etc.
    runtime="firecracker",       # "firecracker" (microVM), "container", or "gvisor"
    image="python:3.11-slim",    # Custom container image (optional)
    
    # Resources
    cpu="1",                     # CPU cores ("500m" = 0.5 cores)
    memory="2Gi",                # Memory limit
    
    # GPU (optional)
    gpu_count=1,
    gpu_models=["A100", "H100"],
    
    # Timeouts
    timeout_seconds=3600,        # Max lifetime (1 hour)
    idle_timeout_seconds=600,    # Auto-terminate after 10 min idle
    
    # Security
    network_isolation=NetworkIsolation.EGRESS,  # Allow outbound only
    
    # Persistence
    auto_snapshot=True,          # Save state on termination
    restore_from="snap-abc123",  # Restore from previous snapshot
    
    # Connection
    wait=True,                   # Wait for Ready state
    wait_timeout=300,            # Timeout for waiting
)

try:
    result = sandbox.run("print('configured!')")
finally:
    sandbox.delete()
```

### Getting an Existing Sandbox

```python
from basilica import Sandbox

# Reconnect to a running sandbox
sandbox = Sandbox.get("sandbox-abc12345")
print(f"State: {sandbox.state}")
```

---

## Code Execution

### Running Code

The `run()` method executes code in the sandbox's configured language:

```python
with python_sandbox() as sb:
    # Simple code execution
    result = sb.run("print('Hello!')")
    print(result.stdout)     # "Hello!\n"
    print(result.stderr)     # ""
    print(result.exit_code)  # 0
    print(result.success)    # True
    print(result.duration_ms) # Execution time
    
    # Multi-line code
    result = sb.run("""
import math
for i in range(5):
    print(f"sqrt({i}) = {math.sqrt(i):.2f}")
""")
    
    # With environment variables
    result = sb.run(
        "import os; print(os.environ['API_KEY'])",
        env={"API_KEY": "secret123"}
    )
    
    # With timeout
    result = sb.run(
        "import time; time.sleep(10)",
        timeout_seconds=5  # Will timeout
    )
```

### Executing Shell Commands

The `exec()` method runs arbitrary commands:

```python
with python_sandbox() as sb:
    # Run shell commands
    result = sb.exec(["ls", "-la", "/workspace"])
    print(result.stdout)
    
    # With working directory
    result = sb.exec(
        ["python3", "app.py"],
        workdir="/workspace/myproject"
    )
    
    # With stdin
    result = sb.exec(
        ["python3", "-c", "print(input())"],
        stdin="Hello from stdin!"
    )
    
    # With environment variables
    result = sb.exec(
        ["env"],
        env={"MY_VAR": "my_value"}
    )
```

### Namespaced Process API

For cleaner code organization:

```python
with python_sandbox() as sb:
    # Same as sb.run() and sb.exec()
    sb.process.run("print('hello')")
    sb.process.exec(["ls", "-la"], cwd="/workspace")
```

---

## File Operations

### Basic File Operations

```python
with python_sandbox() as sb:
    # Write a file
    sb.write_file("/workspace/app.py", """
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    print(hello())
""")
    
    # Read a file
    content = sb.read_file("/workspace/app.py")
    print(content)
    
    # List files
    files = sb.list_files("/workspace")
    for f in files:
        print(f"{f.name}: {f.size} bytes, dir={f.is_dir}")
    
    # Recursive listing
    all_files = sb.list_files("/workspace", recursive=True)
```

### Namespaced Files API (Recommended)

The `files` namespace provides automatic `/workspace` path prefixing:

```python
with python_sandbox() as sb:
    # Relative paths are prefixed with /workspace
    sb.files.write("app.py", "print('hello')")    # → /workspace/app.py
    sb.files.write("src/utils.py", "...")          # → /workspace/src/utils.py
    
    # Read files
    content = sb.files.read("app.py")
    
    # Check existence
    if sb.files.exists("config.json"):
        config = sb.files.read("config.json")
    
    # List files
    files = sb.files.list()                        # List /workspace
    files = sb.files.list("src", recursive=True)   # List /workspace/src
    
    # Absolute paths work too
    sb.files.write("/tmp/temp.txt", "temporary")
```

### File Permissions

```python
# Set file as executable
sb.write_file("/workspace/script.sh", "#!/bin/bash\necho hello", mode="755")
sb.exec(["./script.sh"], workdir="/workspace")
```

---

## Git Operations

Sandboxes have first-class Git support for cloning, committing, and pushing code.

### Basic Git Operations

```python
with python_sandbox() as sb:
    # Clone a repository
    result = sb.git_clone(
        url="https://github.com/user/repo.git",
        path="/workspace/repo",
        branch="main",
        depth=1  # Shallow clone
    )
    print(f"Cloned {result.branch} at {result.commit}")
    
    # Check status
    status = sb.git_status("/workspace/repo")
    print(f"Branch: {status.branch}")
    print(f"Clean: {status.clean}")
    print(f"Modified: {status.modified}")
    print(f"Untracked: {status.untracked}")
    
    # Make changes and commit
    sb.write_file("/workspace/repo/README.md", "# Updated README")
    
    commit = sb.git_commit(
        message="Update README",
        path="/workspace/repo",
        author="Agent <agent@basilica.ai>"
    )
    print(f"Created commit: {commit.commit_hash}")
    
    # Push changes
    push = sb.git_push("/workspace/repo", remote="origin", branch="main")
    if push.success:
        print("Pushed successfully!")
    
    # Pull latest
    pull = sb.git_pull("/workspace/repo")
    print(f"Pulled {pull.commits_pulled} commits")
```

### Namespaced Git API

```python
with python_sandbox() as sb:
    # Cleaner syntax with default paths
    sb.git.clone("https://github.com/user/repo.git")
    
    # Defaults to /workspace/repo
    status = sb.git.status()
    sb.git.commit("Fix bug")
    sb.git.push()
    sb.git.pull()
    
    # Custom paths
    sb.git.clone("https://github.com/other/repo.git", path="/workspace/other")
    sb.git.status(path="/workspace/other")
```

### Private Repositories

```python
# Use auth_token for private repos
sb.git.clone(
    url="https://github.com/private/repo.git",
    auth_token="ghp_xxxxxxxxxxxx"
)

# Or set via environment
sandbox = Sandbox.create(
    language="python",
    env={"GIT_TOKEN": "ghp_xxxxxxxxxxxx"}
)
```

---

## Language Server Protocol (LSP)

Sandboxes include LSP support for code intelligence features like completion, hover, and diagnostics.

### Initialize LSP

```python
with python_sandbox() as sb:
    # Write some code
    sb.files.write("app.py", """
import os

def greet(name: str) -> str:
    return f"Hello, {name}!"

message = greet("World")
print(message)
""")
    
    # Initialize LSP server
    capabilities = sb.lsp_init(language="python", root_path="/workspace")
    print(f"Completion: {capabilities.completion_provider}")
    print(f"Hover: {capabilities.hover_provider}")
    print(f"Go to definition: {capabilities.definition_provider}")
```

### Code Completion

```python
# Get completions at cursor position (line 4, after "os.")
completions = sb.lsp_completion("app.py", line=4, character=3)

for item in completions[:5]:
    print(f"{item.label}: {item.detail}")
    # environ: module attribute
    # path: module attribute
    # getcwd: function
```

### Hover Documentation

```python
# Get documentation for symbol at position
hover = sb.lsp_hover("app.py", line=3, character=4)  # Over "greet"
if hover:
    print(hover.contents)
    # def greet(name: str) -> str
    # Returns a greeting message
```

### Diagnostics (Errors/Warnings)

```python
# Get errors and warnings
diagnostics = sb.lsp_diagnostics("app.py")

for d in diagnostics:
    severity = {1: "Error", 2: "Warning", 3: "Info", 4: "Hint"}[d.severity]
    print(f"{severity} at line {d.line}: {d.message}")
```

### Go to Definition

```python
# Find where a symbol is defined
locations = sb.lsp_definition("app.py", line=7, character=10)  # "greet" call

for loc in locations:
    print(f"Defined at {loc.uri}:{loc.line}")
```

### Document Sync

```python
# Notify LSP of file changes
sb.lsp_did_open("app.py")

# After editing
new_content = sb.files.read("app.py")
sb.lsp_did_change("app.py", new_content)

# Cleanup
sb.lsp_shutdown()
```

---

## Snapshots

Snapshots capture the complete state of a sandbox's `/workspace` directory, allowing you to save and restore work.

### Creating Snapshots

```python
with python_sandbox() as sb:
    # Do some work
    sb.files.write("model.py", "# trained model code")
    sb.files.write("weights.bin", binary_data)
    
    # Create snapshot
    snapshot = sb.create_snapshot(name="trained-model-v1")
    print(f"Snapshot ID: {snapshot.snapshot_id}")
    print(f"Size: {snapshot.size_bytes / 1024 / 1024:.2f} MB")
    print(f"Created: {snapshot.created_at}")
```

### Restoring from Snapshot

```python
# Create new sandbox from snapshot
sandbox = Sandbox.create(
    language="python",
    restore_from="snap-abc123"  # Snapshot ID
)

# /workspace now contains the snapshotted files
content = sandbox.read_file("/workspace/model.py")
```

### Auto-Snapshot on Termination

```python
# Automatically create snapshot when sandbox is deleted
sandbox = Sandbox.create(
    language="python",
    auto_snapshot=True
)

# ... do work ...

# Deletion returns snapshot ID
snapshot_id = sandbox.delete()
print(f"Work saved to: {snapshot_id}")
```

---

## Security & Isolation

### Network Isolation Modes

```python
from basilica import Sandbox, NetworkIsolation

# No restrictions (default)
sandbox = Sandbox.create(
    language="python",
    network_isolation=NetworkIsolation.NONE
)

# Egress only - can reach internet but not receive connections
sandbox = Sandbox.create(
    language="python",
    network_isolation=NetworkIsolation.EGRESS
)

# Full isolation - no network access
sandbox = Sandbox.create(
    language="python",
    network_isolation=NetworkIsolation.FULL
)
```

### Runtime Options

| Runtime | Isolation Level | Performance | Use Case |
|---------|-----------------|-------------|----------|
| `firecracker` | VM (highest) | Good | Untrusted code, multi-tenant |
| `gvisor` | Syscall filtering | Better | Defense in depth |
| `container` | Container | Best | Trusted code, development |

```python
# Firecracker microVM (recommended for untrusted code)
sandbox = Sandbox.create(
    language="python",
    runtime="firecracker"
)

# gVisor for syscall filtering
sandbox = Sandbox.create(
    language="python",
    runtime="gvisor"
)

# Standard container (fastest, least isolation)
sandbox = Sandbox.create(
    language="python",
    runtime="container"
)
```

### Resource Limits

```python
sandbox = Sandbox.create(
    language="python",
    cpu="1",              # 1 CPU core
    memory="2Gi",         # 2 GB RAM
    timeout_seconds=3600, # Max 1 hour lifetime
    idle_timeout_seconds=300,  # Terminate after 5 min idle
)
```

### Security Best Practices

1. **Always use context managers** - Ensures cleanup even on errors
2. **Use Firecracker for untrusted code** - VM-level isolation
3. **Set appropriate timeouts** - Prevent runaway processes
4. **Use network isolation** - Limit attack surface
5. **Don't pass secrets as code** - Use environment variables

```python
# ✅ Good: Use context manager and env vars
with python_sandbox() as sb:
    result = sb.run(
        "import os; api_call(os.environ['SECRET'])",
        env={"SECRET": secret_value}
    )

# ❌ Bad: Secrets in code, no cleanup
sb = Sandbox.create(language="python")
sb.run(f"api_call('{secret_value}')")  # Secret leaked in logs!
# Forgot to call sb.delete()
```

---

## GPU Support

```python
from basilica import Sandbox

# Request GPU access
sandbox = Sandbox.create(
    language="python",
    gpu_count=1,
    gpu_models=["A100", "H100"],  # Acceptable GPU types
    memory="16Gi",  # GPUs often need more RAM
)

result = sandbox.run("""
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
""")
```

---

## Error Handling

### Exception Types

```python
from basilica import (
    SandboxError,       # Base exception
    SandboxNotFound,    # Sandbox doesn't exist
    SandboxNotReady,    # Sandbox not in Ready state
    ExecutionError,     # Code execution failed
)

try:
    with python_sandbox() as sb:
        result = sb.run("invalid python code !!!")
except SandboxNotFound as e:
    print(f"Sandbox {e.sandbox_id} not found")
except SandboxNotReady as e:
    print(f"Sandbox {e.sandbox_id} is {e.state}, not ready")
except ExecutionError as e:
    print(f"Execution failed (exit {e.exit_code}): {e.stderr}")
except SandboxError as e:
    print(f"Sandbox error: {e}")
```

### Checking Execution Results

```python
with python_sandbox() as sb:
    result = sb.run("exit(1)")
    
    if not result.success:
        print(f"Command failed with exit code {result.exit_code}")
        print(f"stderr: {result.stderr}")
    
    # Or check explicitly
    if result.exit_code != 0:
        raise Exception(f"Unexpected failure: {result.stderr}")
```

---

## Configuration

### Global Configuration

Set defaults that apply to all sandboxes:

```python
import basilica

# Configure once at startup
basilica.configure(
    api_url="https://api.basilica.ai",
    api_key="basilica_..."
)

# All sandboxes now use these settings
with basilica.python_sandbox() as sb:
    sb.run("print('configured!')")
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BASILICA_API_TOKEN` | API authentication token | Required |
| `BASILICA_API_URL` | API endpoint | `https://api.basilica.ai` |

### Per-Sandbox Configuration

Override globals for specific sandboxes:

```python
sandbox = Sandbox.create(
    language="python",
    api_url="https://custom.api.endpoint",
    api_key="different_token"
)
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Recommended
with python_sandbox() as sb:
    sb.run("print('safe!')")
# Automatic cleanup

# ❌ Avoid
sb = python_sandbox()
sb.run("print('risky')")
sb.delete()  # Easy to forget!
```

### 2. Handle Timeouts Gracefully

```python
with python_sandbox() as sb:
    try:
        result = sb.run(
            "while True: pass",  # Infinite loop
            timeout_seconds=5
        )
    except SandboxError as e:
        if "timeout" in str(e).lower():
            print("Code took too long, terminated")
```

### 3. Validate Untrusted Code Output

```python
with python_sandbox() as sb:
    result = sb.run(user_code)
    
    # Limit output size
    if len(result.stdout) > 10000:
        output = result.stdout[:10000] + "... (truncated)"
    
    # Check for suspicious patterns
    if "rm -rf" in user_code:
        raise SecurityError("Dangerous command detected")
```

### 4. Use Appropriate Isolation for Trust Level

```python
# Untrusted user code → Firecracker
def run_user_code(code: str) -> str:
    with Sandbox.create(
        language="python",
        runtime="firecracker",
        network_isolation=NetworkIsolation.FULL,
        timeout_seconds=30,
    ) as sb:
        return sb.run(code).stdout

# Trusted internal code → Container (faster)
def run_internal_code(code: str) -> str:
    with Sandbox.create(
        language="python",
        runtime="container",
    ) as sb:
        return sb.run(code).stdout
```

### 5. Reuse Sandboxes for Multiple Operations

```python
# ✅ Efficient: One sandbox for multiple operations
with python_sandbox() as sb:
    sb.files.write("data.json", json.dumps(data))
    sb.run("process_data()")
    sb.run("validate_results()")
    result = sb.files.read("output.json")

# ❌ Wasteful: New sandbox for each operation
for item in items:
    with python_sandbox() as sb:  # Slow!
        sb.run(f"process({item})")
```

---

## API Reference

### Sandbox Class

```python
class Sandbox:
    # Properties
    sandbox_id: str          # Unique identifier
    language: str            # Programming language
    state: SandboxState      # Current state
    is_ready: bool           # Ready for execution?
    
    # Namespaced APIs
    files: SandboxFiles      # File operations
    process: SandboxProcess  # Process execution
    git: SandboxGit          # Git operations
    
    # Class methods
    @classmethod
    def create(
        language: str = "python",
        runtime: str = "firecracker",
        image: Optional[str] = None,
        cpu: str = "500m",
        memory: str = "512Mi",
        gpu_count: Optional[int] = None,
        gpu_models: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 3600,
        idle_timeout_seconds: int = 600,
        auto_snapshot: bool = False,
        restore_from: Optional[str] = None,
        network_isolation: NetworkIsolation = NetworkIsolation.NONE,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        wait: bool = True,
        wait_timeout: int = 300,
    ) -> "Sandbox": ...
    
    @classmethod
    def get(sandbox_id: str, ...) -> "Sandbox": ...
    
    # Instance methods
    def run(
        code: str,
        entrypoint: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult: ...
    
    def exec(
        command: List[str],
        workdir: Optional[str] = None,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult: ...
    
    def read_file(path: str, encoding: str = "utf-8") -> str: ...
    def write_file(path: str, content: str, mode: Optional[str] = None) -> None: ...
    def list_files(path: str = "/workspace", recursive: bool = False) -> List[FileInfo]: ...
    
    def git_clone(url: str, path: Optional[str] = None, ...) -> GitCloneResult: ...
    def git_status(path: Optional[str] = None) -> GitStatusResult: ...
    def git_commit(message: str, ...) -> GitCommitResult: ...
    def git_push(...) -> GitPushResult: ...
    def git_pull(...) -> GitPullResult: ...
    
    def lsp_init(language: Optional[str] = None, root_path: str = "/workspace") -> LspCapabilities: ...
    def lsp_completion(file: str, line: int, character: int) -> List[CompletionItem]: ...
    def lsp_hover(file: str, line: int, character: int) -> Optional[HoverResult]: ...
    def lsp_diagnostics(file: str) -> List[Diagnostic]: ...
    def lsp_definition(file: str, line: int, character: int) -> List[Location]: ...
    
    def create_snapshot(name: Optional[str] = None) -> Snapshot: ...
    
    def refresh() -> None: ...
    def wait_until_ready(timeout: int = 300, poll_interval: float = 1.0) -> None: ...
    def delete() -> Optional[str]: ...
```

### Data Classes

```python
@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    success: bool  # Property: exit_code == 0

@dataclass
class FileInfo:
    name: str
    path: str
    is_dir: bool
    size: int
    modified_at: Optional[str]

@dataclass
class Snapshot:
    snapshot_id: str
    sandbox_id: str
    name: Optional[str]
    created_at: str
    size_bytes: int

class SandboxState(Enum):
    CREATING = "creating"
    INITIALIZING = "initializing"
    READY = "ready"
    EXECUTING = "executing"
    SNAPSHOTTING = "snapshotting"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    FAILED = "failed"

class NetworkIsolation(Enum):
    NONE = "none"      # Full network access
    EGRESS = "egress"  # Outbound only
    FULL = "full"      # No network
```

---

## Examples

### AI Agent Code Execution

```python
from basilica import python_sandbox

def execute_agent_code(code: str, context: dict) -> dict:
    """Safely execute AI-generated code."""
    with python_sandbox() as sb:
        # Provide context as files
        sb.files.write("context.json", json.dumps(context))
        
        # Run agent code with isolation
        result = sb.run(f"""
import json
context = json.load(open('context.json'))
{code}
# Write output
with open('result.json', 'w') as f:
    json.dump(result, f)
""", timeout_seconds=30)
        
        if result.success:
            return json.loads(sb.files.read("result.json"))
        else:
            raise Exception(f"Agent code failed: {result.stderr}")
```

### Interactive Jupyter-like REPL

```python
from basilica import python_sandbox

class SandboxREPL:
    def __init__(self):
        self.sandbox = python_sandbox().__enter__()
        self._history = []
    
    def execute(self, code: str) -> str:
        self._history.append(code)
        result = self.sandbox.run(code)
        return result.stdout if result.success else f"Error: {result.stderr}"
    
    def close(self):
        self.sandbox.__exit__(None, None, None)

# Usage
repl = SandboxREPL()
print(repl.execute("x = 42"))
print(repl.execute("print(x * 2)"))  # 84
repl.close()
```

### Code Evaluation System

```python
from basilica import python_sandbox
from dataclasses import dataclass
from typing import List

@dataclass
class TestCase:
    input: str
    expected_output: str

@dataclass
class EvaluationResult:
    passed: int
    failed: int
    errors: List[str]

def evaluate_submission(code: str, test_cases: List[TestCase]) -> EvaluationResult:
    """Evaluate student code against test cases."""
    results = EvaluationResult(passed=0, failed=0, errors=[])
    
    with python_sandbox() as sb:
        sb.files.write("solution.py", code)
        
        for i, tc in enumerate(test_cases):
            result = sb.run(
                f"from solution import solve; print(solve({tc.input!r}))",
                timeout_seconds=5
            )
            
            if not result.success:
                results.errors.append(f"Test {i}: Runtime error")
                results.failed += 1
            elif result.stdout.strip() == tc.expected_output:
                results.passed += 1
            else:
                results.errors.append(
                    f"Test {i}: Expected {tc.expected_output!r}, got {result.stdout.strip()!r}"
                )
                results.failed += 1
    
    return results
```

### Multi-Language Support

```python
from basilica import Sandbox

def run_polyglot(snippets: dict[str, str]) -> dict[str, str]:
    """Run code in multiple languages."""
    results = {}
    
    for language, code in snippets.items():
        with Sandbox.create(language=language) as sb:
            result = sb.run(code)
            results[language] = result.stdout
    
    return results

# Example
outputs = run_polyglot({
    "python": "print('Hello from Python!')",
    "javascript": "console.log('Hello from JavaScript!')",
    "bash": "echo 'Hello from Bash!'",
})
```

---

## Troubleshooting

### Sandbox Creation Timeout

```python
# Increase wait timeout
sandbox = Sandbox.create(
    language="python",
    wait_timeout=600  # 10 minutes
)
```

### Network Issues in Isolated Mode

```python
# Use EGRESS mode if you need outbound access
sandbox = Sandbox.create(
    language="python",
    network_isolation=NetworkIsolation.EGRESS
)

# Pre-install packages if using FULL isolation
sandbox = Sandbox.create(
    language="python",
    image="my-registry/python-with-deps:latest",  # Custom image with deps
    network_isolation=NetworkIsolation.FULL
)
```

### Out of Memory

```python
# Increase memory limit
sandbox = Sandbox.create(
    language="python",
    memory="4Gi",
    cpu="2"
)
```

---

## License

MIT OR Apache-2.0

## Links

- [Basilica Documentation](https://docs.basilica.ai)
- [SDK GitHub](https://github.com/one-covenant/basilica)
- [PyPI Package](https://pypi.org/project/basilica-sdk/)
- [Issue Tracker](https://github.com/one-covenant/basilica/issues)

