"""
Sandbox module for Basilica SDK (Rust-backed).

Provides API for running code in isolated sandboxes.
This module preserves the previous Python API, but delegates to Rust bindings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from basilica._basilica import (
    Sandbox as _RustSandbox,
    SandboxConfig as _SandboxConfig,
    SandboxResourceSpec as _SandboxResourceSpec,
    SandboxGpuSpec as _SandboxGpuSpec,
    SandboxEnvVar as _SandboxEnvVar,
    NetworkIsolation as _NetworkIsolation,
    SandboxState,
    SandboxStatus,
    ExecResult,
    FileInfo,
    SnapshotInfo,
    GitCloneResult,
    GitStatusResult,
    GitCommitResult,
    GitPushResult,
    GitPullResult,
    LspCapabilities,
    CompletionItem,
    HoverResult,
    Diagnostic,
    Location,
)

# Backwards-compatible alias
Snapshot = SnapshotInfo


class NetworkIsolation(Enum):
    """Network isolation mode."""

    NONE = "none"
    EGRESS = "egress"
    FULL = "full"

    def to_rust(self) -> _NetworkIsolation:
        if self == NetworkIsolation.NONE:
            return getattr(_NetworkIsolation, "None")
        if self == NetworkIsolation.EGRESS:
            return _NetworkIsolation.Egress
        return _NetworkIsolation.Full


@dataclass
class GpuSpec:
    """GPU requirements for sandbox."""

    count: int
    model: List[str] = field(default_factory=list)
    min_cuda_version: Optional[str] = None
    min_gpu_memory_gb: Optional[int] = None

    def to_rust(self) -> _GpuSpec:
        return _SandboxGpuSpec(
            count=self.count,
            model=self.model,
            min_cuda_version=self.min_cuda_version,
            min_gpu_memory_gb=self.min_gpu_memory_gb,
        )


@dataclass
class ResourceSpec:
    """Resource requirements for sandbox."""

    cpu: str = "500m"
    memory: str = "512Mi"
    gpus: Optional[GpuSpec] = None

    def to_rust(self) -> _SandboxResourceSpec:
        return _SandboxResourceSpec(
            cpu=self.cpu,
            memory=self.memory,
            gpus=self.gpus.to_rust() if self.gpus else None,
        )


# =============================================================================
# Exceptions (kept for compatibility)
# =============================================================================


class SandboxError(Exception):
    """Base exception for sandbox errors."""

    pass


class SandboxNotFound(SandboxError):
    """Sandbox not found."""

    def __init__(self, sandbox_id: str):
        super().__init__(f"Sandbox not found: {sandbox_id}")


class SandboxNotReady(SandboxError):
    """Sandbox is not ready for execution."""

    def __init__(self, sandbox_id: str, state: str):
        super().__init__(f"Sandbox {sandbox_id} is not ready (state: {state})")


class ExecutionError(SandboxError):
    """Command execution failed."""

    def __init__(self, message: str, exit_code: int, stderr: str):
        super().__init__(f"{message}: {stderr}")


# =============================================================================
# Namespace Classes for Improved DX (delegating to Rust)
# =============================================================================


class SandboxFiles:
    """File operations namespace for Sandbox."""

    def __init__(self, sandbox: "Sandbox"):
        self._sandbox = sandbox

    def _resolve_path(self, path: str) -> str:
        if not path.startswith("/"):
            return f"/workspace/{path}"
        if path.startswith("/sandbox"):
            # TODO: Deprecate /sandbox paths once callers migrate to /workspace.
            return f"/workspace{path[len('/sandbox'):]}"
        return path

    def read(self, path: str, encoding: str = "utf-8") -> str:
        # TODO: Support non-UTF8 encodings for binary files.
        return self._sandbox.read_file(self._resolve_path(path), encoding)

    def write(self, path: str, content: str, mode: Optional[str] = None) -> None:
        # TODO: Translate `mode` into chmod if backend adds support.
        self._sandbox.write_file(self._resolve_path(path), content, mode)

    def list(self, path: str = "/workspace", recursive: bool = False) -> List[FileInfo]:
        return self._sandbox.list_files(self._resolve_path(path), recursive)

    def exists(self, path: str) -> bool:
        try:
            self._sandbox.read_file(self._resolve_path(path))
            return True
        except SandboxError:
            return False


class SandboxProcess:
    """Process execution namespace for Sandbox."""

    def __init__(self, sandbox: "Sandbox"):
        self._sandbox = sandbox

    def run(
        self,
        code: str,
        entrypoint: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> ExecResult:
        return self._sandbox.run(code, entrypoint, args, env, timeout)

    def exec(
        self,
        command: List[str],
        cwd: Optional[str] = None,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: int = 300,
    ) -> ExecResult:
        return self._sandbox.exec(command, cwd, stdin, env, timeout)


class SandboxGit:
    """Git operations namespace for Sandbox."""

    def __init__(self, sandbox: "Sandbox"):
        self._sandbox = sandbox

    def clone(
        self,
        url: str,
        path: str = "/workspace/repo",
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        auth_token: Optional[str] = None,
    ) -> GitCloneResult:
        # TODO: Wire auth_token once backend supports it.
        return self._sandbox.git_clone(url, path, branch, depth, auth_token)

    def status(self, path: str = "/workspace/repo") -> GitStatusResult:
        return self._sandbox.git_status(path)

    def commit(
        self,
        message: str,
        path: str = "/workspace/repo",
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
    ) -> GitCommitResult:
        author = None
        if author_name and author_email:
            author = f"{author_name} <{author_email}>"
        return self._sandbox.git_commit(message, path, author)

    def push(
        self,
        path: str = "/workspace/repo",
        remote: str = "origin",
        branch: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> GitPushResult:
        # TODO: Wire auth_token once backend supports it.
        return self._sandbox.git_push(path, remote, branch, auth_token)

    def pull(
        self,
        path: str = "/workspace/repo",
        remote: str = "origin",
        branch: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> GitPullResult:
        # TODO: Wire auth_token once backend supports it.
        return self._sandbox.git_pull(path, remote, branch, auth_token)


# =============================================================================
# Sandbox Wrapper
# =============================================================================


class Sandbox:
    """A Basilica sandbox for running code in isolation (Rust-backed)."""

    def __init__(self, inner: _RustSandbox):
        self._inner = inner
        self._files: Optional[SandboxFiles] = None
        self._process: Optional[SandboxProcess] = None
        self._git: Optional[SandboxGit] = None

    @property
    def sandbox_id(self) -> str:
        return self._inner.id()

    @property
    def language(self) -> str:
        return self._inner.language()

    @property
    def files(self) -> SandboxFiles:
        if self._files is None:
            self._files = SandboxFiles(self)
        return self._files

    @property
    def process(self) -> SandboxProcess:
        if self._process is None:
            self._process = SandboxProcess(self)
        return self._process

    @property
    def git(self) -> SandboxGit:
        if self._git is None:
            self._git = SandboxGit(self)
        return self._git

    @property
    def state(self) -> SandboxState:
        return self._inner.status().state

    @property
    def is_ready(self) -> bool:
        return self.state in (SandboxState.Ready, SandboxState.Executing)

    def __enter__(self) -> "Sandbox":
        if not self.is_ready:
            self.wait_until_ready()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self.delete()
        except Exception:
            pass

    @classmethod
    def create(
        cls,
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
    ) -> "Sandbox":
        from basilica import get_config
        config = get_config()
        api_url = api_url or config.api_url
        api_key = api_key or config.api_key or os.environ.get("BASILICA_API_TOKEN", "")
        if not api_key:
            raise SandboxError("API key is required. Set BASILICA_API_TOKEN or call basilica.configure().")

        resources = ResourceSpec(cpu=cpu, memory=memory, gpus=GpuSpec(gpu_count, gpu_models or []) if gpu_count else None)
        env_vars = [_SandboxEnvVar(k, v) for k, v in (env or {}).items()]

        rust_config = _SandboxConfig(
            language=language,
            runtime=runtime,
            image=image,
            resources=resources.to_rust(),
            env=env_vars,
            timeout_seconds=timeout_seconds,
            idle_timeout_seconds=idle_timeout_seconds,
            auto_snapshot=auto_snapshot,
            restore_from=restore_from,
            network_isolation=network_isolation.to_rust(),
            namespace=None,
        )

        try:
            inner = _RustSandbox.create(api_url, api_key, rust_config)
            sandbox = cls(inner)
            if wait:
                sandbox.wait_until_ready(wait_timeout)
                # Ensure /sandbox path exists for legacy callers
                # TODO: Remove once all callers use /workspace paths.
                try:
                    sandbox.process.exec(["bash", "-lc", "test -e /sandbox || ln -s /workspace /sandbox"])
                except Exception:
                    pass
            return sandbox
        except Exception as e:
            raise SandboxError(str(e)) from e

    @classmethod
    def get(
        cls,
        sandbox_id: str,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> "Sandbox":
        from basilica import get_config
        config = get_config()
        api_url = api_url or config.api_url
        api_key = api_key or config.api_key or os.environ.get("BASILICA_API_TOKEN", "")
        if not api_key:
            raise SandboxError("API key is required. Set BASILICA_API_TOKEN or call basilica.configure().")
        try:
            return cls(_RustSandbox.get(api_url, api_key, sandbox_id))
        except Exception as e:
            raise SandboxError(str(e)) from e

    def refresh(self) -> SandboxStatus:
        return self._inner.status()

    def wait_until_ready(self, timeout: int = 300) -> SandboxStatus:
        return self._inner.wait_until_ready(timeout)

    def run(
        self,
        code: str,
        entrypoint: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult:
        if not self.is_ready:
            self.wait_until_ready()
        return self._inner.run(code, entrypoint, args, env, timeout_seconds)

    def exec(
        self,
        command: List[str],
        workdir: Optional[str] = None,
        stdin: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 300,
    ) -> ExecResult:
        if not self.is_ready:
            self.wait_until_ready()
        return self._inner.exec(command, workdir, stdin, env, timeout_seconds)

    def read_file(self, path: str, encoding: str = "utf-8") -> str:
        # TODO: Support `encoding` selection once Rust binding adds it.
        return self._inner.read_file(path)

    def write_file(self, path: str, content: str, mode: Optional[str] = None) -> None:
        # TODO: Support `mode` once Rust binding adds chmod support.
        self._inner.write_file(path, content)

    def list_files(self, path: str = "/workspace", recursive: bool = False) -> List[FileInfo]:
        return self._inner.list_files(path, recursive)

    def create_snapshot(self, name: Optional[str] = None) -> SnapshotInfo:
        return self._inner.create_snapshot(name)

    # Git operations
    def git_clone(
        self,
        url: str,
        path: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        auth_token: Optional[str] = None,
    ) -> GitCloneResult:
        # auth_token currently unused in API
        return self._inner.git_clone(url, path, branch, depth)

    def git_status(self, path: Optional[str] = None) -> GitStatusResult:
        return self._inner.git_status(path)

    def git_commit(
        self,
        message: str,
        path: Optional[str] = None,
        author: Optional[str] = None,
    ) -> GitCommitResult:
        return self._inner.git_commit(message, path, author)

    def git_push(
        self,
        path: Optional[str] = None,
        remote: str = "origin",
        branch: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> GitPushResult:
        # auth_token currently unused in API
        return self._inner.git_push(path, remote, branch)

    def git_pull(
        self,
        path: Optional[str] = None,
        remote: str = "origin",
        branch: Optional[str] = None,
        auth_token: Optional[str] = None,
    ) -> GitPullResult:
        # auth_token currently unused in API
        return self._inner.git_pull(path, remote, branch)

    # LSP
    def lsp_init(self, language: Optional[str] = None, root_path: str = "/workspace") -> LspCapabilities:
        return self._inner.lsp_init(language, root_path)

    def lsp_completion(self, file: str, line: int, character: int) -> List[CompletionItem]:
        return self._inner.lsp_completion(file, line, character)

    def lsp_hover(self, file: str, line: int, character: int) -> Optional[HoverResult]:
        return self._inner.lsp_hover(file, line, character)

    def lsp_definition(self, file: str, line: int, character: int) -> List[Location]:
        return self._inner.lsp_definition(file, line, character)

    def lsp_did_open(self, file: str, content: Optional[str] = None) -> None:
        if content is None:
            try:
                content = self.read_file(file)
            except Exception:
                content = ""
        self._inner.lsp_did_open(file, content)

    def lsp_did_change(self, file: str, content: str) -> None:
        self._inner.lsp_did_change(file, content)

    def lsp_shutdown(self) -> None:
        self._inner.lsp_shutdown()

    def delete(self) -> None:
        self._inner.delete()

    def __repr__(self) -> str:
        return f"Sandbox(id={self.sandbox_id!r}, language={self.language!r})"


