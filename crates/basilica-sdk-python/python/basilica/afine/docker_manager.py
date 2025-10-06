"""
Docker image lifecycle management for Basilica AFINE SDK.
"""

import os
from pathlib import Path
from typing import Optional

import docker


class DockerManager:
    """Manages Docker image lifecycle for Basilica services."""

    def __init__(self) -> None:
        """Initialize Docker manager."""
        self._client = docker.from_env()

    def build_image(
        self,
        path: Path,
        tag: str,
        dockerfile: str = "Dockerfile",
        service_file: str = "service.py"
    ) -> str:
        """
        Build Docker image from path.

        Args:
            path: Path to build context
            tag: Image tag
            dockerfile: Dockerfile name
            service_file: Service file name (default: service.py)

        Returns:
            Image ID
        """
        dockerfile_path = path / dockerfile
        if not dockerfile_path.exists():
            self.generate_dockerfile(path, service_file)

        dockerignore_path = path / ".dockerignore"
        if not dockerignore_path.exists():
            self.generate_dockerignore(path)

        image, build_logs = self._client.images.build(
            path=str(path),
            tag=tag,
            dockerfile=dockerfile,
            rm=True
        )

        return image.id

    def push_image(self, tag: str, registry: Optional[str] = None) -> None:
        """
        Push image to Docker Hub or private registry.

        Args:
            tag: Image tag
            registry: Optional registry URL (defaults to Docker Hub)
        """
        for line in self._client.images.push(tag, stream=True, decode=True):
            if 'error' in line:
                raise RuntimeError(f"Failed to push image: {line['error']}")

    def pull_image(self, tag: str) -> str:
        """
        Pull image from Docker Hub or private registry.

        Args:
            tag: Image tag

        Returns:
            Image ID
        """
        image = self._client.images.pull(tag)
        return image.id

    def login_registry(
        self,
        registry: str = "docker.io",
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> None:
        """
        Authenticate with Docker registry.

        Args:
            registry: Registry URL
            username: Registry username (defaults to DOCKER_HUB_USERNAME env var)
            password: Registry password/token (defaults to DOCKER_HUB_TOKEN env var)
        """
        username = username or os.environ.get("DOCKER_HUB_USERNAME")
        password = password or os.environ.get("DOCKER_HUB_TOKEN")

        if not username or not password:
            raise ValueError(
                "Docker registry credentials not provided. "
                "Set DOCKER_HUB_USERNAME and DOCKER_HUB_TOKEN environment variables."
            )

        self._client.login(
            username=username,
            password=password,
            registry=registry
        )

    def generate_dockerfile(
        self,
        path: Path,
        service_file: str = "service.py",
        requirements_file: str = "requirements.txt"
    ) -> None:
        """
        Generate a Dockerfile optimized for Basilica services.

        Args:
            path: Path to service directory
            service_file: Service file name
            requirements_file: Requirements file name
        """
        has_requirements = (path / requirements_file).exists()

        dockerfile_content = f"""FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
{f'COPY {requirements_file} .' if has_requirements else ''}
{f'RUN pip install --no-cache-dir -r {requirements_file}' if has_requirements else ''}

# Install basilica SDK and dependencies
RUN pip install --no-cache-dir basilica uvicorn fastapi pydantic cloudpickle tenacity httpx

# Copy application code
COPY {service_file} .

# Create state directory for persistence
RUN mkdir -p /app/state

# Expose service port
EXPOSE 8000

# Run service
CMD ["python", "{service_file}"]
"""

        (path / "Dockerfile").write_text(dockerfile_content)

    def generate_dockerignore(self, path: Path) -> None:
        """
        Generate .dockerignore to exclude unnecessary files from build context.

        Args:
            path: Path to service directory
        """
        dockerignore_content = """# Version control
.git
.gitignore
.gitattributes

# Python
__pycache__
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
.hypothesis/

# Documentation
docs/
*.md
LICENSE

# Environment files
.env
.env.local

# Large files
*.log
*.zip
*.tar.gz
"""

        (path / ".dockerignore").write_text(dockerignore_content)
