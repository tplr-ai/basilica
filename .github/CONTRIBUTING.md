# Contributing to Basilica

Thanks for your interest in contributing to Basilica. We welcome contributions from the community.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/basilica.git`
3. Add upstream remote: `git remote add upstream https://github.com/one-covenant/basilica.git`
4. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

- Rust 1.75.0 or higher
- Docker (for containerized execution)
- CUDA toolkit (for GPU support)

### Building

```bash
# Build all crates
cargo build

# Build specific crate
cargo build -p basilica-validator
cargo build -p basilica-miner
cargo build -p basilica-executor

# Build with release optimizations
cargo build --release
```

### Testing

```bash
# Run all tests
cargo test

# Run tests for specific crate
cargo test -p basilica-common

# Run tests with logging
RUST_LOG=debug cargo test
```

## Code Standards

### Style Guidelines

- Follow Rust standard style guidelines
- Run `cargo fmt` before committing
- Run `cargo clippy` and fix all warnings
- Keep functions under 50 lines when possible
- Write self-documenting code with clear variable names

### Code Quality

```bash
# Format code
cargo fmt

# Check for linting issues
cargo clippy

# Fix linting issues and format
just fix
```

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb: "Add", "Fix", "Update", "Remove"
- Keep the first line under 50 characters
- Add detailed description if needed after a blank line

Example:
```
Fix weight calculation in validator scoring

- Correct GPU performance normalization
- Add bounds checking for weight values
- Update tests to cover edge cases
```

## Pull Request Process

1. Update your fork with latest upstream changes:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. Rebase your feature branch:
   ```bash
   git checkout feature/your-feature-name
   git rebase main
   ```

3. Run tests and ensure they pass:
   ```bash
   cargo test
   cargo fmt --check
   cargo clippy
   ```

4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

5. Create a pull request:
   - Use the PR template
   - Link related issues
   - Provide clear description of changes
   - Include test results

## Testing Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve code coverage
- Test edge cases and error conditions

## Documentation

- Update documentation for any API changes
- Add inline documentation for complex logic
- Update README if adding new features
- Include examples where helpful

## Security

- Never commit secrets or private keys
- Use environment variables for sensitive data
- Follow security best practices
- Report vulnerabilities via Discord

## Community

- Join our [Discord](https://discord.gg/tsErZGXX) for discussions
- Be respectful and constructive
- Help others when you can
- Share your ideas and feedback

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
