# Contributing to Quarry

Thank you for your interest in contributing to quarry. This document outlines the process for contributing code, reporting issues, and proposing features.

## Code of Conduct

- Be respectful and inclusive in all interactions
- Focus on what is best for the community and the project
- Gracefully accept constructive criticism

## Reporting Bugs

Before submitting a bug report, please:

1. Search existing issues to avoid duplicates
2. Test with the latest `main` branch
3. Gather diagnostic information

Submit bugs using the [bug report template](/.github/ISSUE_TEMPLATE/bug_report.md). Include:

- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Go version, Rust version, GPU)
- Relevant logs

## Proposing Features

For significant changes, start a discussion before implementing:

1. Open an issue with the [feature request template](/.github/ISSUE_TEMPLATE/feature_request.md)
2. Describe the problem you're solving
3. Outline your proposed solution
4. Wait for maintainer feedback before starting work

Small improvements (documentation, minor bug fixes) can be submitted directly via pull request.

## Development Setup

### Prerequisites

- Go 1.22+
- Rust 1.75+ with clippy and rustfmt
- CUDA 13.0 (for GPU development)
- Git 2.40+

### Initial Setup

```bash
# Clone the repository
git clone https://github.com/your-org/quarry.git
cd quarry

# Install Go dependencies
cd crawler
go mod download
cd ..

# Install Rust dependencies
cd semantic-search
cargo fetch
cd ..

# Download and export the ONNX model
cd semantic-search
./scripts/download_model.sh
python scripts/export_onnx.py --output models/
cd ..

# Run tests to verify setup
make test
```

### Project Structure

```
quarry/
├── crawler/           # Go crawler source
│   ├── cmd/           # CLI entry points
│   ├── internal/      # Internal packages
│   └── tests/         # Go tests
├── semantic-search/   # Rust semantic search source
│   ├── crates/        # Library crates
│   ├── bin/           # Binary crates
│   └── tests/         # Integration tests
├── tests/             # End-to-end tests
│   ├── e2e/           # Pipeline tests
│   ├── go/            # Go test utilities
│   └── rust/          # Rust test utilities
└── docs/              # Documentation
```

## Pull Request Requirements

Before submitting a pull request, ensure all requirements are met:

- [ ] All existing tests pass
  ```bash
  # Go tests
  cd crawler && go test ./... -race
  
  # Rust tests
  cd semantic-search && cargo test --workspace
  ```
- [ ] New code has tests (unit tests AND edge cases)
- [ ] No `unwrap()` in Rust library code (use `?` or `expect()` with context)
- [ ] No goroutine leaks (goleak passes)
  ```bash
  cd crawler && go test ./... -tags=goleak
  ```
- [ ] CLI flags documented in relevant docs file
- [ ] config.toml fields documented in docs/semantic-search.md
- [ ] Commit messages follow conventional commits format

## Code Style

### Go

```bash
# Format code
gofmt -w .

# Run linter
go vet ./...

# Run additional linters (optional but recommended)
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
golangci-lint run
```

**Guidelines:**

- Use `gofmt` formatting (no configuration needed)
- Exported functions must have documentation comments
- Handle errors explicitly (no ignored errors)
- Use `context` for cancellation in long-running operations

### Rust

```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

**Guidelines:**

- Use `rustfmt` with default configuration
- All public items must have documentation comments with examples
- Avoid `unwrap()` in library code; use `Result` with meaningful error types
- Use `clippy` suggestions for idiomatic code

## Commit Message Format

Use conventional commits:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `refactor` | Code change without feature/fix |
| `perf` | Performance improvement |
| `chore` | Build process or auxiliary tool change |

**Examples:**

```
feat(crawler): add support for HTTP/2 connections

Enable HTTP/2 for improved connection multiplexing on servers
that support it. Falls back to HTTP/1.1 for compatibility.

Closes #123
```

```
fix(indexer): correct HNSW distance calculation for cosine similarity

The previous implementation used Euclidean distance, which gave
incorrect rankings for normalized vectors.
```

```
docs(readme): update installation instructions for macOS

Add Homebrew commands for Go and Rust installation.
```

## Branch Naming

Use descriptive branch names with prefixes:

| Prefix | Example | Purpose |
|--------|---------|---------|
| `feature/` | `feature/http2-support` | New features |
| `fix/` | `fix/hnsw-distance-calc` | Bug fixes |
| `docs/` | `docs/installation-guide` | Documentation |
| `test/` | `test/edge-case-redirects` | Test improvements |
| `refactor/` | `refactor/dedup-implementation` | Code refactoring |

## Review Process

1. **Submission**: Open a pull request against `main`
2. **Automated Checks**: CI runs tests, linting, and formatting checks
3. **Code Review**: Maintainer reviews within 3 business days
4. **Address Feedback**: Make requested changes
5. **Approval**: Once approved, a maintainer will merge

### Review Criteria

- Code correctness and test coverage
- Adherence to style guidelines
- Documentation completeness
- Backward compatibility (unless breaking change is intended)
- Performance implications for MX250 target hardware

### After Merge

- Delete your feature branch
- Your commit will be in `main`
- If applicable, update documentation for the next release

## Development Workflow

### Making Changes

1. Create a feature branch from `main`
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make changes and commit incrementally
   ```bash
   git add -p
   git commit -m "feat(scope): description"
   ```

3. Keep your branch up to date
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. Push and open a pull request
   ```bash
   git push origin feature/my-feature
   ```

### Running Tests Locally

```bash
# All tests
make test

# Go tests only
make test-go

# Rust tests only
make test-rust

# E2E tests only
make test-e2e

# With coverage
make coverage
```

### Debugging

For Go debugging:

```bash
# Verbose test output
go test ./... -v -run TestSpecificFunction

# With delve debugger
go install github.com/go-delve/delve/cmd/dlv@latest
dlv test ./internal/crawler -- -test.run TestCrawler
```

For Rust debugging:

```bash
# Verbose test output
RUST_LOG=debug cargo test -- --nocapture

# With lldb
rustup component add lldb
rust-lldb ./target/debug/search
```

## Release Process

Releases are handled by maintainers:

1. Update version in `Cargo.toml` and `go.mod`
2. Update `CHANGELOG.md` with changes since last release
3. Tag release: `git tag v1.x.x`
4. Push tag: `git push origin v1.x.x`
5. CI builds and publishes binaries

## Getting Help

- **Documentation**: Check the `docs/` directory
- **Issues**: Search existing issues or open a new one
- **Discussions**: Use GitHub Discussions for questions

Thank you for contributing to quarry!
