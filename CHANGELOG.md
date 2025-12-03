# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-12-02

### Added - Major Modernization

#### Testing Infrastructure
- **Comprehensive test suite** with 53 tests achieving 92% code coverage
  - ODE integration methods: Explicit Euler, RK4, Implicit Euler, Adams-Bashforth
  - Lineshape analysis: Gaussian, Lorentzian, Voigt functions
  - Edge cases, numerical stability tests, and error handling
- **pytest** configuration with coverage reporting
- Test files: `tests/test_ode_int.py`, `tests/test_lineshapes.py`

#### Performance Benchmarking
- **10 comprehensive benchmarks** comparing ODE integration methods
  - Exponential decay, harmonic oscillator, and stiff problem tests
  - Scaling analysis for different system dimensions
  - Comparison with SciPy reference implementations
- Detailed documentation in `benchmarks/BENCHMARK_RESULTS.md`
- Key findings:
  - RK4: Best accuracy/speed trade-off (462μs)
  - Explicit Euler: Fastest but least accurate (102μs)
  - Implicit Euler: Required for stiff problems (stable with k=100)

#### Code Improvements
- **Proper Newton iteration** for implicit Euler method
  - Finite difference Jacobian computation
  - Convergence tolerance and iteration limits
  - Handles stiff problems (k=100) without numerical instability
  - Fallback for singular matrices
- **Modern type hints** following PEP 585
  - Replaced deprecated `Tuple`/`Dict` with `tuple`/`dict`
  - Added `Union` types for numpy scalar compatibility
- **Docstring examples** for key functions with usage patterns
- Fixed numerical precision issues in implicit Euler convergence

#### Development Tools
- **Pre-commit hooks** for automated code quality
  - black: Code formatting
  - ruff: Fast Python linter (all checks passing)
  - mypy: Static type checking
  - pytest: Run tests before commit
- Configuration file: `.pre-commit-config.yaml`
- **pyproject.toml** with modern Python 3.9+ configuration
- **requirements.txt** with pinned dependencies and dev tools
- `.gitignore` for Python artifacts and IDE files

#### Documentation
- **Updated README.md** with comprehensive Development section
  - Testing instructions with coverage reporting
  - Performance benchmarking guide
  - Code quality tools and linting
  - Contributing guidelines
- **BENCHMARK_RESULTS.md** with detailed performance analysis
  - Method comparisons across problem types
  - Recommendations for different use cases
  - Accuracy vs speed trade-offs documented

### Changed

#### Code Quality
- Coverage improved from 54% to 92%
- All ruff linting checks passing with appropriate per-file ignores
- Type hints on all public functions
- Consistent code formatting with black

#### Dependencies
- Minimum Python version: 3.9+
- NumPy >= 1.24.0
- SciPy >= 1.10.0
- Matplotlib >= 3.7.0
- Added dev dependencies: pytest, pytest-cov, pytest-benchmark, black, ruff, mypy

### Technical Details

#### ODE Integration
- Implicit Euler now uses proper Newton's method with:
  - Jacobian approximation: `J[i,j] ≈ (f(y + ε*e_j) - f(y)) / ε`
  - Newton step: `(I - h*J)*δy = -residual`
  - Convergence check: `||residual|| < tolerance`
  - Maximum 10 iterations per time step

#### Test Coverage Breakdown
- `ode_int.py`: 91% coverage (85/94 lines)
- `lineshape_analysis.py`: 94% coverage (95/101 lines)
- 8 new test classes with comprehensive edge case coverage

#### Benchmark Results Summary
| Method | Mean Time (μs) | Best For |
|--------|---------------|----------|
| Explicit Euler | 101.8 | Quick prototyping |
| SciPy RK45 | 115.6 | Best accuracy (adaptive) |
| Adams-Bashforth | 189.8 | Smooth problems |
| Runge-Kutta 4 | 462.7 | Default choice |
| Implicit Euler | 1,073.9 | Stiff problems |

### Notes for Maintainers

This is a significant modernization that maintains backward compatibility while adding:
- Professional-grade testing and benchmarking
- Modern Python 3.9+ idioms and type hints
- Development tooling for long-term maintainability
- Comprehensive documentation

All Jupyter notebooks remain functional and have been validated.

---

## [1.x] - Previous versions

See git history for changes before modernization.
