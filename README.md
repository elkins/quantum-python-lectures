quantum-python-lectures
=======================

This is a series of self-study lectures on using Python for scientific
computing at the graduate level in atomic physics and quantum optics.

**🔄 Modernized for Python 3.9+** - Updated December 2025 with modern Python features including type hints, f-strings, and current best practices. All Jupyter notebooks updated to v4.5 format.

It aims to introduce you to using Python in both theoretical and experimental contexts through some common in-lab examples, like:

- Reading data from a photon counter
- Binning and smoothing data
- Finding the steady state of an open quantum system
- Making a publication-quality plot

This is **not** an introduction to programming nor Python. You don't need to install anything to read the lectures, but if you want to download and use the example code it is a prerequisite that you already have Python working on your computer along with the standard scientific computing libraries: Numpy, Scipy and Matplotlib.

## Installation

### Requirements
- Python 3.9 or later
- Standard scientific computing libraries (NumPy, SciPy, Matplotlib)

### Quick Start
```bash
# Clone the repository
git clone https://github.com/elkins/quantum-python-lectures.git
cd quantum-python-lectures

# Install dependencies
pip install -r requirements.txt

# Or use modern pip install
pip install -e .

# Launch Jupyter Lab
jupyter lab
```

## Modern Python Features

This updated version includes:
- ✅ Type hints on all functions
- ✅ Modern string formatting (f-strings)
- ✅ Proper package structure with `pyproject.toml`
- ✅ Updated dependencies for Python 3.9+
- ✅ Improved docstrings following NumPy/Google style
- ✅ Elimination of deprecated patterns (pylab, mutable defaults)
- ✅ Modern matplotlib and numpy idioms
- ✅ All Jupyter notebooks converted to v4.5 format
- ✅ Compatible with JupyterLab 4.5.0

If you need help with Python or getting it installed there are many resources online, including the <a href="http://labs.physics.dur.ac.uk/computing/resources/python.php">Durham Physics Lab Guide to Python</a>. We&rsquo;ve listed more on the <a href="{{ site.baseurl }}/resources/">Resources</a> page.

## Development

### Testing
The codebase includes comprehensive test coverage for numerical methods and lineshape analysis:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_ode_int.py -v

# Generate coverage report
pytest tests/ --cov=code --cov-report=term --cov-report=html
```

Current test coverage: **92% overall** (91% on `ode_int.py`, 94% on `lineshape_analysis.py`)

### Performance Benchmarks
The repository includes comprehensive benchmarks comparing ODE integration methods:

```bash
# Run all benchmarks
pytest benchmarks/benchmark_ode.py --benchmark-only -v

# View detailed results
cat benchmarks/BENCHMARK_RESULTS.md
```

Key findings:
- **RK4** offers best accuracy/speed trade-off for non-stiff problems
- **Implicit Euler** required for stiff problems (stable with k=100)
- **Explicit Euler** is 4.5x faster but less accurate
- See `benchmarks/BENCHMARK_RESULTS.md` for detailed performance analysis

### Code Quality
Pre-commit hooks are configured for automated code quality checks:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Configured tools:
- **black**: Code formatting
- **ruff**: Fast Python linter (all checks passing with 92% test coverage)
- **mypy**: Static type checking
- **pytest**: Run tests on commit with pytest-benchmark for performance tracking

### Linting
The project uses ruff with comprehensive checks enabled:

```bash
# Run linter
ruff check code/ tests/

# Auto-fix issues where possible
ruff check code/ tests/ --fix
```

### Contributing
When contributing code:
1. Ensure all tests pass
2. Add tests for new functionality
3. Follow existing code style (type hints, f-strings, NumPy docstrings)
4. Run pre-commit hooks before committing

The lectures are in four sections: I/O, Plotting, Data Analysis and Numerical Methods.

## Lectures

### I/O

  <ol>
   <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/1_Reading-and-Writing-Files.ipynb">Reading and Writing Files</a></li>
  </ol>

### Plotting

  <ol start="2">
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/2_Publication-Quality-Plot.ipynb">Publication quality plot</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/3_Lineshape-Comparison-and-Analysis.ipynb">Lineshape Comparison and Analysis</a></li>
  </ol>

### Data Analysis

  <ol start="4">
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/4_Fitting-Data-to-Theory.ipynb">Fitting Data to Theory</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/5_Smoothing-and-Binning-Data.ipynb">Smoothing and Binning</a></li>
  </ol>

### Integrating <abbr title="Ordinary Differential Equations">ODEs</abbr>

  <ol start="6">
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/6_The-Explicit-Euler-Method-and-Order-of-Accuracy.ipynb">The Explicit Euler Method and Order of Accuracy</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/7_The-Runge-Kutta-Method-Higher-Order-ODEs-and-Multistep-Methods.ipynb">The Runge-Kutta Method, Higher-Order ODEs and Multistep Methods</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/8_Stiff-Problems-Implicit-Methods-and-Computational-Cost.ipynb">Stiff Problems, Implicit Methods and Computational Cost</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/9_Integrating-with-SciPy-and-QuTiP.ipynb">Integrating with SciPy and QuTiP</a></li>
  </ol>

### Monte Carlo Methods

  <ol start="10">
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/10_Monte-Carlo-Calculating-Pi.ipynb">Calculating π</a></li>
    <li><a href="http://nbviewer.ipython.org/urls/github.com/tommyogden/quantum-python-lectures/blob/master/11_Monte-Carlo-Maxwell-Boltzmann-Distributions.ipynb">Maxwell-Boltzmann Distributions</a></li>
  </ol>
