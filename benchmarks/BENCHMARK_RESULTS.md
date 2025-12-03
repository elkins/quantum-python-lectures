# ODE Integration Performance Benchmarks

Benchmarks comparing different ODE integration methods on various test problems.

## System Configuration
- Python 3.12.10
- NumPy 1.24+
- SciPy 1.10+
- macOS (Apple Silicon)

## Test Problems

### 1. Exponential Decay (dy/dt = -k*y)
Simple non-stiff problem with analytical solution.
- Time span: [0, 2]
- Steps: 101
- k = 1.0

### 2. Harmonic Oscillator (d²x/dt² = -ω²x)
2D system testing energy conservation.
- Time span: [0, 2π]
- Steps: 201
- ω = 1.0

### 3. Stiff Problem (dy/dt = -k*y, k=100)
Stiff decay problem testing stability.
- Time span: [0, 1]
- Steps: 51
- k = 100.0

## Performance Results

### Exponential Decay Problem

| Method | Mean Time (μs) | Speedup vs RK4 | Max Error |
|--------|---------------|----------------|-----------|
| Explicit Euler | 101.8 | 4.54x | ~5e-2 |
| SciPy RK45 (reference) | 115.6 | 4.00x | ~1e-10 |
| Adams-Bashforth | 189.8 | 2.44x | ~1e-3 |
| **Runge-Kutta 4** | **462.7** | **1.00x** | **~1e-6** |
| Implicit Euler | 1,073.9 | 0.43x | ~1e-1 |

**Key Findings:**
- **Explicit Euler** is fastest but least accurate (max error ~0.05)
- **RK4** provides excellent accuracy (error ~1e-6) at moderate cost
- **SciPy RK45** is surprisingly fast with best accuracy (adaptive stepping)
- **Implicit Euler** is slower due to Newton iteration overhead
- **Adams-Bashforth** offers good balance for smooth problems

### Harmonic Oscillator (Energy Conservation)

| Method | Mean Time (μs) | Energy Drift |
|--------|---------------|--------------|
| RK4 | 928.8 | < 0.01 |

**Key Findings:**
- RK4 maintains excellent energy conservation for oscillatory problems
- Critical for long-time integration of conservative systems

### Stiff Problem (k=100)

| Method | Mean Time (μs) | Stability |
|--------|---------------|-----------|
| RK4 | 235.3 | ✅ Stable |
| **Implicit Euler** | **257.8** | ✅ **Stable** |

**Key Findings:**
- **Implicit Euler** handles stiff problems reliably
- Only ~10% slower than RK4 for this problem
- Explicit methods would require much smaller time steps for stability
- For very stiff problems (k>1000), implicit methods become essential

### Scaling with Problem Dimension

| System Size | Mean Time (μs) |
|-------------|---------------|
| 2D (oscillator) | 934.6 |
| 10D (coupled) | 1,178.1 |

**Scaling:** ~1.26x increase for 5x larger system
- Sub-linear scaling due to efficient NumPy vectorization
- Function evaluation dominates for larger systems

## Recommendations

### For Non-Stiff Problems:
1. **Runge-Kutta 4** - Best default choice (accuracy vs speed)
2. **SciPy RK45** - Best accuracy with adaptive stepping
3. **Adams-Bashforth** - Good for smooth, well-behaved problems

### For Stiff Problems:
1. **Implicit Euler** - Required for stability
2. Explicit methods will fail or require impractically small steps

### For Long-Time Integration:
1. **RK4** - Maintains energy conservation for Hamiltonian systems
2. Higher-order methods reduce cumulative error

### For Quick Prototyping:
1. **Explicit Euler** - Fastest, adequate for gentle problems and initial testing
2. Be aware of accuracy limitations

## Accuracy vs Speed Trade-offs

```
Speed:     Explicit Euler > SciPy RK45 > Adams-Bashforth > RK4 > Implicit Euler
Accuracy:  SciPy RK45 > RK4 > Adams-Bashforth > Implicit Euler > Explicit Euler
Stability: Implicit Euler > RK4 ≈ SciPy RK45 > Adams-Bashforth > Explicit Euler
```

## Running Benchmarks

```bash
# Run all benchmarks
pytest benchmarks/benchmark_ode.py --benchmark-only -v

# Run with detailed statistics
pytest benchmarks/benchmark_ode.py --benchmark-only \
  --benchmark-columns=mean,stddev,min,max,rounds

# Save benchmark results
pytest benchmarks/benchmark_ode.py --benchmark-only \
  --benchmark-save=ode_results
```

## Notes

- Benchmarks use `pytest-benchmark` with automatic calibration
- Each benchmark runs multiple rounds for statistical significance
- Results may vary based on system load and hardware
- Focus on relative performance rather than absolute times
