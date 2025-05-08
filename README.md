Wavefield Propagation in PyTorch
========


<p>Finite difference wavefield propagation and visualization as an exersice.
Currently built for 2D media with Mur absorbing boundary condition.
<p/>

![wave_propagation](.github/wave_propagation.gif)

# Benchmarks
Propagation on CPU for 1200x800 velocity grid with 1x1 step.<br>
```python
%%timeit
solve_one_step(wavefield, tau, kappa, laplacian_kernel, device="cpu")
```
\>>> 33.5 ms ± 1.75 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)<br>
Propagation on GPU for 1200x800 velocity grid with 1x1 step.<br><br>
```python
%%timeit
solve_one_step(wavefield, tau, kappa, laplacian_kernel, device="cuda")
```
\>>> 1.55 ms ± 61.8 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)<br>