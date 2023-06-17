# Marr-Hildreth Edge Detection

The Marr-Hildreth algorithm implemented in Python with NumPy.

### Usage as CLI tool

Install with:

```pip install 'marr-hildreth @ git+https://github.com/gpapadok/marr-hildreth.git'```

Run:

```marr lena.jpg --sigma 2.5 --threshold 0.7```

Note: Marr-hildreth is an outdated algorithm and this implementation is inefficient for large images or large values of sigma of the Gaussian. Purely for educational purposes.

<img src="lena.jpg" alt="Lena" width="300"/>|
<img src="edges.jpg" alt="Lena edges" width="300"/>

Source: [https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm](https://en.wikipedia.org/wiki/Marr%E2%80%93Hildreth_algorithm)
