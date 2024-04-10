# LinAlgBook

C++ version of the [code](https://github.com/mikexcohen/LinAlgBook) for "Linear Algebra: Theory, Intuition, Code" book by Mike X Cohen. Minimum language version is [C++20](https://en.cppreference.com/w/Template:cpp/compiler_support/20).

As configured, package management requires [vcpkg](https://vcpkg.io/en/). Dependencies are [Armadillo](https://arma.sourceforge.net/), [Matplot++](https://github.com/alandefreitas/matplotplusplus), [stb](https://github.com/nothings/stb).

For plotting, Matplot++ requires that [gnuplot](http://www.gnuplot.info/) is installed.

If using [VSCode](https://code.visualstudio.com/) (recommended), install extensions for C/C++ (+ Extension Pack), CMake (+Tools).

If using a Wayland desktop, the gnuplot viewer will fault (as of March 2024) if invoked from a VSCode terminal; execute from an external terminal.
