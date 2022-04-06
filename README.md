# Deep active subspace data
Data and Jupyter notebooks to reproduce the results of:

W.N. Edeling, "On the deep active subspace method", (submitted), 2021.

We applied the deep-active subspace method to:

* An HIV model consisting of 7 coupled ordinary differential equations, with 27 uncertain input parameters.

* A COVID19 model with 51 inputs parameters.

See the paper above for more information.

## Contents

To reproduce the results of the HIV model, the following Jupyter notebook are present:

* `HIV/HIV.ipynb`: reproduce the results of the scalar quantities of interest.

* `HIV/HIV_vector.ipynb`: reproduce the results of the vector-values quantity of interest.

To reproduce the results for the COVID19 model, run

* `COVID19/COVID19.ipynb`

All required training data is also present in the `HIV` and `COVID19` directories, see the notebooks for a description.

### Funding

This research is funded by the European Union Horizon 2020 research and innovation programme under grant agreement \#800925 ([VECMA](https://www.vecma.eu) project).
