# EIT_Epsilon

## Overview

This is your new Kedro project with Kedro-Viz setup, which was generated using `kedro 0.19.3`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`


## Set up your Python virtual environment

### Creating an environment using Conda

We strongly recommend installing [Miniconda](https://docs.anaconda.com/free/miniconda/index.html) as your virtual environment manager if you don’t already use it.

1. Install Miniconda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/
2. Create a new virtual environment using conda
   - `conda create --name <name_of_your_choice> python=3.11.9 -y`
   - In this example, we use Python 3.11.9, but you can opt for a different version if you need it for your
     particular project
3. Activate the new environment: `conda activate <name_of_your_choice>`


### How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

Pre-commit is enabled for this project. To install the required hooks, run:

```
pre-commit install
```

### Installing the exact environment used in the project

You can also use the exact Conda environment used for development. To do this, follow the steps below:
1. Install MiniConda as explained above.
2. Create the EIT-Epsilon environment using the environment.yml file
  - `conda env create -f environment.yml`
3. Activate the new environment: `conda activate EIT-Epsilon`

### Installing System dependencies for `eccodes` package
The Python module depends on the ECMWF ecCodes library that must be installed on the system and accessible as a shared library.

Or if you manage binary packages with Conda use:

$ conda install -c conda-forge eccodes

You may run a simple selfcheck command to ensure that your system is set up correctly:

````commandline
python -m eccodes selfcheck
Found: ecCodes v2.27.0.
Your system is ready.
````


## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/test_data_science.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## IMPORTANT - HOW to update Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

If you created your environment with Conda using the environment file, you can use `conda env update --file environment.yml --prune` to update your environment to match the environment file again. 

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).
## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html).
