Installation
============

Colony Differentiation can be downloaded from GitHub and used in a Python environment.

To use the software, download or clone the Colony Differentiation repository from GitHub.

In the downloaded directory, install the module and its required dependencies using:

.. code-block:: bash

   pip install .


Requirements
------------

Colony Differentiation requires Python 3.11 or higher and the following third-party packages:

- NumPy
- SciPy (>= 1.14)
- Matplotlib (>= 3.8)
- scikit-learn
- OpenCV
- rawpy
- Pillow
- tqdm
- termcolor
- IPython
- ipywidgets
- ipympl

All required dependencies are installed automatically when installing the package.


Setup
-----

The software can be used in a Python environment, such as Jupyter Notebook.


Graphical User Interface
------------------------

Colony Differentiation also provides a graphical user interface based on Streamlit, which is currently under development.

To install the additional dependencies, run:

.. code-block:: bash

   pip install streamlit streamlit-drawable-canvas

To launch the interface, run the following command from the repository directory:

.. code-block:: bash

   python -m streamlit run app.py

The interface opens in a web browser. Some features may be incomplete or subject to change.