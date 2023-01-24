================
Running quTARANG
================


Packages Required
-----------------
One can run quTARANG on a single CPU as well as a single GPU.

You need the following libraries to run quTARANG:

pathlib, h5py, numpy (to run code on CPU), cupy (to run code on GPU). 

You can install the above package by using the following commands:

.. code-block:: bash

    pip install pathlib
    pip install h5py
    pip install numpy

``cupy`` library is required for GPU run. In order to install ``cupy`` you must first install a compatible version of ``CUDA`` for your GPU. Then install the ``cupy`` library for your cuda version. For more details:
https://docs.cupy.dev/en/stable/install.html

Running the code
----------------

To run a simulation:

#. Import the required libraries

    .. code-block:: python
        
        from quTARANG import Params, GPE
        import numpy as xp 
        # import cupy as xp # For GPU run


#. Set the parameters

    Create an instance of the ``Params`` class and set the parameters according to your need.
    The parameters have been detailed in the documentation. Example:

    .. code-block:: python

        # Create an instance of the Params class for storing parameters.
        par = gpe.Params(N = [64, 64, 64],
                     L = [16, 16, 16],
                     g = 0.1,
                     dt = 0.001,
                     tmax = 5,
                     rms = [True, 0, 100])
                     
   
#. Initiate ``GPE`` class
    
    Create an instance of the ``GPE`` class by passing the ``Params`` instance created previously.

    .. code-block:: python

        # Create an instance of the GPE class.
        G = gpe.GPE(par)

#. Set initial conditon

    You can give initial condition in terms of wavefunction and potential by defining their functions and passing them to the function ``set_init``.

    .. code-block:: python

        # Set wavefunction
        wfc = lambda x, y=0, z=0: (1/xp.pi**(1/4)) * xp.exp(-(x**2/2 + y**2/2 + z**2/2))  

        # Set potential 
        pot = lambda x, y=0, z=0: (x**2 + y**2 + z**2)/2

        G.set_init(wfc, pot)

    ``wfc`` function will be used to set the initial wavefunction and ``pot`` variable will be used to set the initial potential.

#. Start the simulation:

    .. code-block:: python
        
        G.evolve()

The results are stored as hdf5 files in the cwd or the path set by the user in the Params instance.
