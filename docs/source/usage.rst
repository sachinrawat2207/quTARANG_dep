================
Running quTARANG
================


Packages Required
-----------------
One can run quTARANG in a single cpu as well as single gpu.

You need the following libraries to run quTARANG:

pathlib, h5py, shutils, numpy. 

You can install the above package by using the following commands:

.. code-block:: bash

    pip install pathlib
    pip install h5py
    pip install numpy

If need to require cupy library for gpu run.
In order to install cupy you must have installed a compatible version of CUDA for your GPU. and then install the ``cupy`` library for your cuda version by 
using the compatible version by using the following link:

https://docs.cupy.dev/en/stable/install.html


Running the code
----------------

To run a simulation:

#. Import the required libraries

    .. code-block:: python
        
        from quTARANG import xp, Params, GPE


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
    Create an instance of the GPE class by passing the Params instance created previously.

    .. code-block:: python

        # Create an instance of the GPE class.
        G = gpe.GPE(par)

#. Set initial conditon

    You can give initial condition in terms of wavefunction and potential by defining their functions and passing them to the function ``set_init``.

    .. code-block:: python

        # Set wavefunction
        wfc = (1/xp.pi**(1/4)) * xp.exp(-(x**2/2 + y**2/2 + z**2/2))  

        # Set potential 
        pot = (x**2 + y**2 + z**2)/2

        G.set_init(wfc, pot)

    ``wfc`` function will be used to set the initial wavefunction and ``pot`` variable will be used to set the initial potential.

#. Start the simulation:

    .. code-block:: python
        
        G.evolve()

The results are stored as hdf5 files in the cwd or the path set by the user in the Params instance.