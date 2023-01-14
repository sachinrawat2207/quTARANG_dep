================
Running quTARANG
================


Packages Required
-----------------
One can run quTARANG in a single cpu as well as single gpu. You need to set that inside the ``para.py`` file whoose descripsion is given in the 
subsequent section. 
You need the following libraries to run quTARANG:

pathlib, h5py, shutils, numpy. 

You can install the above package by using the following commands:

.. code-block:: bash

    pip install pathlib
    pip install h5py
    pip install shutils
    pip install numpy

If need to require cupy library for gpu run.
In order to install cupy you must have installed a compatible version of CUDA for your GPU. and then install the ``cupy`` library for your cuda version by 
using the command mentioned in the following link:

https://docs.cupy.dev/en/stable/install.html


Setting initial condition and parameters
----------------------------------------

The structure of the files inside the **quTARANG** is as follows:

::

    ├── fns.py
    ├── gpe.py
    ├── init_cond.py
    ├── input
    ├── main.py
    ├── my_fft.py
    ├── op.py
    ├── para.py
    ├── postprocessing
    │   ├── plot_energy.py
    │   └── plot_spectrum.py
    ├── preprocessing
    │   └── set_initcond.py
    ├── set_device.py
    ├── solver.py
    └── univ_arr.py

You need to follow following steps to run the code

#. Set the parameters inside the ``para.py`` file.

    There is a ``para.py`` file inside the **qutarang** folder, where you need to set the parameters and change the path of the input and output files. 
    The parameters are explained in the following section. The parameters you need to chage inside the ``para.py`` file is as follows:

    .. code-block:: python

        op_path = Path('/home/phyguest/Sachin/Code_dev/modified1/output/') # Path of the output folder
        real_dtype = np.float64
        complex_dtype = np.complex128
        # Device Setting 
        device = 'cpu'                    # Choose the device: "cpu", "gpu"
        device_rank = 1                   # Set GPU no in case of more then 1 GPUs in system default is '0'

        # Set grid size 
        Nx = 256
        Ny = 256
        Nz = 1

        # Set box length
        Lx = 29
        Ly = 29
        Lz = 1

        # Set filename if input from file is true
        input_from_file = False   
        filename = 'wfc_t96.hdf5'  

        # Choose initial condition if input from file is False otherwise ignore it
        initcond = 2

        tmax = 1                       # Maximum time
        dt = 0.001

        # Choose the value of the non linerarity
        g = 0    

        # Choose the scheme need to implement in the code
        scheme = 'TSSP'
        evolution = 'real'            # set real for real time evolution and imag for imaginary time evolution

        save_wfc = True               # Set True to save the wfc
        save_wfc_interval = 100       # Wfc save interval

        save_energy = True             # Set True to save the energy 
        save_energy_interval = 10      # Energy save interval

        print_energy_interval = 20


    After setting the parameters you can run ``main.py`` file and the output will be stored under the output folder whosse path is mentioned on the ``para.py`` file. 


#. Set initial conditon

    You can give initial condition in terms of wavefunction and potential by changing and the following section inside the  ``preprocessing/set_initcond.py`` file. 

    .. code-block:: python

        # Set wavefunction
        wfc = (1/ncp.pi**(1/4)) * ncp.exp(-(x**2/2 + y**2/2 + z**2/2))  

        # Set potential 
        V = (x**2 + y**2 + z**2)/2   


    ``wfc`` variable will set the initial wavefunction and ``V`` variable will set the initial potential. After that you need to run the ``set_initcond.py`` file. 
    The file will generated will stored insdied the ``input`` folder.

#. Run the code by running the following command:

.. code-block:: bash

    python main.py