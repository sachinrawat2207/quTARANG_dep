# quTARANG

Welcome to quTARANG, a fast GPU enabled python solver to solve Gross-Pitaeskii equation. 


### Capablities of ``quTARANG``
It is a solver developed to study turbulence in quantum systems specially in Bose-Einstein condensates. It can run on both GPU and CPU.   

The documentation of code is available at [quTARANG](https://quTARANG.readthedocs.io/en/latest/). 


### Dependencies
``quTARANG`` depends on the following packages:
- ``numpy``
- ``cupy`` (If you want to use GPU)
- ``pathlib``
- ``h5py``
- ``matplotlib``

### Instllation
You can install ``quTARANG`` using ``pip``

```python
   pip install quTARANG  
```


### How to use:

To run a simulation:

1. Import the required libraries

    ```python
        from quTARANG import xp, Params, GPE
    ```


2. Set the parameters

    Create an instance of the ``Params`` class and set the parameters according to your need.
    The parameters have been detailed in the documentation. Example:

    ```python
    # Create an instance of the Params class for storing parameters.
        par = gpe.Params(N = [64, 64, 64],
                     L = [16, 16, 16],
                     g = 0.1,
                     dt = 0.001,
                     tmax = 5,
                     rms = [True, 0, 100])
    ```

3. Initiate ``GPE`` class
    Create an instance of the GPE class by passing the Params instance created previously.

    ```python
    # Create an instance of the GPE class.
       G = gpe.GPE(par)
    ```

4. Set initial conditon

    You can give initial condition in terms of wavefunction and potential by defining their functions and passing them to the function ``set_init``.

    ```python

        # Set wavefunction
        wfc = (1/xp.pi**(1/4)) * xp.exp(-(x**2/2 + y**2/2 + z**2/2))
        
        # Set potential 
        pot = (x**2 + y**2 + z**2)/2

        G.set_init(wfc, pot)
    ```

    ``wfc`` function will be used to set the initial wavefunction and ``pot`` variable will be used to set the initial potential.

5. Start the simulation:

    ```python
        G.evolve()
    ```
The results are stored as hdf5 files in the cwd or the path set by the user in the Params instance.


