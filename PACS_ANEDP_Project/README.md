# PACS_ANEDP_Project
Margherita Guido, Michele Vidulis

In this folder we have collected all the file needed for the pre and post
processing phase of the construction of our OpenNN example ``OpenNN/examples/SUPG``,
where we enclose what we have implemented for our Project (Courses: Advanced
Programming for Scientific Computing, Prof. Luca Formaggia and Numerical Analysis
for Partial Differential Equations, Prof. Paola Antonietti), supervised by
Prof. Luca Dedè.

## DatasetGeneration
- It contains the Matlab script ``generate_dataset.m``, used to generate the
dataset ``SUPG.txt``needed by our example.
This dataset is generated using the exact solution of the PDE problem we are
considering.

  Every row of the dataset contains:
  * mu: input of the network
  * the exact solution computed in the gauss point of the mesh. In our example
  the computational domain is a rectangle and we already know the number of DofS
  and Elements of the mesh.

- In the folder ``SUPGDataset`` there are many dataset already generated for
 both test cases we have considered.
 They are called ``SUPG_N_M_K`` and they are generated using as inputs
 ` mu = linspace(M, K, N)`


 ## PostProcessing
- SUPGOutput
  are the results of the SUPG program
    * the file `output.txt`is an example of full output given by the execution of `OpenNN/build/examples/SUPG/SUPG`.
    * in the folders corresponding to the different Tests, there are file of type `output_N_M_K`. Here, for every dataset of type ``SUPG_N_M_K``, we collected only the relevant outputs: Training error, selection error, Tesing inputs and testing outputs.

- ``after_training_and_testing.m`` is needed to read the results of the training and testing of the  neural network from file of type `output_N_M_K`, and plot them to analyze the goodness of the results.

- ``best_tau_isoglib.m `` uses the file stored in the folder `IsoGlib_Errors`, computed using IsoGlib library, to search the best stabilization parameter with respect to the L2 norm.

- ``plot_tau.m `` is an helpful script used to compare the trend of the stabilization parameter computed in different ways.Ú
