# INSTALLATION GUIDE

## Configuration of the virtual machine
Configurazione virtual machine e moduli (mettiamo anche il link che ci ha dato africa?)

load_modules.sh -> da caricare prima di lanciare i programmi

## Download and installation of IsoGlib
Download IsoGlib library from the provided link (private on bitbucket.org) and switch to branch `Learning_pacs_anedp_project_guido_vidulis`

Compile the library:

• forniamo il file di configurazione per isoglib (In una cartella della repo OpenNN Forse meglio in isoglib? Sennò bisogna invertire questo step e il successivo)

+ source configure.sh
+ in BuildRelease: make

## Download and installation of OpenNN
Download OpenNN from https://github.com/margheguido/OpenNN.git.

NB: Download OpenNN in the same folder of isoglib, since linking phase have been tested only in this conditions.

To compile it type:
+ mkdir build
+ cd build
+ cmake ..

The library comes with some examples (`OpenNN/examples`). We implemented the Neural Network as a new example named SUPG (folder `OpenNN/examples/SUPG`) which will be built automatically.

-> risolvere il problema dei file dati che non si copiano da examples a build (Fatto?)

## How to use the provided test
-> in data creare una cartella in cui mettiamo vari possibili dataset da usare (sono in OpenNN/PACS_ANEDP_Project/SUPGDataset)
e scrivere che per usarlo va spostato in data e rinominato SUPG.txt

-File di post processing in matlab (ora sono nella cartella OpenNN/PACS_ANEDP_Project/PostProcessing)

## How to add a test

NB: We worked on the PACS virtual machine, so everything have been tested only inside that enviroment.
