# INSTALLATION GUIDE
NB: We worked on the PACS virtual machine, so everything have been tested only inside that enviroment.
## Configuration of the virtual machine

Configurazione virtual machine e moduli (mettiamo anche il link che ci ha dato africa?)

load_modules.sh -> da caricare prima di lanciare i programmi

## Download and installation of IsoGlib
-Download IsoGlib library (private in bitbucket) e andare nella nostra branch “Learning_pacs_anedp_project_guido_vidulis”

-Compilare isoglib:
• forniamo il file di configurazione per isoglib (In una cartella della repo OpenNN Forse meglio in isoglib? Sennò bisogna invertire questo step e il successivo)
• source configure.sh
• in BuildRelease: make
–isoglib nella stessa cartella di OpenNN

## Download and installation of OpenNN
-Download della repo OpenNN da GitHub
Compilare OpenNN:
+ mkdir build
+ cd build
+ cmake ..

-> risolvere il problema dei file dati che non si copiano da examples a build (Fatto?)

## How to use the provided test
-> in data creare una cartella in cui mettiamo vari possibili dataset da usare (sono in OpenNN/PACS_ANEDP_Project/SUPGDataset)
e scrivere che per usarlo va spostato in data e rinominato SUPG.txt

-File di post processing in matlab (ora sono nella cartella OpenNN/PACS_ANEDP_Project/PostProcessing)

## How to add a test
