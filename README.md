# README #

This repository is a bundle of computer vision tools. Currently, the available tools are:

- __The automatic Galvos calibration:__ With a specific setup, this program allows you to calibrate the galvos
with the software automatically with very little setup and surveillance.

- __A Galvos - Rangefinder - Camera calibration module:__ its purpose is to calibrate the camera accurately with 
a Galvos and a rangefinder (laser).

- __A grid orientation finder module:__ its purpose is to find the calibration grid's orientation compared to a perfect
  perpendicular projection plane

- __A calibration correction tool:__ its purpose is to help calibrate the galvos by achieving a precision of 0.2 mm on
  the grid

## Project Setup ##

The project has been developed in Python3.8, be sure to use this version.

First run "pip install -r requirements.txt" in order to install all required python libraries.

If you are on linux, you also need to install opencv. Follow the instruction on this link (
linux): https://docs.opencv.org/3.4/d7/d9f/tutorial_linux_install.html

For the modules using IdsCameras, you will also need to install the ueye software suite of your cameras here:
https://www.ids-imaging.us/downloads.html

(Last camera package used is: https://www.ids-imaging.us/download-details/AB02302.html)

### Camera-Galvos-Rangefinder Calibration Module ###

run "python3 main.py -calibrate-ids-camera-to-laser-and-galvos" to run the calibration of the camera to the rangefinder
and the Galvos

### Galvos Calibration Module ###

NOTE: Since we need the controller module, using the corrector need superuser privileges.

The computer you are using to run the command must be connected to 4 Ids Cameras with a specific setup detailed here:

Launch the Galvos calibration software until you are projecting your first point on the grid.

Launch the program "sudo python3 main.py -correct --n [Your number of point cloud]" (verbosity is recommended).

### ~~Camera - Galvos Calibration Procedure~~ ###

(Obsolete)

First you have to take 20 - 50 pictures of a chessboard with the camera as explained and shown here
https://mechasys.atlassian.net/wiki/spaces/0E/pages/edit-v2/1154514947 and put them in a folder.

Do the same thing with a projected laser rectangle pattern on it (described further on the document) for 10-20 pictures.

Then go to the core/calibration/config.py and edit all variables that differs from the previous setup
(path to the files, laser pattern used, ...)

Once you have finished this setup, on a terminal run

"python3 main.py -calibrate-plan"

It is recommended to increase the verbosity to have details of the calibration:

"python3 main.py -calibrate-camera -vvv"

The error during the camera calibration should be less than 0.5 pixels and less than 1.5 for the galvos-camera
calibration.

If the calibration errors are too high, it is most plausible some pictures you took create high error. Calibrate using
maximum verbosity (-vvv) and look for high error images.

You can simply remove them from the folder, or if you don't have enough, correct them using Gimp or photoshop by
darkening problematic areas.

### ~~Grid orientation Module~~ ###

(Obsolete)

It is important to understand this module requires a Galvos-Camera calibration in order to work. If the position of the
Camera and / or Galvos has been changed, you have to follow the whole calibration procedure.

If they haven't changed, you can directly go to the "Find grid angle" step

### ~~Find grid angle~~ ###

(OBSOLETE)

If the setup of the camera-galvos hasn't changed from the last time, you can only run:

- python3 main.py --angles [PATH TO YOUR IMAGE]

If you had to do the calibration recently, whether you have to write on "core/calibration/config.py" the results of:

- CAM_MTX
- CAM_DIST
- KEXT_GUESS

And then run

- python3 main.py -calibrate-plan --angles [PATH TO YOUR IMAGE]

Or, if you don't want to overwrite the values of previous calibration, you can calibrate and then find the angles :

- python3 main.py --angles [PATH TO YOUR IMAGE]

## Contribution guidelines ##

Developed by Ivan Alt

Reviewed by Mondher Souilah and Mohammad Edabi

Property of Mechasys
