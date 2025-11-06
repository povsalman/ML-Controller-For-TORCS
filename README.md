# ML-Based Controller for TORCS

This repository contains an ML-based controller for the TORCS (The Open Racing Car Simulator) racing simulator, developed as the AI 2002 final project.

The objective of this project is to optimize race performance by creating a controller that uses telemetry data and machine learning (Behavior Cloning) to drive the car.

## Project Approach

### 1. Telemetry Data

Telemetry data was extracted from multiple tracks (e.g., E-Track3, Dirt2, G-Speedway). The `carState.py` and `msgParser.py` modules handle parsing UDP messages from the TORCS server, collecting:
* 19 track sensors
* 36 opponent sensors
* Speed (X, Y, Z), RPM, angle, and lap time
* 4 wheel spin velocities

### 2. Machine Learning Model (Behavior Cloning)

The controller uses a Behavior Cloning (BC) approach, implemented in `train_supervised.py` and executed by `driver.py`.

* **Architecture:** A Convolutional Neural Network (Conv1D) combined with a Gated Recurrent Unit (GRU) to process sequential telemetry data (sequence length of 5).
* **Model Details:** Conv1D (32 filters) -> GRU (64 units) -> Dense (64 units, ReLU) -> Dropout (0.2).
* **Training:** The model was trained on data from E-Track3 using an Adam optimizer and MSE loss, achieving a best validation loss of 0.0813.
* **Outputs:** The model predicts 5 control outputs: **steering, acceleration, braking, clutch, and gear.**

## Performance

The trained model (`bc_model.keras`) achieved an overall R²-based accuracy of **90.12%** on the validation set.
* **SpeedX Prediction:** 92.34% accuracy
* **Angle Prediction:** 88.90% accuracy
* **Clutch Prediction:** 85.67% accuracy
* **Gear Prediction:** 83.12% accuracy

## Code Structure

The primary project code is located in the `client-final/` directory:
* `pyclient.py`: Main Python client to run the simulation and connect the driver.
* `driver.py`: Implements the `Driver` class that loads the ML model and controls the car.
* `train_supervised.py`: Script used to train the CNN-GRU model.
* `carState.py` & `msgParser.py`: Modules for parsing telemetry data.
* `bc_model.keras`: The final trained model file.

---

## Base TORCS Information (Original README)

First a big welcome, I hope you will enjoy your ride:-) Do not forget to look into the TORCS program menu, it contains very interesting things (e.g. a link to the online track generator, track editor, documentation, etc.).

Kind regards

Bernhard

### 1. Licenses

Not the whole TORCS distribution content has the same license. Non free in the GPL or Free Art License sense are:
- The cars in cars/kc-* and cars/pw-* directories, have a look at the specific readme.txt files in those directories (rally and classic cars).
- The stripe utility. It is free for noncommercial use and is included with permission form Steven Skiena. See http://www.cs.sunysb.edu/~stripe for details. It is used by the accc tool. The accc tool is used for creating cars and advanced tracks.

### 2. TORCS Requirements

You need at least a 600MHz CPU (800MHz recommended), 128 MB RAM (256MB recommended) and an OpenGL 1.3 capable graphics card with 32MB RAM (64MB recommended). Make sure that you have installed the latest sound and graphics drivers.

### 3. Running TORCS

- Read the "How-To Drive" on www.torcs.org (http://torcs.sourceforge.net/index.php?name=Sections&op=viewarticle&artid=10).
- Press F1 anytime to get help (do it as well during the ride to learn about the options).
- Read the FAQ to learn about split screen multiplayer and more.
- Or watch videos, you find the links on www.torcs.org

**(Note: To run this specific ML driver, you will need to run the Python client, e.g., `pyclient.py`, which loads the AI driver from `driver.py`.)**

### 4. Getting Help

First have a look at the available documentation on www.torcs.org and www.berniw.org. If you find no solution for the problem learn in the FAQ how and where to report a problem. The best place to get help is the torcs-users mailing list, you find the link on www.torcs.org.

### 5. Car Setups

Since 1.3.5 there is a car setup screen integrated in TORCS, if you run a practice or qualifying session as human player you can hit the "esc" (escape) key, then choose in the menu "setup car, restart", or if the track has a pit, make a pit stop and hit there the "setup" button. Your setups are stored then in "drivers/human/*.xml". For some information about the properties look into the robot tutorial chapter 5 (http://www.berniw.org/tutorials/robot/ch5/properties.html).

### 6. Creating Tracks

The track editor is included in the TORCS distribution for Windows. It is automatically installed if you select to install the "Trackeditor and Tools". Alternatively you can get it from http://sourceforge.net/projects/trackeditor or http://www.berniw.org/trb/download/trackeditor-0.6.2c.tar.bz2. The sources are included in the jars.

### 7. Robot (AI driver) programming

You find a robot programming tutorial on www.berniw.org in the TORCS section. Have a look at the FAQ as well.

### 8. Official Championships

Visit www.berniw.org/trb for more information.
