# Dino-AI
An AI to teach Google Chrome's dinosaur to jump obstacles. The sample of a working model can be viewed at https://www.linkedin.com/feed/update/urn:li:activity:6439502467947331584

## Getting started
All the code was written on Python, using Tensorflow.

### Prerequisites
This program has been succesfully built using:

```
dash_core_components==0.26.0
tensorflow==1.10.0
numpy==1.15.0
dash_html_components==0.11.0
dash==0.23.1
plotly==3.1.0
pynput==1.4
selenium==3.13.0
```

Together with Python 3.6.

### Installation
1. Install python 3.6
2. Install all dependencies
3. Clone/download this folder
4. Modify parameters at your taste (number of cells in each layer, hyperparameters...)
5. Run Main.py (```python Main.py```) to start the program without the GUI
6. Run also gui_v2.py (```python gui_v2.py```) to establish a connection with the GUI
7. Navigate to ```127.0.0.1:8050``` on your browser to open the GUI

You have to note that all the key press work with Selenium, so you can move around and minimise windows while the program is running.

On PyCharm it is possible to create a Multirun configuration that runs first the GUI and then the Main program, so it is easier.

Both programs work independently, so Main will work perfectly without the GUI.

To quit the program, you can press the "End" key, it will stop when the current generation has finished and after saving all the models in use.

## Details
In every iteration, while the dinosaur is not crashed, the program reads 4 inputs:
1. Distance until next object (X)
2. Y position of next object
3. Current speed, adjusting the "Speed offset", if any
4. Size

There is only output, ranging from 0 to 1:
1. output < 0.45: Press DOWN key
2. output > 0.55: Press UP key
3. default: pass

This version uses a genetic algorithm to evolve the network from random weights and biases to working parameters. Each generation has a number of genomes, and each of them is tested until it crashes. When this happens, the score is saved as the fitness of that genome.

When all the genomes in a generation are completed, the highest scores are selected and kept for the next generation. The rest of genomes are calculated by crossover and/or mutating them. The genomes that take part in these processes of crossover and mutation are chosen at random, but attending to the score each one has achieved. To reach this, a very simple probability distribution is calculated, so the highest score gets more probability of being chosen as a basis for next generations.

## Implementation
The implementation has been done entirely on Python, using Tensorflow and Selenium to interact with the JS of the game.

These are the files on the project:
* ```Dino.py```: it contains two classes: Game and Dino. The first is responsible for communicating with the browser. The latter gets the inputs of the game and sends them to the controller.
* ```Main.py```: it joins all the other scripts
* ```genome.py```: This script does everything: mutation, crossover, control of generations...
* ```gui_v2.py```: All the gui related functions
* ```keys.py```: A short script to stop the program by pressing End key
* ```perceptron_2l.py```: Makes the Neural Networks, gets dictionaries to use them, etc.
