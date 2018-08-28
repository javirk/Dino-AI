import keys
from genome import Genome
from Dino import Game

import logging
from pynput import keyboard

numGenes = 12
mutationProb = 0.1
selection = 4
keys.break_program = False
folder = '' #If a folder is specified, no networks will be generated, it will try and take them from there.
mode = 'TRAINING'
nGenerations = 90

LOG_FILENAME = './logs/ui.log'

logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO, format='%(asctime)s, %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')

logger = logging.getLogger('main')

logger.info(f'Programa lanzado. Datos: Número de genes: {numGenes}, '
            f'Probabilidad de mutación: {mutationProb}, Genes seleccionados: {selection}')
logger.info(f'Se ha elegido el modo {mode}')
logger.info('Loading genome...')
genome = Genome(numGenes, mutationProb, selection, folder,  nGenerations)
logger.info('The genome was successfully loaded')

game = Game(mode, nGenerations, numGenes)

if mode == 'TRAINING':
    logger.info(f'Training will last {nGenerations} generations.')
    for i in range(1,  nGenerations):
        with keyboard.Listener(on_press=keys.on_press) as listener:
            if not keys.break_program: #End key to stop a running program by the end of the generation.
                genome.execute_generation(game)

                if (i % 5 == 0 and i != 0) or i == nGenerations:
                    genome.save_all()

                genome.kill_and_reproduce()
            else:
                genome.save_all()
                break

        listener.join()

else:
    with keyboard.Listener(on_press=keys.on_press) as listener:
        if not keys.break_program:
            genome.execute_generation(game)
    listener.join()