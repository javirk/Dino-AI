from pynput import keyboard
import logging

logger = logging.getLogger('keys')

def on_press(key):
    global break_program
    if key == keyboard.Key.end:
        print('End pressed.')
        logger.info('End pressed. The program will stop by the end of current generation')
        break_program = True
        return False
