from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import logging
from shutil import copy2

logger = logging.getLogger('Dino')


#path variables
game_url = "chrome://dino"
chrome_driver_path = "./chromedriver"

#scripts
#create id for canvas for faster selection from DOM. Also keep Dino position fixed
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas';" \
              "setInterval(function (){Runner.instance_.tRex.xPos = 0}, 500);"


'''
 * Game class: Selenium interfacing between the python and browser* Game
* __init__():  Launch the broswer window using the attributes in chrome_options
* get_crashed() : return true if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
* get_playing(): true if game in progress, false if crashed
* restart() : sends a signal to browser-javascript to restart the game
* press_up()/press_down(): sends a single to press up/down get to the browser
* get_score(): gets current game score from javascript variables.
* pause(): pause the game
* resume(): resume a paused game if not crashed
* end(): close the browser and end the game
'''


class Game:
    def __init__(self,mode, nGenerations, nGenes, custom_config=True):
        self.mode = mode
        self.nGenerations = nGenerations
        self.nGenes = nGenes
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(executable_path=chrome_driver_path, chrome_options=chrome_options)
        self._driver.set_window_position(x=-10, y=0)
        self._driver.get('chrome://dino')
        self._driver.execute_script(init_script)
        self.press_up()
        self.return_nor()

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")
    def get_playing(self):
        return not self._driver.execute_script("return Runner.instance_.crashed")
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    def return_nor(self):
        pass
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array) # the javascript object is of type array with score in the format [1,0,0] which is 100.
        return int(score)
    def get_speed(self):
        try:
            # Some obstacles have an offset, so that has to be added.
            speed_offset = self._driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].speedOffset")
        except:
            speed_offset = 0
        speed_intrinsic = self._driver.execute_script("return Runner.instance_.currentSpeed")

        speed_total = speed_offset + speed_intrinsic

        return speed_total

    def get_position(self):
        try:
            pos_obs = self._driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].xPos")
        except:
            pos_obs = 0
        dino_width = self._driver.execute_script("return Runner.instance_.tRex.config.WIDTH_DUCK")
        #If there is no object, return a high number.
        if pos_obs != 0:
            return pos_obs - dino_width
        else:
            return 600

    def get_ypos(self):
        pixels = 140
        try:
            height = self._driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].typeConfig.height")
            ypos = self._driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].yPos")

            return pixels - (height+ypos)
        except:
            return pixels

    def get_size(self):
        try:
            return self._driver.execute_script("return (Runner.instance_.horizon.obstacles)[0].width")
        except:
            return 0
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")
    def end(self):
        self._driver.close()

class Dino:
    def __init__(self, game):
        self.gameOutputNumber = 0
        self.game = game
        self.inputs = []
        self.gameOutputString = 'NORM'
        self.gamePreviousString = 'NORM'

    def get_inputs(self):
        size = self.game.get_size()
        speed = self.game.get_speed()
        position = self.game.get_position()
        y_pos = self.game.get_ypos()

        return [[size, speed, position, y_pos]]

    def play(self, gene, i_gen, i_generation, folder):
        while self.game.get_playing():
            self.inputs = self.get_inputs()
            self.gameOutputNumber = gene.activate(self.inputs)[0][0]
            self.gameOutputString = self.game_output_string()

            self.game_key()

            #Read log
            log_lines = ''
            with open('./logs/ui.log', 'r') as f_read:
                lineas = f_read.readlines()
            #Save last 5 lines
            for line_number in range(1, 6):
                log_lines = log_lines + '<br>'+lineas[-line_number].replace('\n', '')

            #Write the data to a file, only once for performance reasons.
            data = []
            data.append(str(self.game.get_score())+'\n')
            data.append(str(self.inputs[0][0]) + '\n')
            data.append(str(self.inputs[0][1]) + '\n')
            data.append(str(self.inputs[0][2]) + '\n')
            data.append(str(i_gen) + '\n')
            data.append(str(i_generation) + '\n')
            data.append(str(self.gameOutputNumber) + '\n')
            data.append(str(self.game.mode) + '\n')
            data.append(str(self.gameOutputString) + '\n')
            data.append(str(folder) + '\n')
            data.append(str(self.game.nGenes) + '\n')
            data.append(str(self.game.nGenerations) + '\n')
            data.append(str(self.inputs[0][3]) + '\n')
            data.append(log_lines + '\n')

            with open("./tmp/working_data.txt", "w") as F:
                F.write(''.join(data))


            copy2("./tmp/working_data.txt", "./tmp/dash_data.txt") #Copy the file for the gui to read

            self.gamePreviousString = self.gameOutputString

        return self.game.get_score()

    def game_output_string(self): #This is just to have a good way to recognise what the output means.
        if self.gameOutputNumber < 0.45:
            return 'DOWN'
        elif self.gameOutputNumber > 0.55:
            return 'JUMP'
        else:
            return 'NORM'

    def game_key(self): #Actual action related to the output
        if self.gameOutputString == 'DOWN':
            self.game.press_down()
        elif self.gameOutputString == 'JUMP':
            self.game.press_up()
        else:
            if self.gamePreviousString == 'JUMP':
                self.game.press_down()
            self.game.return_nor()