# Guerilla

Guerilla is a deep learning chess engine developed by Miguel and Stephane Aroca-Ouellette. More details on the project can be found at [here](https://unarresteddev.wordpress.com/2017/02/23/guerilla-a-chess-engine-part-1/).
  
The project was developed in Python 2.7 on Linux systems (Ubuntu). All instructions below are for linux systems. If someone gets it running on a different operating system successfully, a pull request with the instructions would be appreciated.  

## Dependencies
Packages and programs required for things to work.

### Required Packages to Play
- python-chess:
  - `sudo pip install python-chess`
- tensorflow 1.0:
  - https://www.tensorflow.org/install/
- numpy
  - `sudo apt-get install python-numpy`
- pygame (for gui):
  - `sudo apt-get build-dep python-pygame`
    - If this does not work check the 2nd answer here: https://askubuntu.com/questions/312767/installing-pygame-with-pip
  - `sudo apt-get install python-pygame`
- ruamel.yaml:
  - `pip install ruamel.yaml`

### Required Packages to Train
- stockfish:
  - `sudo apt-get install stockfish`
- psutil (used when calling stockfish):
  - `sudo pip install psutil`

### Required Packages for Testing
- guppy:
  - `pip install guppy`

### Optional other chess engines
- stockfish:
  - `sudo apt-get install stockfish` (if installed to train, no need to do this again)
- sunfish:
  - `cd ../guerilla/other_engines/`
  - `git clone https://github.com/thomasahle/sunfish.git`
  - `touch sunfish/__init__.py`

## Usage
### How to Play
1. Go to guerilla directory (outer guerilla directory)
  - `cd /path/to/guerilla`
2. Run game by running:
  - `python -m guerilla.play.game`
3. Select players:
  - Typing 'd' then \<ENTER>, will choose defaults, which is a game against the current best version of the Guerilla engine. You will start as white, Guerilla as white. After each game, opponents will swap colors
  - Typing 'c' then \<ENTER>, will let you choose the players. Player 1 (first inputted) will start as white. The terminal will prompt you to input the following:   
    - Player name (i.e. Donkey Kong)
    - Player type ['human', 'guerilla', 'Stockfish' (if installed), 'Sunfish' (if installed)]
    - If guerilla selected, weight file to load from. 'd' will default to the best current version of guerilla

### How to Train
1. Go to guerilla directory (outer guerilla directory)
  - `cd /path/to/guerilla`
2. Get fens:
  - `python -m guerilla.train.chess_game_parser`  
3. Get stockfish values:
  - `python -m guerilla.train.stockfish_eval`
4. Train by running:
  - `python -m guerilla.train.teacher <number hours (optional)> <number minutes (optional)> <number seconds (optional)>`

If you need further information on development, feel free to contact [us](#authors).

### How to Test
1. Go to guerilla directory (outer guerilla directory)
  - `cd /path/to/guerilla`
2. Choose test to run:
  - data_test: Tests data transformations.
  - play_test: Tests all functionality required to play the game.
  - train_test: Tests all the functionality required to train guerilla.
  - all_test: Tests all of the above
3. Run test:
  - `python -m tests.<test_chosen>`

## License
Licensed under MIT license. Refer to [LICENSE.md](LICENSE.md).  

## Authors
  - Miguel Aroca-Ouellette: mig_ao[at]live[dot]com
  - Stephane Aroca-Ouellette: stephanearocaouellette[at]gmail[dot].com
