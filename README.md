# Guerilla

Guerilla is a deep learning chess engine developed by Miguel and Stephane Aroca-Ouellette. More details on the project can be found at [here](https://unarresteddev.wordpress.com/2017/02/23/guerilla-a-chess-engine-part-1/) and [here](https://unarresteddev.wordpress.com/2017/03/08/guerilla-a-chess-engine-part-2/).
  
The project was developed in Python 2.7 on Linux systems (Ubuntu). All instructions below are for linux systems. If someone gets it running on a different operating system successfully, a pull request with the instructions would be appreciated. 

## To Play
### Install
- Install Tensorflow 1.x.x (https://www.tensorflow.org/install/)
- run `pip install --user Guerilla`
### Play
- Add Guerilla to your favorite UCI Compatible chess GUI (after the pip installation, the command for the GUI to run Guerilla is `Guerilla`). 
- Currenlty only supports time limit and depth limit (depth 4 ~= 1 minutes per move).
- Example using Chess Arena (http://www.playwitharena.com/):
  - Run Chess Arena
  - In the Menu Bar (top bar), select `Engines->Install New Engine...`
  - Go to `~/.local/bin/` (or `\usr\local\bin\` if installed without the `--user` option) and select `Guerilla`
  - A window asking UCI or Winboard should appear. Select UCI.
  - In the Menu Bar (top bar), select `Levels->Adjust`.
  - In the window, choose either `Fixed search depth` or `Time per move` and input desired value.
  - In the Menu Bar (top bar), select `File->New`.


## Improve Guerilla
## Dependencies
Packages and programs required for things to work.

### Required Packages to Play
- python-chess:
  - `pip install --user python-chess`
- tensorflow 1.0:
  - https://www.tensorflow.org/install/
- numpy
  - `sudo apt-get install python-numpy`
- ruamel.yaml <= 0.15:
  - `pip install --user ruamel.yaml`

### Required Packages to Train
- stockfish:
  - `sudo apt-get install stockfish`
- psutil (used when calling stockfish):
  - `pip install --user psutil`

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
  - Google LLC
