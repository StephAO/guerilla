# Guerilla

Project uses python 2.7

### Required Packages

- tensorflow:
  - https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
- stockfish:
  - `sudo apt-get install stockfish`
- psutil (used when calling stockfish):
  - `sudo pip install psutil`
- pygame (for gui):
  - `sudo apt-get build-dep python-pygame`
  - `sudo apt-get install python-pygame`

### How to Run

1. Get fens:
  - `python chess_game_parser.py <num_games_to_look_at> <maximum_num_of_fens>`  
1. Get stockfish values:
  - `python stockfish_eval.py`
1. Train by running:
  - `python teacher.py`
1. Run game by running:
  - `python game.py` TODO add input options

