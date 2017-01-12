# Guerilla

Project uses python 2.7

### Required Packages
- python-chess:
  - `sudo pip install python-chess`
- tensorflow:
  - https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
- pyYAML:
  - `pip install pyYAML`
- stockfish:
  - `sudo apt-get install stockfish`
- psutil (used when calling stockfish):
  - `sudo pip install psutil`
- pygame (for gui):
  - `sudo apt-get build-dep python-pygame`
  - `sudo apt-get install python-pygame`
- guppy (only for testing):
  - `pip install guppy`

### Optional Packages

- sunfish (chess engine):
  - `cd ../guerilla/other_engines/`
  - `git clone https://github.com/thomasahle/sunfish.git`
  - `touch sunfish/__init__.py`
  
### How to Run

1. Get fens:
  - `python -m guerilla.train.chess_game_parser`  
1. Get stockfish values:
  - `python -m guerilla.train.stockfish_eval`
1. Train by running:
  - `python -m guerilla.train.teacher <number hours (optional)> <number minutes (optional)> <number seconds (optional)>`
1. Run game by running:
  - `python -m guerilla.play.game` TODO add input options

