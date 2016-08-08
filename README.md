# guerilla
Project uses python 2.7

required packages:
    - tensorflow:
        * https://www.tensorflow.org/versions/r0.10/get_started/os_setup.html
    - stockfish
        * sudo apt-get install stockfish  
    - psutil (used when calling stockfish)
        * sudo pip install psutil

To run:
1. Get fens by running: python chess_game_parser.py <num_games_to_look_at> <maximum_num_of_fens>
2. Get stockfish values by running: stockfish_eval.py

