#! /usr/bin/env python
"""
Modified from:
    Copyright (C) 2009 Steve Osborne, srosborne (at) gmail.com
    http://yakinikuman.wordpress.com/

Project: Python Chess
File name: ChessGUI.py
Description:  Uses pygame (http://www.pygame.org/) to draw the
chess board, as well as get user input through mouse clicks.
The chess tile graphics were taken from Wikimedia Commons, 
http://commons.wikimedia.org/wiki/File:Chess_tile_pd.png

Copyright (C) 2009 Steve Osborne, srosborne (at) gmail.com
http://yakinikuman.wordpress.com/


 """
 
import pygame
import chess
import os
import sys
from pygame.locals import *
from guerilla.play.gui.scrolling_text_box import ScrollingTextBox
from pkg_resources import resource_filename

class ChessGUI:

    size = 8

    def __init__(self, view=False):
        """
            Class Constructor. Sets params and initializes basics.
        """
        os.environ['SDL_VIDEO_CENTERED'] = '1' #should center pygame window on the screen
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((850, 600))
        self.board_offset_x = 50
        self.board_offset_y = 50
        pygame.display.set_caption('Python Chess')

        self.text_box = ScrollingTextBox(self.screen, 525, 825, 50, 450)
        self.load_images()
        # pygame.font.init() - should be already called by pygame.init()
        self.default_font = pygame.font.Font(None, 20)
        self.qrbk_font = pygame.font.Font(None, 30)

        self.ranks = ['8','7','6','5','4','3','2','1']
        self.files = ['a','b','c','d','e','f','g','h']
        self.end_of_game = False
        self.view = view

    def load_images(self):
        """
            Load images used to represent chess pieces.
        """
        dir_path = resource_filename('guerilla.play.gui', '.')
        self.square_size = 50
        self.white_square = pygame.image.load(os.path.join(dir_path,"images","white_square.png")).convert()
        self.brown_square = pygame.image.load(os.path.join(dir_path,"images","brown_square.png")).convert()
        self.cyan_square = pygame.image.load(os.path.join(dir_path,"images","cyan_square.png")).convert()

        self.new_game_button = pygame.image.load(os.path.join(dir_path,"images","new_game_button.png")).convert()
        self.next_ = pygame.image.load(os.path.join(dir_path,"images","next.png")).convert()
        self.prev_ = pygame.image.load(os.path.join(dir_path,"images","prev.png")).convert()
        
        self.black_pawn = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_pd.png")).convert()
        self.black_pawn = pygame.transform.scale(self.black_pawn, (self.square_size,self.square_size))
        self.black_rook = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_rd.png")).convert()
        self.black_rook = pygame.transform.scale(self.black_rook, (self.square_size,self.square_size))
        self.black_knight = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_nd.png")).convert()
        self.black_knight = pygame.transform.scale(self.black_knight, (self.square_size,self.square_size))
        self.black_bishop = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_bd.png")).convert()
        self.black_bishop = pygame.transform.scale(self.black_bishop, (self.square_size,self.square_size))
        self.black_king = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_kd.png")).convert()
        self.black_king = pygame.transform.scale(self.black_king, (self.square_size,self.square_size))
        self.black_queen = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_qd.png")).convert()
        self.black_queen = pygame.transform.scale(self.black_queen, (self.square_size,self.square_size))

        self.white_pawn = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_pl.png")).convert()
        self.white_pawn = pygame.transform.scale(self.white_pawn, (self.square_size,self.square_size))
        self.white_rook = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_rl.png")).convert()
        self.white_rook = pygame.transform.scale(self.white_rook, (self.square_size,self.square_size))
        self.white_knight = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_nl.png")).convert()
        self.white_knight = pygame.transform.scale(self.white_knight, (self.square_size,self.square_size))
        self.white_bishop = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_bl.png")).convert()
        self.white_bishop = pygame.transform.scale(self.white_bishop, (self.square_size,self.square_size))
        self.white_king = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_kl.png")).convert()
        self.white_king = pygame.transform.scale(self.white_king, (self.square_size,self.square_size))
        self.white_queen = pygame.image.load(os.path.join(dir_path,"images","Chess_tile_ql.png")).convert()
        self.white_queen = pygame.transform.scale(self.white_queen, (self.square_size,self.square_size))


    def print_msg(self, message):
        """
            Prints a message to the text box
            Inputs:
                messgage[string]:
                    message to print
        """
        # print message
        self.text_box.add(message)
        self.text_box.draw()
        
    def get_screen_coords(self, _rank, _file):
        """ 
            Converts a rank and file index to the location of the pixel in the upper-left corner of the tile 
            Inputs:
                rank[int]:
                    rank index
                file[int]:
                    file index
            Outputs:
                pixel_indices[tuple size 2]:
                    (x, y) positions of pixed in upper-left corner of tile
        """
        screen_x = self.board_offset_x + _file*self.square_size
        screen_y = self.board_offset_y + _rank*self.square_size
        return (screen_x, screen_y)

    def get_screen_coords_from_tile(self, tile):
        """ 
            Converts a (rank, file) to the location of the pixel in the upper-left corner of the tile 
            Inputs:
                tile[tuple of size 2]:
                    (rank, file) indicies
            Outputs:
                pixel_indices[tuple size 2]:
                    (x, y) positions of pixed in upper-left corner of tile
        """
        _file = self.files.index(tile[0])
        _rank = self.ranks.index(tile[1])
        return self.get_screen_coords(_rank, _file)
        
    def get_tile(self, x, y):
        """
            Converts a screen pixel location (X,Y) into a chessSquare tuple (row,col)
            Note: (0, 0) is upper-left corner of the screen
            Inputs:
                x[int]:
                    x position of pixel
                y[int]:
                    y position of pixel
            Outputs:
                tile[string]:
                    algebraic notation of tile (e.g. 'a3')
        """
        _rank = (y-self.board_offset_y) / self.square_size
        _file = (x-self.board_offset_x) / self.square_size
        if _rank < 0 or _file < 0 or _rank > 7 or _file > 7:
            return None
        return self.files[_file] + self.ranks[_rank]        
        
    def draw(self, board, highlight_tiles=[], qrbk=False):
        """ 
            Draw board
            Inputs:
                board[chess.Board]:
                    board to draw
        """
        board_fen = board.fen().split()[0]
        self.screen.fill((0,0,0))
        self.text_box.draw()

        # Draw blank board
        current_square = 0
        for r in range(ChessGUI.size):
            for f in range(ChessGUI.size):
                (screen_x, screen_y) = self.get_screen_coords(r, f)
                if current_square:
                    self.screen.blit(self.brown_square, (screen_x, screen_y))
                    current_square = (current_square + 1) % 2
                else:
                    self.screen.blit(self.white_square, (screen_x, screen_y))
                    current_square = (current_square + 1) % 2

            current_square = (current_square + 1) % 2

        # Draw rank/file labels around the edge of the board
        color = (255,255,255) # White
        antialias = 1
        
        # Top and bottom - display cols
        for f in range(ChessGUI.size):
            for r in [-1,ChessGUI.size]:
                (screen_x, screen_y) = self.get_screen_coords(r, f)
                screen_x = screen_x + self.square_size/2
                screen_y = screen_y + self.square_size/2
                notation = self.files[f]
                renderedLine = self.default_font.render(notation, antialias, color)
                self.screen.blit(renderedLine, (screen_x, screen_y))
        
        # left and right - display rows
        for r in range(ChessGUI.size):
            for f in [-1, ChessGUI.size]:
                (screen_x,screen_y) = self.get_screen_coords(r, f)
                screen_x = screen_x + self.square_size/2
                screen_y = screen_y + self.square_size/2
                notation = self.ranks[r]
                renderedLine = self.default_font.render(notation, antialias, color)
                self.screen.blit(renderedLine, (screen_x, screen_y))
                
        # highlight squares if specified
        for tile in highlight_tiles:
            (screen_x, screen_y) = self.get_screen_coords_from_tile(tile)
            self.screen.blit(self.cyan_square, (screen_x, screen_y))
        
        #draw pieces
        _rank = 0
        _file = 0
        for char in board_fen:
            if char == '/':
                _rank += 1
                _file = 0
                continue
            if char.isdigit():
                _file += int(char)
                continue

            (screen_x, screen_y) = self.get_screen_coords(_rank, _file)
            if char == 'p':
                self.screen.blit(self.black_pawn, (screen_x, screen_y))
            elif char == 'r':
                self.screen.blit(self.black_rook, (screen_x, screen_y))
            elif char == 'n':
                self.screen.blit(self.black_knight, (screen_x, screen_y))
            elif char == 'b':
                self.screen.blit(self.black_bishop, (screen_x, screen_y))
            elif char == 'q':
                self.screen.blit(self.black_queen, (screen_x, screen_y))
            elif char == 'k':
                self.screen.blit(self.black_king, (screen_x, screen_y))
            elif char == 'P':
                self.screen.blit(self.white_pawn, (screen_x, screen_y))
            elif char == 'R':
                self.screen.blit(self.white_rook, (screen_x, screen_y))
            elif char == 'N':
                self.screen.blit(self.white_knight, (screen_x, screen_y))
            elif char == 'B':
                self.screen.blit(self.white_bishop, (screen_x, screen_y))
            elif char == 'Q':
                self.screen.blit(self.white_queen, (screen_x, screen_y))
            elif char == 'K':
                self.screen.blit(self.white_king, (screen_x, screen_y))
            _file += 1
        
        if qrbk:    
            for tile in highlight_tiles:
                (screen_x, screen_y) = self.get_screen_coords_from_tile(tile)
                screen_x += self.square_size/2
                screen_y += self.square_size/2
                off = 25
                x_off = 5
                renderedLine = self.qrbk_font.render('q', antialias, (150,150,150))
                self.screen.blit(renderedLine, (screen_x - off/2 - x_off, screen_y - off))
                renderedLine = self.qrbk_font.render('r', antialias, (150,150,150))
                self.screen.blit(renderedLine, (screen_x + x_off, screen_y - off))
                renderedLine = self.qrbk_font.render('b', antialias, (150,150,150))
                self.screen.blit(renderedLine, (screen_x - off/2 - x_off, screen_y))
                renderedLine = self.qrbk_font.render('k', antialias, (150,150,150))
                self.screen.blit(renderedLine, (screen_x + x_off, screen_y))


        # Draw Next Game button
        if self.end_of_game:
            self.screen.blit(self.new_game_button, (300,525))
        
        # Draw view button
        if self.view:
            self.screen.blit(self.prev_, (233,525))
            self.screen.blit(self.next_, (517,525))

        pygame.display.flip()
            
    def get_player_input(self, board):
        """ 
            Returns player's move inputted to guy
            Inputs:
                board[chess.Board]:
                    current state of chess board
            Outputs:
                move[string]:
                    algebraic notation of move (e.g. 'b1c3')
        """
        from_tile = None
        to_tile = None
        while from_tile is None or to_tile is None:
            tile = None
            pygame.event.set_blocked(MOUSEMOTION)
            # Wait for input
            e = pygame.event.wait()
            if e.type is KEYDOWN:
                if e.key is K_ESCAPE:
                    fromSquareChosen = 0
                    fromTuple = []
            # On click, register tile clicked, continue
            if e.type is MOUSEBUTTONDOWN:
                (mouseX,mouseY) = pygame.mouse.get_pos()
                tile = self.get_tile(mouseX,mouseY)
            if e.type is QUIT: #the "x" kill button
                pygame.quit()
                sys.exit(0)
                   
            # Set from tile 
            if from_tile is None and to_tile is None:
                self.draw(board)
                if tile is not None:
                    if tile in [str(x)[:2] for x in board.legal_moves]:
                        from_tile = tile
                        tile = None
                    else:
                        print "No possible legal moves from that tile"
                        tile = None
                
            # Set to tile             
            elif from_tile is not None and to_tile is None:
                possible_to_tiles = [str(x)[2:] for x in board.legal_moves if from_tile == str(x)[:2]]
                self.draw(board, possible_to_tiles)
                if tile is not None:
                    promoted_type = ''
                    valid_to_tile = False
                    for possible_to_tile in possible_to_tiles:
                        if tile == possible_to_tile[:2]:
                            if len(possible_to_tile) == 3:
                                self.print_msg("Pawn Promotion:")
                                self.print_msg("        - q for Queen")
                                self.print_msg("        - r for Rook")
                                self.print_msg("        - b for Bishop")
                                self.print_msg("        - n for Knight")
                                self.print_msg("        - c to cancel")
                                self.draw(board, [tile], qrbk=True)
                                promoted_type = self.wait_for_promotion_input()
                            if promoted_type == 'c':
                                valid_to_tile = False
                            else:
                                valid_to_tile = True
                            to_tile = tile + promoted_type
                            break
                    if not valid_to_tile:
                        print "Illegal or invalid move"
                        from_tile = None
                        to_tile = None
                        tile = None

        return from_tile + to_tile

    def wait_for_endgame_input(self):
        pygame.event.set_blocked(MOUSEMOTION)
            # Wait for input
        while True:
            e = pygame.event.wait()
            # On click, register tile clicked, continue
            if e.type is MOUSEBUTTONDOWN and self.end_of_game:
                (mouse_x, mouse_y) = pygame.mouse.get_pos()
                if mouse_x > 300 and mouse_x < 500 and mouse_y > 525 and mouse_y < 575:
                    self.end_of_game = False
                    break
            elif e.type is QUIT: #the "x" kill button
                pygame.quit()
                sys.exit(0)

    def wait_for_view_input(self):
        pygame.event.set_blocked(MOUSEMOTION)
            # Wait for input
        while True:
            e = pygame.event.wait()
            # On click, register tile clicked, continue
            if e.type is MOUSEBUTTONDOWN:
                (mouse_x, mouse_y) = pygame.mouse.get_pos()
                if mouse_x > 233 and mouse_x < 283 and mouse_y > 525 and mouse_y < 575:
                    return False
                if mouse_x > 517 and mouse_x < 567 and mouse_y > 525 and mouse_y < 575:
                    return True
            if e.type is KEYDOWN:
                if e.key == K_LEFT:
                    return False
                elif e.key == K_RIGHT:
                    return True
            elif e.type is QUIT: #the "x" kill button
                pygame.quit()
                sys.exit(0)

    def wait_for_promotion_input(self):
        pygame.event.set_blocked(MOUSEMOTION)
        print 'start'
            # Wait for input
        while True:
            e = pygame.event.wait()
            if e.type is KEYDOWN:
                return chr(e.key) if chr(e.key) in ['q', 'r', 'b', 'n'] else 'c'
            elif e.type is QUIT: #the "x" kill button
                pygame.quit()
                sys.exit(0)



