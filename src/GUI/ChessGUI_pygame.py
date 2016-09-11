#! /usr/bin/env python
"""
 Project: Python Chess
 File name: ChessGUI_pygame.py
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
from ScrollingTextBox import ScrollingTextBox

class ChessGUI_pygame:
    def __init__(self,graphicStyle=1):
        os.environ['SDL_VIDEO_CENTERED'] = '1' #should center pygame window on the screen
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((850,500))
        self.boardStart_x = 50
        self.boardStart_y = 50
        pygame.display.set_caption('Python Chess')

        self.textBox = ScrollingTextBox(self.screen,525,825,50,450)
        self.LoadImages(graphicStyle)
        #pygame.font.init() - should be already called by pygame.init()
        self.fontDefault = pygame.font.Font( None, 20 )

        self.ranks = ['8','7','6','5','4','3','2','1']
        self.files = ['a','b','c','d','e','f','g','h']

    def LoadImages(self,graphicStyle):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        if graphicStyle == 0:
            self.square_size = 50 #all images must be images 50 x 50 pixels
            self.white_square = pygame.image.load(os.path.join(dir_path,"images","white_square.png")).convert()
            self.brown_square = pygame.image.load(os.path.join(dir_path,"images","brown_square.png")).convert()
            self.cyan_square = pygame.image.load(os.path.join(dir_path,"images","cyan_square.png")).convert()
            #"convert()" is supposed to help pygame display the images faster.  It seems to mess up transparency - makes it all black!
            #And, for this chess program, the images don't need to change that fast.
            self.black_pawn = pygame.image.load(os.path.join(dir_path,"images","blackPawn.png")) 
            self.black_rook = pygame.image.load(os.path.join(dir_path,"images","blackRook.png"))
            self.black_knight = pygame.image.load(os.path.join(dir_path,"images","blackKnight.png"))
            self.black_bishop = pygame.image.load(os.path.join(dir_path,"images","blackBishop.png"))
            self.black_king = pygame.image.load(os.path.join(dir_path,"images","blackKing.png"))
            self.black_queen = pygame.image.load(os.path.join(dir_path,"images","blackQueen.png"))
            self.white_pawn = pygame.image.load(os.path.join(dir_path,dir_path,"images","whitePawn.png"))
            self.white_rook = pygame.image.load(os.path.join(dir_path,dir_path,"images","whiteRook.png"))
            self.white_knight = pygame.image.load(os.path.join(dir_path,dir_path,"images","whiteKnight.png"))
            self.white_bishop = pygame.image.load(os.path.join(dir_path,dir_path,"images","whiteBishop.png"))
            self.white_king = pygame.image.load(os.path.join(dir_path,dir_path,"images","whiteKing.png"))
            self.white_queen = pygame.image.load(os.path.join(dir_path,dir_path,"images","whiteQueen.png"))
        elif graphicStyle == 1:
            self.square_size = 50
            self.white_square = pygame.image.load(os.path.join(dir_path,"images","white_square.png")).convert()
            self.brown_square = pygame.image.load(os.path.join(dir_path,"images","brown_square.png")).convert()
            self.cyan_square = pygame.image.load(os.path.join(dir_path,"images","cyan_square.png")).convert()
            
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


    def print_msg(self,message):
        #prints a string to the area to the right of the board
        self.textBox.Add(message)
        self.textBox.Draw()
        
    def get_screen_coords(self, _rank, _file):
        #converts a (row,col) chessSquare into the pixel location of the upper-left corner of the square
        screenX = self.boardStart_x + _file*self.square_size
        screenY = self.boardStart_y + _rank*self.square_size
        return (screenX,screenY)

    def get_screen_coords_from_tile(self, tile):
        _file = self.files.index(tile[0])
        _rank = self.ranks.index(tile[1])
        return self.get_screen_coords(_rank, _file)
        
    def get_tile(self, x, y):
        #converts a screen pixel location (X,Y) into a chessSquare tuple (row,col)
        #x is horizontal, y is vertical
        #(x=0,y=0) is upper-left corner of the screen
        _rank = (y-self.boardStart_y) / self.square_size
        _file = (x-self.boardStart_x) / self.square_size
        if _rank < 0 or _file < 0 or _rank > 7 or _file > 7:
            return None
        return self.files[_file] + self.ranks[_rank]        
        
    def draw(self, board, highlight_tiles=[]):
        board_fen = board.fen().split()[0]
        self.screen.fill((0,0,0))
        self.textBox.Draw()
        boardSize = 8 

        #draw blank board
        current_square = 0
        for r in range(boardSize):
            for c in range(boardSize):
                (screenX,screenY) = self.get_screen_coords(r,c)
                if current_square:
                    self.screen.blit(self.brown_square,(screenX,screenY))
                    current_square = (current_square+1)%2
                else:
                    self.screen.blit(self.white_square,(screenX,screenY))
                    current_square = (current_square+1)%2

            current_square = (current_square+1)%2

        #draw row/column labels around the edge of the board
        color = (255,255,255)#white
        antialias = 1
        
        #top and bottom - display cols
        for c in range(boardSize):
            for r in [-1,boardSize]:
                (screenX,screenY) = self.get_screen_coords(r,c)
                screenX = screenX + self.square_size/2
                screenY = screenY + self.square_size/2
                notation = self.files[c]
                renderedLine = self.fontDefault.render(notation,antialias,color)
                self.screen.blit(renderedLine,(screenX,screenY))
        
        #left and right - display rows
        for r in range(boardSize):
            for c in [-1,boardSize]:
                (screenX,screenY) = self.get_screen_coords(r,c)
                screenX = screenX + self.square_size/2
                screenY = screenY + self.square_size/2
                notation = self.ranks[r]
                renderedLine = self.fontDefault.render(notation,antialias,color)
                self.screen.blit(renderedLine,(screenX,screenY))
                
        #highlight squares if specified
        for tile in highlight_tiles:
            (screenX,screenY) = self.get_screen_coords_from_tile(tile)
            self.screen.blit(self.cyan_square,(screenX,screenY))
        
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

            (screenX,screenY) = self.get_screen_coords(_rank,_file)
            if char == 'p':
                self.screen.blit(self.black_pawn,(screenX,screenY))
            elif char == 'r':
                self.screen.blit(self.black_rook,(screenX,screenY))
            elif char == 'n':
                self.screen.blit(self.black_knight,(screenX,screenY))
            elif char == 'b':
                self.screen.blit(self.black_bishop,(screenX,screenY))
            elif char == 'q':
                self.screen.blit(self.black_queen,(screenX,screenY))
            elif char == 'k':
                self.screen.blit(self.black_king,(screenX,screenY))
            elif char == 'P':
                self.screen.blit(self.white_pawn,(screenX,screenY))
            elif char == 'R':
                self.screen.blit(self.white_rook,(screenX,screenY))
            elif char == 'N':
                self.screen.blit(self.white_knight,(screenX,screenY))
            elif char == 'B':
                self.screen.blit(self.white_bishop,(screenX,screenY))
            elif char == 'Q':
                self.screen.blit(self.white_queen,(screenX,screenY))
            elif char == 'K':
                self.screen.blit(self.white_king,(screenX,screenY))
            _file += 1
            
        pygame.display.flip()
            
    def get_player_input(self, board, currentColor):
        #returns ((from_row,from_col),(to_row,to_col))
        from_tile = None
        to_tile = None
        while from_tile is None or to_tile is None:
            tile = None
            pygame.event.set_blocked(MOUSEMOTION)
            e = pygame.event.wait()
            if e.type is KEYDOWN:
                if e.key is K_ESCAPE:
                    fromSquareChosen = 0
                    fromTuple = []
            if e.type is MOUSEBUTTONDOWN:
                (mouseX,mouseY) = pygame.mouse.get_pos()
                tile = self.get_tile(mouseX,mouseY)
            if e.type is QUIT: #the "x" kill button
                pygame.quit()
                sys.exit(0)
                    
            if from_tile is None and to_tile is None:
                self.draw(board)
                if tile is not None:
                    if tile in [str(x)[:2] for x in board.legal_moves]:
                        from_tile = tile
                        tile = None
                    else:
                        print "No possible legal moves from that tile"
                        tile = None
                             
            elif from_tile is not None and to_tile is None:
                possible_to_tiles = [str(x)[2:] for x in board.legal_moves if from_tile == str(x)[:2]]
                self.draw(board, possible_to_tiles)
                if tile is not None:
                    if tile in possible_to_tiles:
                        to_tile = tile
                    else:
                        print "Illegal or invalid move"
                        from_tile = None
                        to_tile = None
                        tile = None

        return from_tile + to_tile