#! /usr/bin/env python
"""
Adapted from:
	 Copyright (C) 2009 Steve Osborne, srosborne (at) gmail.com
	 http://yakinikuman.wordpress.com/

 Project: Python Chess
 File name: ScrollingTextBox.py
 Description:  Uses pygame to draw a scrolling text box, which is 
	incorporated in the ChessGUI_pygame class.

 """
 
import math
import pygame
from pygame.locals import *

class ScrollingTextBox:

	def __init__(self, screen, xmin, xmax, ymin, ymax):
		""" 
			Class constructor
			Inputs:
				screen[pygame.display]:
					screen to display to
				xmin[int]:
					x lower bound of window
				xmax[int]:
					x upper bound of window
				ymin[int]:
					y lower bound of window
				ymax[int]:
					y upper bound of window
		"""
		self.screen = screen
		pygame.font.init()
		self.default_font = pygame.font.Font( None, 20 )
		
		self.xmin = xmin
		self.xmax = xmax
		self.x_pixel_len = xmax - xmin
		self.ymin = ymin
		self.ymax = ymax
		self.y_pixel_len = ymax - ymin
		
		(width,height) = self.default_font.size('A')
		self.line_height = height
		self.max_num_lines = math.floor(self.y_pixel_len / self.line_height)

		self.lines = []
	
	def add_line(self, line):
		""" 
			Add line to textbox. Remove one first if queue is full.
			Inputs:
				line[string]:
					string to add
		"""
		if len(self.lines) > self.max_num_lines:
			self.lines.pop(0)
		self.lines.append(line)
	
	def add(self, message):
		""" 
			Add text to text box. Will add reformat message into correct line lenghts
			Inputs:
				message[string]:
					message to add
		"""
		(width, height) = self.default_font.size(message)
		remainder = ""
		if width > self.x_pixel_len:
			while width > self.x_pixel_len:
				remainder = message[-1] + remainder
				message = message[0:-1]
				(width, height) = self.default_font.size(message)
		
		if len(remainder) > 0:
			if message[-1].isalnum() and remainder[0].isalnum():
				remainder = message[-1] + remainder
				message = message[0:-1] + '-'
				if message[-2] == ' ':
					message = message[0:-1]
			
		self.add_line(message)

		if len(remainder) > 0:
			#remove leading spaces
			while remainder[0] == ' ':
				remainder = remainder[1:len(remainder)]
			self.add(remainder)

		
	def draw(self):
		"""
			Draw text
		"""
		xpos = self.xmin
		ypos = self.ymin
		color = (255, 255, 255) # White
		antialias = 1
		for line in self.lines:
			renderedLine = self.default_font.render(line, antialias, color)
			self.screen.blit(renderedLine, (xpos, ypos))
			ypos = ypos + self.line_height
		

		

if __name__ == "__main__":
#testing stuff (if this file is run directly)
	pygame.init()
	pygame.display.init()
	screen = pygame.display.set_mode((800,500))
	screen.fill( (0,0,0) )
	xmin = 400
	xmax = 750
	ymin = 100
	ymax = 400
	textbox = ScrollingTextBox(screen,xmin,xmax,ymin,ymax)
	
	textbox.add("hello")
	textbox.add("When requesting fullscreen display modes, sometimes an exact match for the requested resolution cannot be made. In these situations pygame will select the closest compatable match. The returned surface will still always match the requested resolution.")
	textbox.add("Soup for me!")
	textbox.add("Another data structure for which a list works well in practice, as long as the structure is reasonably small, is an LRU (least-recently-used) container. The following statements moves an object to the end of the list:")
	textbox.add("Set the current alpha value fo r the Surface. When blitting this Surface onto a destination, the pixels will be drawn slightly transparent. The alpha value is an integer from 0 to 255, 0 is fully transparent and 255 is fully opaque. If None is passed for the alpha value, then the Surface alpha will be disabled.")
	textbox.add("All pygame functions will automatically lock and unlock the Surface data as needed. If a section of code is going to make calls that will repeatedly lock and unlock the Surface many times, it can be helpful to wrap the block inside a lock and unlock pair.")
	textbox.draw()
	pygame.display.flip()
	
	while 1:
		for e in pygame.event.get():
			if e.type is KEYDOWN:
				pygame.quit()
				exit()
			if e.type is MOUSEBUTTONDOWN:
				(mouseX,mouseY) = pygame.mouse.get_pos()
				textbox.Add("Mouse clicked at ("+str(mouseX)+","+str(mouseY)+")")
				textbox.Draw()
				pygame.display.flip()


