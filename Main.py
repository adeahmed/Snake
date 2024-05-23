import random
import pygame

SCREEN_WIDTH=800
SCREEN_HEIGHT=600

FRUIT_SIZE=10
SNAKE_SIZE=10

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    KEYDOWN,
    QUIT,
)


class Fruit():
    def __init__(self):
        self.loc=[random.randint(SNAKE_SIZE, SCREEN_WIDTH - SNAKE_SIZE),
        SNAKE_SIZE, SCREEN_HEIGHT - SNAKE_SIZE],
        self.color=((212,0,0))
    
        
        


class Snake():
    def __init__(self):
        self.loc=[random.randint(FRUIT_SIZE,SCREEN_WIDTH-FRUIT_SIZE),
        random.randint(FRUIT_SIZE,SCREEN_HEIGHT-FRUIT_SIZE)]
        self._moves=("UP","DOWN","RIGHT","LEFT")
        self.direction=random.choice(self._moves)   
        self.color=((0,212,0)) 

    def move(self,direction):
        if direction==self._moves[0]:
            self.loc[1]=(self.loc[1]-SNAKE_SIZE)%SCREEN_HEIGHT
        elif direction==self._moves[1]:
            self.loc[1]=(self.loc[1]+SNAKE_SIZE)%SCREEN_HEIGHT
        elif direction==self._moves[2]:
            self.loc[0]=(self.loc[0]+SNAKE_SIZE)%SCREEN_WIDTH
        elif direction==self._moves[3]:
            self.loc[0]=(self.loc[0]-SNAKE_SIZE)%SCREEN_WIDTH
        self.direction=direction



class Game():
    def __init__(self):
        self.running=True
        self.snake=Snake()
        self.fruit=Fruit()
        self.screen=pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
        self.clock=pygame.time.Clock()

    def main_loop(self):
        direction=None
        while self.running:
            direction=self.snake.direction
            for event in pygame.event.get():
                if event.type==KEYDOWN:
                    if event.key==K_DOWN:
                        direction="DOWN"
                    elif event.key==K_UP:
                        direction="UP"
                    elif event.key==K_LEFT:
                        direction="LEFT"
                    elif event.key==K_RIGHT:
                        direction="RIGHT"
            self.screen.fill((0,0,0))
            self.snake.move(direction)
            x,y=self.snake.loc
            snake_block=pygame.Surface((SNAKE_SIZE,SNAKE_SIZE))
            snake_block.fill(self.snake.color)
            self.screen.blit(snake_block,self.snake.loc)
            pygame.display.flip()
            self.clock.tick(60)
game=Game()
game.main_loop()
