import random
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

torch.set_default_tensor_type("torch.cuda.FloatTensor")

SCREEN_WIDTH = 250
SCREEN_HEIGHT = 250#250

FRUIT_SIZE = 10
SNAKE_SIZE = 10

VIEW_SIZE = 10

from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    KEYDOWN,
    QUIT,
)

class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(VIEW_SIZE**2 + 9, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 4)
        self.gamma = 0.99
        self.epsilon = 0.007
        self.epsilon_decay = 0.999975
        self.epsilon_min = 0.007
        self.lr = 0.0002
        self.memory = []
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.action_space = 4
        self.batch_size = 64

    def preprocess_state(self, state):
        return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def select_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            state = self.preprocess_state(state)
            with torch.no_grad():
                return self(state).argmax().item()

    def update_model(self):
        if len(self.memory) < self.batch_size:
            return
        if len(self.memory) > 10000:
            self.memory = self.memory[len(self.memory) - 10000:]
        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state = zip(*batch)
        state = torch.cat(state)
        next_state = torch.cat(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)

        q_values = self(state).gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_values = self(next_state).max(1)[0]
        expected_q_values = reward + self.gamma * next_q_values

        loss = self.criterion(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc4(x)
        return x

class Fruit:
    def __init__(self):
        self.loc = (random.randint(0, (SCREEN_WIDTH // FRUIT_SIZE) - 1) * FRUIT_SIZE,
                    random.randint(0, (SCREEN_HEIGHT // FRUIT_SIZE) - 1) * FRUIT_SIZE)
        self.color = (212, 0, 0)

class Snake:
    def __init__(self, game):
        self.loc = [[random.randint(0, (SCREEN_WIDTH // SNAKE_SIZE) - 1) * SNAKE_SIZE,
                     random.randint(0, (SCREEN_HEIGHT // SNAKE_SIZE) - 1) * SNAKE_SIZE]]
        self._moves = {"UP": (0, -SNAKE_SIZE), "DOWN": (0, SNAKE_SIZE), "RIGHT": (SNAKE_SIZE, 0), "LEFT": (-SNAKE_SIZE, 0)}
        self.direction = random.choice(list(self._moves.keys()))
        self.color = (0, 212, 0)
        self.is_growing = 0
        self._game = game
        self.prev_loc = [-1, -1]

    def move(self, direction):
        self.prev_loc = self.loc[0]
        if direction == "UP" and self.direction == "DOWN":
            direction = self.direction
        elif direction == "DOWN" and self.direction == "UP":
            direction = self.direction
        elif direction == "LEFT" and self.direction == "RIGHT":
            direction = self.direction
        elif direction == "RIGHT" and self.direction == "LEFT":
            direction = self.direction

        new_head = [self.loc[0][0] + self._moves[direction][0], self.loc[0][1] + self._moves[direction][1]]
        new_head[0] = new_head[0] #% SCREEN_WIDTH
        new_head[1] = new_head[1] #% SCREEN_HEIGHT
        
        # Check if the new head position collides with the walls
        if new_head[0] < 0 or new_head[0] >= SCREEN_WIDTH or new_head[1] < 0 or new_head[1] >= SCREEN_HEIGHT:
            self._game.running = False
            return -100
        
        if new_head in self.loc:
            self._game.running = False
            return -100

        self.loc = [new_head] + self.loc
        if self.is_growing > 0:
            self.is_growing -= 1
        else:
            self.loc.pop()
        self.direction = direction
        return -1

    def eat(self):
        self.is_growing += 1

class Game:
    def __init__(self):
        self.running = True
        self.snake = Snake(self)
        self.fruit = Fruit()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.is_ai = False
        self.agent = Agent()
        self.model_path = "snake_ai_model.pt"  # Path to save and load the model
        self.game_counter = 0
        self.score = 0
        self.current_score = 0
        self.highest = self.score

        # Load the model if exists
        try:
            checkpoint = torch.load(self.model_path)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully!")
        except FileNotFoundError:
            print("No saved model found. Training from scratch...")

    def get_game_state(self):
        head_x, head_y = self.snake.loc[0]
        direction = self.snake.direction
        future_head = [head_x + self.snake._moves[direction][0], head_y + self.snake._moves[direction][1]]

        grid = np.zeros((VIEW_SIZE, VIEW_SIZE))

        for i in range(-VIEW_SIZE//2, VIEW_SIZE//2):
            for j in range(-VIEW_SIZE//2, VIEW_SIZE//2):
                x = head_x + i*SNAKE_SIZE
                y = head_y + j*SNAKE_SIZE
                # If the position is out of bounds, mark it as a wall
                if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
                    grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = -1  # -1 represents a wall
                # If the position is the snake's body, mark it as a part of the snake
                elif [x, y] in self.snake.loc:
                    if [x, y] == self.snake.loc[0]:
                        grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 4  # 4 represents the head of the snake
                    elif [x, y] == self.snake.loc[-1]:
                        grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 5  # 5 represents the tail of the snake
                    else:
                        grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 1  # 1 represents a part of the snake
                # If the position is the fruit, mark it as the fruit
                elif (x, y) == self.fruit.loc:
                    grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 2  # 2 represents the fruit
                # If the position is the future head, mark it as the future head
                elif [x, y] == future_head:
                    grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 3  # 3 represents the future head
                else:
                    grid[i + VIEW_SIZE//2][j + VIEW_SIZE//2] = 0  # 0 represents an empty space

        view_size_encoding = np.array([SCREEN_HEIGHT//SNAKE_SIZE, SCREEN_WIDTH//SNAKE_SIZE])
        snake_location_encoding = np.array([self.snake.loc[0][0]//SNAKE_SIZE, self.snake.loc[0][1]//SNAKE_SIZE])
        snake_prev_location_encoding = np.array([self.snake.prev_loc[0]//SNAKE_SIZE, self.snake.prev_loc[1]//SNAKE_SIZE])
        fruit_location_encoding = np.array([self.fruit.loc[0]//FRUIT_SIZE, self.fruit.loc[1]//FRUIT_SIZE])
        snake_length_encoding = np.array([len(self.snake.loc)])
        return np.concatenate((grid.flatten(), view_size_encoding, snake_location_encoding, fruit_location_encoding, snake_length_encoding, snake_prev_location_encoding))

    def take_action(self, action):
        directions = ["UP", "DOWN", "LEFT", "RIGHT"]
        direction = directions[action]
        reward = self.snake.move(direction)
        if self.snake.loc[0] == list(self.fruit.loc):
            self.snake.eat()
            self.fruit = Fruit()
            self.score += 1
            self.current_score += 1
            return 10
        return reward

    def main_loop(self):
        while True:  # Run forever
            while self.running:
                if game.is_ai:
                    state = self.get_game_state()
                    action = self.agent.select_action(state)
                    reward = self.take_action(action)
                    next_state = self.get_game_state()
                    self.agent.memory.append((self.agent.preprocess_state(state), action, reward, self.agent.preprocess_state(next_state)))

                    self.agent.update_model()
                    self.agent.update_epsilon()

                for event in pygame.event.get():
                    if event.type == QUIT:
                        os._exit(0)
                    elif not self.is_ai:
                        if event.type == KEYDOWN:
                            if event.key == K_DOWN:
                                direction = "DOWN"
                            elif event.key == K_UP:
                                direction = "UP"
                            elif event.key == K_LEFT:
                                direction = "LEFT"
                            elif event.key == K_RIGHT:
                                direction = "RIGHT"
                            self.snake.move(direction)

                self.screen.fill((0, 0, 0))

                for loc in self.snake.loc:
                    snake_block = pygame.Rect(*loc, SNAKE_SIZE, SNAKE_SIZE)
                    pygame.draw.rect(self.screen, self.snake.color, snake_block)

                fruit_block = pygame.Rect(*self.fruit.loc, FRUIT_SIZE, FRUIT_SIZE)
                pygame.draw.rect(self.screen, self.fruit.color, fruit_block)

                pygame.display.flip()
                if not self.is_ai:
                    self.clock.tick(10)

            # Save the model and optimizer upon death
            torch.save({
                'model_state_dict': self.agent.state_dict(),
                'optimizer_state_dict': self.agent.optimizer.state_dict(),
            }, self.model_path)

            # Reset the game
            self.game_counter += 1
            self.highest = max(self.current_score, self.highest)
            self.current_score = 0
            os.system("cls")
            print(f"Highest score of current agent during this run is: {self.highest}")
            print(f"Average score per game is: {self.score / self.game_counter}")
            print(f"epsilon value is: {self.agent.epsilon}")
            print(f"{self.game_counter} games learned")
            self.snake = Snake(self)
            self.fruit = Fruit()
            self.running = True

pygame.init()
game = Game()
game.is_ai = True
game.main_loop()
pygame.quit()

