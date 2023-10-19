import torch
import random
import numpy as np
from game import SnakeGameML, Direction, Point
from collections import deque
from model import Linear_Net, Trainer
from plotter import plot

#TODO:
#Remember Function
#In Game Memory
#Between Game Memory
#Action Definer
#Plotter


MAX_MEMORY = 1000000
BATCH_SIZE = 1000

LR = 0.001

class Agent:

    def __init__(self):
        self.game_iter = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_Net(11, 256, 3)  # Hidden Size can change
        self.trainer = Trainer(self.model, linreg=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # Pick a point arbitrarily far out
        left_point = Point(head.x - 20, head.y)
        right_point = Point(head.x + 20, head.y)
        up_point = Point(head.x, head.y - 20)
        down_point = Point(head.x, head.y + 20)

        up_boardD, right_boardD,\
        down_boardD, left_boardD = \
            game.is_collision(Point(head.x, head.y-20)), game.is_collision(Point(head.x + 20, head.y)),\
            game.is_collision(Point(head.x, head.y + 20)), game.is_collision(Point(head.x + 20, head.y))

        danger_list = [left_boardD, up_boardD, right_boardD, down_boardD]
        if game.direction == Direction.UP:
            left_dir, right_dir, up_dir, down_dir = [False, False, True, False]
            leftD, aheadD, rightD = danger_list[:3]
        elif game.direction == Direction.RIGHT:
            left_dir, right_dir, up_dir, down_dir = [False, True, False, False]
            leftD, aheadD, rightD = danger_list[1:4]
        elif game.direction == Direction.DOWN:
            left_dir, right_dir, up_dir, down_dir = [False, False, False, True]
            leftD, aheadD, rightD = danger_list[-2:] + danger_list[:1]
        else: # Left
            left_dir, right_dir, up_dir, down_dir = [True, False, False, False]
            leftD, aheadD, rightD = danger_list[-1:] + danger_list[:2]

        state = [

            #Danger Ahead:: Danger Snake Thinks is Right
            aheadD,
            #Danger Right
            rightD,
            #Danger Left
            leftD,

            left_dir,
            right_dir,
            up_dir,
            down_dir,

            game.food.x < game.head.x, # Food Left
            game.food.x > game.head.x, # Food Right
            game.food.y < game.head.y, # Food Up
            game.food.y > game.head.y, # Food Down


        ]

        return np.array(state, dtype=int) #Changes State to 0,1s

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done)) # 1 tuple

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # List of tuples
        else:
            mini_sample = self.memory

        states , actions, rewards, next_states, dones = zip(*mini_sample) #Concats each state, each action
        self.trainer.train_step(states, actions, rewards, next_states, dones) #Calls for each state, action etc

    def train_short_memory(self,state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random Move: Explore vs Exploit tradeoff Read Up More
        self.epsilon = 80 - self.game_iter #TODO:Change
        final_move = [0,0,0]

        #Setting random move based on epsilon
        if random.randint(0,200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameML()
    while True:
        state_old = agent.get_state(game)

        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #train long mem, and plot results
            game.reset()
            agent.game_iter += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.game_iter, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.game_iter
            plot_mean_scores.append(mean_score)
            print("Plotted")
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()