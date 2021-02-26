# Jakub Gołębiowski
# PZ2
# 1.802.2021

import torch
import random
import numpy as np
from collections import deque  # kolejka obustoronna: queue and stack w jednym
from snake_game import SnakeGameAI, Direction, Point
from model import LinearQNet, LinearQNet2, LinearQNet3, LinearQNet4,  QTrainer
import matplotlib.pyplot as plt

MAX_MEMORY = 10_000_000
BATCH_SIZE = 1_000
LR = 0.0000001


class Agent:

    def __init__(self):
        self.number_games = 0
        self.epsilon = 0  # lvl of randomness
        self.gamma = 0.99999  # TODO: play with this
        self.memory = deque(maxlen=MAX_MEMORY)  # automatycznie wyrzuca wszystko po lewej przy przekroczeniu maxlen

        # self.model = LinearQNet(11, 256, 3)
        #self.model = LinearQNet2(11+4, 256, 256, 3)
        self.model = LinearQNet3(11+4, 256, 256, 256, 3)
        #self.model = LinearQNet4(11+4, 256, 256, 256, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        # sprawdzanie niebezpiecznych miejsc (skrętów)
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        point_lu = Point(head.x - 20, head.y - 20)
        point_ld = Point(head.x - 20, head.y + 20)
        point_ru = Point(head.x + 20, head.y - 20)
        point_rd = Point(head.x + 20, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Niebezpieczeństwo z przodu
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_u)),

            # Niebezpieczeństwo z prawej
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Niebezpieczeństwo z lewej
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Niebezpieczeństwo z RU ####################################################### TODO: check props
            (dir_r and game.is_collision(point_rd)) or
            (dir_u and game.is_collision(point_ru)) or
            (dir_l and game.is_collision(point_lu)) or
            (dir_d and game.is_collision(point_ld)),

            # Niebezpieczeństwo z LU
            (dir_r and game.is_collision(point_ru)) or
            (dir_u and game.is_collision(point_lu)) or
            (dir_l and game.is_collision(point_ld)) or
            (dir_d and game.is_collision(point_rd)),

            # Niebezpieczeństwo z RD
            (dir_r and game.is_collision(point_ld)) or
            (dir_u and game.is_collision(point_rd)) or
            (dir_l and game.is_collision(point_ru)) or
            (dir_d and game.is_collision(point_lu)),

            # Niebezpieczeństwo z LD
            (dir_r and game.is_collision(point_lu)) or
            (dir_u and game.is_collision(point_ld)) or
            (dir_l and game.is_collision(point_rd)) or
            (dir_d and game.is_collision(point_ru)),

            # ############################################################################### END

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # lokacja pokarmu (świartki wokół węża)
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # zwraca listę tuple
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.number_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)

            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_score = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # pobiera stary stan
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # preform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long term memory, plt results
            game.reset()
            agent.number_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', agent.number_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.number_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)
            plt.close()
            plt.plot(plot_scores,'-')
            plt.plot(plot_mean_scores,"-")
            plt.title("Trening Q deep learning gry w snake'a")
            plt.xlabel('Liczba gier')
            plt.ylabel('Punkty')
            plt.savefig('runTest.png')


def test():
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_score = 0
    n_games = 0
    agent = Agent()
    agent.number_games = 200
    #agent.model = LinearQNet(11,256,3)
    #agent.model.load_state_dict(torch.load("model/model02.pth"))
    #agent.model.load_state_dict(torch.load("model3/model3_57.pth"))
    agent.model.load_state_dict(torch.load("model3/model3_59.pth"))
    #agent.model.load_state_dict(torch.load("model15/model02_15_256_256_3-38.pth"))
    #agent.model.load_state_dict(torch.load("model4/model4_best_70.pth"))
    agent.model.eval()
    game = SnakeGameAI()
    while True:
        # pobiera stary stan
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # preform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long term memory, plt results
            game.reset()
            n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game: ', n_games, 'Score: ', score, 'Record: ', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            #plot(plot_scores, plot_mean_scores)
            plt.close()
            plt.plot(plot_scores,'-')
            plt.plot(plot_mean_scores,"-")
            plt.title("Trening Q deep learning gry w snake'a")
            plt.xlabel('Liczba gier')
            plt.ylabel('Punkty')
            plt.savefig('runTest.png')


if __name__ == '__main__':
    #train()
    test()

