import rospkg
import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


TrainingRecord = namedtuple("TrainingRecord", ["ep", "reward"])
Transition = namedtuple("Transition", ["s", "a", "r", "s_"])


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 100)
        self.mu_head = nn.Linear(100, 1)

    def forward(self, s):
        x = F.relu(self.fc(s))
        u = self.mu_head(x)
        return u


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(4, 100)
        self.v_head = nn.Linear(100, 1)

    def forward(self, s, a):
        x = F.relu(self.fc(torch.cat([s, a], dim=1)))
        state_value = self.v_head(x)
        return state_value


class Memory:

    data_pointer = 0
    isfull = False

    def __init__(self, capacity):
        self.memory = np.empty(capacity, dtype=object)
        self.capacity = capacity

    def update(self, transition):
        self.memory[self.data_pointer] = transition
        self.data_pointer += 1
        if self.data_pointer == self.capacity:
            self.data_pointer = 0
            self.isfull = True

    def sample(self, batch_size):
        return np.random.choice(self.memory, batch_size)


class Agent:

    max_grad_norm = 0.5

    def __init__(self):
        self.training_step = 0
        self.var = 3.0
        self.gamma = 0.9
        self.eval_cnet, self.target_cnet = CriticNet().float(), CriticNet().float()
        self.eval_anet, self.target_anet = ActorNet().float(), ActorNet().float()
        self.memory = Memory(2000)
        self.optimizer_c = optim.Adam(self.eval_cnet.parameters(), lr=1e-3)
        self.optimizer_a = optim.Adam(self.eval_anet.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.eval_anet(state)
        dist = Normal(mu, torch.tensor(self.var, dtype=torch.float))
        action = dist.sample()
        return (action.item(),)

    def save_param(self):
        torch.save(self.eval_anet.state_dict(), "param/ddpg_anet_params.pkl")
        torch.save(self.eval_cnet.state_dict(), "param/ddpg_cnet_params.pkl")

    def store_transition(self, transition):
        self.memory.update(transition)

    def update(self):
        self.training_step += 1

        transitions = self.memory.sample(32)
        s = torch.tensor([t.s for t in transitions], dtype=torch.float)
        a = torch.tensor([t.a for t in transitions], dtype=torch.float).view(-1, 1)
        r = torch.tensor([t.r for t in transitions], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in transitions], dtype=torch.float)

        with torch.no_grad():
            q_target = r + self.gamma * self.target_cnet(s_, self.target_anet(s_))
        q_eval = self.eval_cnet(s, a)

        # update critic net
        self.optimizer_c.zero_grad()
        c_loss = F.smooth_l1_loss(q_eval, q_target)
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_cnet.parameters(), self.max_grad_norm)
        self.optimizer_c.step()

        # update actor net
        self.optimizer_a.zero_grad()
        a_loss = -self.eval_cnet(s, self.eval_anet(s)).mean()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_anet.parameters(), self.max_grad_norm)
        self.optimizer_a.step()

        if self.training_step % 200 == 0:
            self.target_cnet.load_state_dict(self.eval_cnet.state_dict())
        if self.training_step % 201 == 0:
            self.target_anet.load_state_dict(self.eval_anet.state_dict())

        self.var = max(self.var * 0.99, 0.01)

        return q_eval.mean().item()


class InferenceAgent:
    def __init__(self, model_name="trained_actor.pkl"):
        self.nn = ActorNet().float()
        rospack = rospkg.RosPack()
        file_path = rospack.get_path("inverted_pendulum_rl_control") + "/models/"
        file_str = file_path + model_name
        self.nn.load_state_dict(torch.load(file_str))

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu = self.nn(state)
        if mu.item() > 50.0:
            mu = torch.tensor(50.0)
        if mu.item() < -50.0:
            mu = torch.tensor(-50.0)
        dist = Normal(mu, torch.tensor(0.1, dtype=torch.float))
        action = dist.sample()
        return (action.item(),)


def main():
    env = gym.make("Pendulum-v1")
    env.seed(np.random.randint(0, 100))

    agent = Agent()

    training_records = []
    running_reward, running_q = -1000, 0
    for i_ep in range(1000):
        score = 0
        state = env.reset()

        for t in range(200):
            action = agent.select_action(state)
            state_, reward, done, _ = env.step(action)
            score += reward
            agent.store_transition(Transition(state, action, (reward + 8) / 8, state_))
            state = state_
            if agent.memory.isfull:
                q = agent.update()
                running_q = 0.99 * running_q + 0.01 * q

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainingRecord(i_ep, running_reward))

        if running_reward > -200:
            print("Solved! Running reward is now {}!".format(running_reward))
            env.close()
            agent.save_param()
            with open("log/ddpg_training_records.pkl", "wb") as f:
                pickle.dump(training_records, f)
            break

    env.close()

    plt.plot([r.ep for r in training_records], [r.reward for r in training_records])
    plt.title("DDPG")
    plt.xlabel("Episode")
    plt.ylabel("Moving averaged episode reward")
    plt.savefig("img/ddpg.png")
    plt.show()


if __name__ == "__main__":
    main()
