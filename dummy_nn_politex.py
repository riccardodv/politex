import gym
import numpy as np
from scipy.special import softmax
import torch as th

# TODO:  replay buff, and batch update

class Qnet(th.nn.Module):
    # Generic Q net class (we start small)
    def __init__(
        self,
        env,
        hidden_size=32,
        activ=th.nn.Tanh,
        optimizer=th.optim.Adam,
    ):
        super().__init__()
        self.env = env
        self.hidden_size = hidden_size
        self.lin1 = th.nn.Linear(self.env.observation_space.shape[0] + 1, self.hidden_size)
        self.lin2 = th.nn.Linear(self.hidden_size, 1)
        self.activ = activ()

        self.optimizer = optimizer(self.parameters(), lr = 1e-3)
        self.optimizer.zero_grad()

    def forward(self, s_a):
        hid1 = self.lin1(s_a)
        # activ = self.activ(hid1)
        hid2 = self.lin2(hid1)
        return hid2

class Pi(th.nn.Module):
    # Policy class to simplify summing q funcs
    def __init__(self, q_pi_0):
        super().__init__()

        self.lin1 = q_pi_0.lin1
        self.lin2 = q_pi_0.lin2
        self.activ = q_pi_0.activ
        self.actions = range(q_pi_0.env.action_space.n)

    def forward(self, s_a):
        with th.no_grad():
            hid1 = self.lin1(s_a)
            # activ = self.activ(hid1)
            hid2 = self.lin2(hid1)
        return hid2

    def add_q(self, q_pi_k):
        with th.no_grad():
            self.lin1.weight.copy_(self.lin1.weight + q_pi_k.lin1.weight)
            self.lin2.weight.copy_(self.lin2.weight + q_pi_k.lin2.weight)

    def get_logits(self, s):
        logits = []
        for a in self.actions:
            s_a = np.append(s, [a])
            logit = self.forward(th.FloatTensor(s_a))
            logits.append(logit.item())
        return logits


# Create the CartPole environment
env = gym.make('CartPole-v1')
A = range(env.action_space.n)
loss = th.nn.MSELoss()
gamma = 0.99
eta = 1
q_pi_k = Qnet(env) #in reality it is q_pi_k-1
pi = Pi(q_pi_k)

for iter in range(10000):
    ###### Q evaluation !!! #######
    s = env.reset()
    logits_a = pi.get_logits(s)
    p_a = softmax(eta * logits_a)
    a = np.random.choice(A, p = p_a)
    done = False
    for steps in range(128):
        snext,r,done,_ = env.step(a)
        logits_anext = pi.get_logits(snext)
        p_anext = softmax(eta * logits_anext)
        anext = np.random.choice(A, p =  p_anext)

        q_s_a = q_pi_k.forward(th.FloatTensor(np.append(s, [a])))
        with th.no_grad():
            target = r + gamma * q_pi_k.forward(th.FloatTensor(np.append(snext, [anext])))
        l = loss(q_s_a, target)
        l.backward()
        q_pi_k.optimizer.step()
        q_pi_k.optimizer.zero_grad()
        s, a = snext, anext

        if done:
            s = env.reset()
            logits_a = pi.get_logits(s)
            p_a = softmax(eta * logits_a)
            a = np.random.choice(A, p = p_a)
    ###################################

    ############## Pol update !! ######
    pi.add_q(q_pi_k)
    ###################################
    # init new nn for q_pi_k
    q_pi_k = Qnet(env)


    ######### Evals !! ################
    if (iter + 1)% 100 == 0:
        s = env.reset()
        done = False
        sum_r = 0
        while not done:
            logits_a = pi.get_logits(s)
            p_a = softmax(eta * logits_a)
            a = np.random.choice(A, p = p_a)
            s,r,done,_ = env.step(a)
            sum_r += r
        print("eval rewards iter {}/10000: {}".format(iter+1, sum_r))
