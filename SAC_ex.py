# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:43:09 2025

@author: julien.hautot
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque, namedtuple
import random
from datetime import datetime
import time
import matplotlib.pyplot as plt
import os

# --- Paramètres globaux du programme ---

ENV_NAME = "Hopper-v5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
SEED = 42
RETURN_AVG_WINDOW = 20

# --- Definition des fonctions utiles ---

def moving_average(x, window):
    """
    x : liste de scalaires
    window : nombre d'épisodes pour la moyenne
    """
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window)/window, mode="valid")

# --- Dossier du grid-search ---
gridsearch_time = datetime.now().strftime("%d_%m_%Y_%Hh_%M_%S")
gridsearch_dir = f"figures/SAC/gridsearch_{gridsearch_time}"

os.makedirs(gridsearch_dir, exist_ok=True)

# --- Main fonction d'apprentissage ---

def train_agent(
        MAX_STEPS = 50000, START_STEPS = 5000, UPDATE_AFTER = 500, UPDATE_EVERY = 2,
        BATCH_SIZE = 256, GAMMA = 0.99, TAU = 0.005, POLICY_LR = 3e-4, Q_LR = 1e-3,
        ALPHA_LR = 1e-3, HIDDEN = 256, REPLAY_SIZE = int(1e6), AUTOMATIC_ENTROPY_TUNING = True
        ):

    # Calcul de la date du début de l'entrainement
    start_time = time.time()

    # -------------------------
    # Hyperparamètres (à adapter pour TP)
    # -------------------------
    dict_params = {}
    dict_params["MAX_STEPS"] = MAX_STEPS
    dict_params["START_STEPS"] = START_STEPS
    dict_params["UPDATE_AFTER"] = UPDATE_AFTER
    dict_params["UPDATE_EVERY"] = UPDATE_EVERY
    dict_params["BATCH_SIZE"] = BATCH_SIZE
    dict_params["GAMMA"] = GAMMA
    dict_params["TAU"] = TAU
    dict_params["POLICY_LR"] = POLICY_LR
    dict_params["Q_LR"] = Q_LR
    dict_params["ALPHA_LR"] = ALPHA_LR
    dict_params["HIDDEN"] = HIDDEN
    dict_params["REPLAY_SIZE"] = REPLAY_SIZE
    dict_params["AUTOMATIC_ENTROPY_TUNING"] = AUTOMATIC_ENTROPY_TUNING

    # -------------------------
    # Utils : replay buffer
    # -------------------------
    Transition = namedtuple("Transition", ("s", "a", "r", "s2", "mask"))

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)

        def push(self, *args):
            self.buffer.append(Transition(*args))

        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            return Transition(*zip(*batch))

        def __len__(self):
            return len(self.buffer)

    # -------------------------
    # Networks
    # -------------------------
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    class MLP(nn.Module):
        def __init__(self, input_dim, output_dim, hidden=HIDDEN, activation=nn.ReLU):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden, device=DEVICE),
                activation(),
                nn.Linear(hidden, hidden, device=DEVICE),
                activation(),
                nn.Linear(hidden, output_dim, device=DEVICE)
            )
            self.net.apply(weight_init)

        def forward(self, x):
            return self.net(x)

    # Critic (Q network) : prend state et action
    class QNetwork(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.net = MLP(input_dim=state_dim+action_dim, output_dim=1, hidden=HIDDEN, activation=nn.ReLU)
        def forward(self, s, a):
            x = torch.cat([s, a], 1)
            return self.net(x)

    # Policy : retourne action sampleable et log_prob (avec correction tanh)
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    class GaussianPolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.shared = MLP(input_dim=state_dim, output_dim=2*action_dim, hidden=HIDDEN, activation=nn.ReLU)  
            self.action_dim = action_dim

        def forward(self, s):
            x = self.shared(s)
            mu, log_std = x[:, :self.action_dim], x[:, self.action_dim:]
            log_std = torch.tanh(log_std)
            # scale log_std to sensible range
            log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
            std = log_std.exp()
            return mu, std

        def sample(self, s):
            mu, std = self.forward(s)
            dist = Normal(mu, std)
            z = dist.rsample()
            action = torch.tanh(z)
            
            log_prob = dist.log_prob(z)
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            return action, log_prob, torch.tanh(mu)  



    # -------------------------
    # Agent SAC
    # -------------------------
    class SACAgent:
        def __init__(self, env):
            self.env = env
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.act_limit = float(env.action_space.high[0])

            # networks
            self.policy = GaussianPolicy(state_dim=self.state_dim, action_dim=self.action_dim)
            self.q1 = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
            self.q2 = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
            self.q1_target = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim)
            self.q2_target = QNetwork(state_dim=self.state_dim, action_dim=self.action_dim)

            # copy params to targets
            self.q1_target.load_state_dict(self.q1.state_dict())
            self.q2_target.load_state_dict(self.q2.state_dict())

            # optimizers
            self.policy_opt = optim.Adam(self.policy.parameters(), lr=POLICY_LR)
            self.q1_opt = optim.Adam(self.q1.parameters(), lr=Q_LR)
            self.q2_opt = optim.Adam(self.q2.parameters(), lr=Q_LR)

            # automatic entropy tuning
            if AUTOMATIC_ENTROPY_TUNING:
                # target_entropy = -|A|
                self.target_entropy = -self.action_dim
                # log alpha as parameter
                self.log_alpha = torch.tensor(0.0, requires_grad=True, device=DEVICE)
                self.alpha_opt = optim.Adam([self.log_alpha], lr=ALPHA_LR)
            else:
                self.alpha = 0.2

        def select_action(self, state, evaluate=False):
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                if evaluate:
                    _, _, mu = self.policy.sample(s)
                    action = mu
                    logp = None
                else:
                    a, logp, _ = self.policy.sample(s)
                    action = a
            action = action.cpu().numpy().squeeze(0)
            return action

        def update(self, replay_buffer, batch_size):
            
            transitions = replay_buffer.sample(batch_size)
            s = torch.FloatTensor(np.array(transitions.s)).to(DEVICE)
            a = torch.FloatTensor(np.array(transitions.a)).to(DEVICE)
            r = torch.FloatTensor(np.array(transitions.r)).to(DEVICE).unsqueeze(-1)
            s2 = torch.FloatTensor(np.array(transitions.s2)).to(DEVICE)
            # Ici, 'd' contient maintenant notre masque (0 si mort, 1 sinon)
            mask = torch.FloatTensor(np.array(transitions.mask)).to(DEVICE).unsqueeze(-1)

            
            # --- compute target Q value ---
            with torch.no_grad():
                a2, logp_a2, _ = self.policy.sample(s2)
                q1_t = self.q1_target(s2, a2)
                q2_t = self.q2_target(s2, a2)
                q_target_min = torch.min(q1_t, q2_t)
                if AUTOMATIC_ENTROPY_TUNING:
                    alpha = self.log_alpha.exp()
                else:
                    alpha = self.alpha
                # target y = r + gamma*(min_q - alpha * logp_a2)
                # On multiplie le futur par le masque !
                # y = r + GAMMA * mask * (Valeur Future)
                next_q_value = q_target_min - alpha * logp_a2
                y = r + GAMMA * mask * next_q_value

            # --- Q losses ---
            q1_pred = self.q1(s, a)
            q2_pred = self.q2(s, a)
            q1_loss = nn.MSELoss()(q1_pred, y)
            q2_loss = nn.MSELoss()(q2_pred, y)

            self.q1_opt.zero_grad()
            q1_loss.backward()
            self.q1_opt.step()

            self.q2_opt.zero_grad()
            q2_loss.backward()
            self.q2_opt.step()

            # --- Policy loss ---
            a_new, logp_new, _ = self.policy.sample(s)
            q1_new = self.q1(s, a_new)
            q2_new = self.q2(s, a_new)
            q_new_min = torch.min(q1_new, q2_new)

            if AUTOMATIC_ENTROPY_TUNING:
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            policy_loss = (alpha*logp_new - q_new_min).mean()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            # --- entropy (alpha) tuning ---
            if AUTOMATIC_ENTROPY_TUNING:
                alpha_loss = (-self.log_alpha * (logp_new + self.target_entropy).detach()).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                alpha = self.log_alpha.exp()
            else:
                alpha = self.alpha

            # --- soft update targets ---
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            
            return {
                "q1_loss": q1_loss.item(),
                "q2_loss": q2_loss.item(),
                "policy_loss": policy_loss.item(),
                "alpha": alpha if not isinstance(alpha, torch.Tensor) else alpha.item()
            }



    env = gym.make(ENV_NAME)
    env.reset(seed=SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    agent = SACAgent(env)
    replay = ReplayBuffer(REPLAY_SIZE)

    total_steps = 0
    episode = 0
    ep_return = 0
    ep_len = 0

    # Listes pour stocker les récompenses obtenues au cours de l'apprentissage
    return_list = []

    state, _ = env.reset()
    while total_steps < MAX_STEPS:
        if total_steps < START_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False)

        next_state, reward, terminated, truncated, _ = env.step(action)
    # --- CORRECTION MASQUE ---
        # Si 'terminated' (mort) -> mask = 0
        # Si 'truncated' (temps écoulé) -> mask = 1 (car on veut continuer d'estimer la valeur)
        # Si rien -> mask = 1
        done = terminated or truncated
        mask = 1.0 if truncated else float(not done)

        # On stocke le 'mask' à la place du 'done' booléen pour simplifier l'update
        replay.push(state, action, reward, next_state, mask)

        state = next_state
        ep_return += reward
        ep_len += 1
        total_steps += 1

        if done:
            return_list.append(ep_return)
            episode += 1
            print(f"Episode {episode} | Steps {total_steps} | Return {ep_return:.2f} | Len {ep_len}")
            state, _ = env.reset()
            ep_return = 0
            ep_len = 0

        
        if total_steps >= UPDATE_AFTER and total_steps % UPDATE_EVERY == 0:
            for j in range(UPDATE_EVERY):
                if len(replay) < BATCH_SIZE:
                    continue
                info = agent.update(replay, BATCH_SIZE)
    
    env.close()

    # --- Statistiques d'apprentissage ---

    # Calcul de la date de la fin de l'entrainement
    end_time = time.time()
    training_time_sec = end_time - start_time

    # Calcul du temps de l'apprentissage
    training_time_str = time.strftime(
        "%H:%M:%S",
        time.gmtime(training_time_sec)
    )

    dict_params["training_time"] = training_time_str

    # --- Affichage des courbes d'apprentissages ---
    smoothed_returns = moving_average(return_list, RETURN_AVG_WINDOW)
    
    plt.figure()
    plt.plot(
        np.arange(len(smoothed_returns)),
        smoothed_returns,
        label=f"Return moyenné ({RETURN_AVG_WINDOW} épisodes)",
        linewidth=2
    )
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.title("Métriques d'entrainement")

    # Affichage des hyper-paramètres utilisés sur la figure
    params_text = "\n".join([f"{k}: {v}" for k, v in dict_params.items()])
    plt.figtext(
        0.02, 0.02,
        f"Hyperparamètres:\n{params_text}",
        fontsize=9,
        ha="left",
        va="bottom",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
    )

    # Sauvegarde du graphique d'apprentissage
    filename = f"{gridsearch_dir}/loss_curves_{datetime.now().strftime('%d_%m_%Y_%Hh_%M_%S')}.png"
    plt.savefig(
        filename, bbox_inches="tight"
    )
    plt.close()

    return return_list, filename


# --- Entrainement de l'agent en faisant de la recherche d'hyper-paramètres ---

# Définition des HP à tester
lr_list = [1e-4, 3e-4, 1e-3]
batch_size_list = [128, 256]
tau_list = [0.005, 0.01]


# Grid-search
all_trains = []

for lr_i in lr_list:
    for batch_size_i in batch_size_list:
        for tau_i in tau_list:
            return_, filename = train_agent(
                POLICY_LR=lr_i, Q_LR=3*lr_i, ALPHA_LR=lr_i, BATCH_SIZE=batch_size_i, TAU=tau_i
            )

            all_trains.append((return_, filename))


# --- Affichage des résultats du grid-search ---

plt.figure(figsize=(12, 7))

for return_, filename in all_trains:
    smoothed = moving_average(return_, RETURN_AVG_WINDOW)
    label = filename.split("/")[-1].replace(".png", "")
    plt.plot(smoothed, label=label)

plt.xlabel("Episode")
plt.ylabel(f"Return moyen sur {RETURN_AVG_WINDOW} épisodes")
plt.title("Comparaison des courbes d'apprentissage")
plt.legend(fontsize=8)
plt.grid(True)

plt.savefig(
    f"{gridsearch_dir}/global_comparison_{datetime.now().strftime('%d_%m_%Y_%Hh_%M_%S')}.png",
    bbox_inches="tight"
)

plt.show()