# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 14:36:26 2025

@author: julien.hautot
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# --- Paramètres globaux du programme ---

env_name = "Hopper-v5"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dossier du grid-search ---
gridsearch_time = datetime.now().strftime("%d_%m_%Y_%Hh_%M_%S")
gridsearch_dir = f"figures/PPO/gridsearch_{gridsearch_time}"

os.makedirs(gridsearch_dir, exist_ok=True)

# --- Main fonction d'apprentissage ---

def train_agent(
        gamma=0.99, lr=3e-4, clip_eps=0.2, epochs=100, steps_per_epoch=4096,
        batch_size=128, entropy_coef=0.01, hidden_dim=256
        ):

    # Calcul de la date du début de l'entrainement
    start_time = time.time()

    # --- Hyperparamètres ---
    dict_params = {}
    dict_params["gamma"] = gamma
    dict_params["lr"] = lr
    dict_params["clip_eps"] = clip_eps
    dict_params["epochs"] = epochs
    dict_params["steps_per_epoch"] = steps_per_epoch
    dict_params["batch_size"] = batch_size
    dict_params["entropy_coef"] = entropy_coef
    dict_params["hidden_dim"] = hidden_dim

    # --- Environnement ---
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # --- Réseau Policy (Gaussian) ---
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            # Construction du modèle séquentiel de l'acteur avec 2 couches cachées
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim, device=device),
                nn.Tanh(),
                nn.Linear(hidden_dim, act_dim, device=device),
                nn.Tanh()
            )
            # Variance de la politique qui est update au cours de l'apprentissage
            self.log_std = nn.Parameter(torch.zeros(act_dim))  

        def forward(self, x):
            # On calcule les paramètres de la distribution de la politique
            mu = self.net(x)
            std = torch.exp(self.log_std)
            return mu, std

    # --- Réseau Value ---
    class Value(nn.Module):
        def __init__(self):
            super().__init__()
            # Construction du modèle séquentiel du critique avec 3 couches cachées
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim, device=device),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim, device=device),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.net(x)

    # --- Initialisation ---
    policy = Policy().to(device)
    value = Value().to(device)
    optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
    optimizer_value = optim.Adam(value.parameters(), lr=lr)

    # --- Fonction pour générer trajectoires ---
    def collect_trajectories():
        obs = env.reset()[0]
        obs_list, act_list, rew_list, logp_list, val_list, done_list = [], [], [], [], [], []
        for _ in range(steps_per_epoch):
            obs_tensor = torch.FloatTensor(obs).to(device)
            # On infère les paramètres de la distribution
            mu, std = policy(obs_tensor)
            # Définition de la distribution Torch
            dist = Normal(mu, std)
            # On génère une action via la distribution instanciée
            act = dist.sample()
            logp = dist.log_prob(act).sum()
            # On évalue le vecteur d'observation via le réseau Critic
            val = value(obs_tensor)

            # Scale action to environment
            # On normalize la valeur de l'action en fonction de l'espace d'action de l'environnement [-1, 1]
            act_clamped = nn.Tanh()(act)
            # Nouvelle étape dans l'épisode
            next_obs, rew, terminated, truncated, _ = env.step(act_clamped.cpu().detach().numpy())
            done = terminated or truncated

            # Stockage
            obs_list.append(obs)
            act_list.append(act.detach())
            rew_list.append(rew)
            logp_list.append(logp.detach())
            val_list.append(val.detach())
            done_list.append(done)

            obs = next_obs
            if done:
                obs = env.reset()[0]

        return obs_list, act_list, rew_list, logp_list, val_list, done_list

    # --- Fonction pour calculer avantages ---
    def compute_advantages(rews, vals, dones):
        advs, gae = [], 0
        vals = vals + [0]  # V(s_T) = 0
        for t in reversed(range(len(rews))):
            # Estimation de l'avantage via Bootstrapping
            delta = rews[t] + gamma * vals[t + 1] * (1 - dones[t]) - vals[t]
            # Calcul du GAE qui est une moyenne pondérée des avantages estimés
            gae = delta + gamma * 0.95 * (1 - dones[t]) * gae
            advs.insert(0, gae)
        # Calcul du return qui est égale à la valeur ciblée
        returns = [adv + val for adv, val in zip(advs, vals[:-1])]
        return torch.FloatTensor(advs).to(device), torch.FloatTensor(returns).to(device)

    # --- Entraînement ---

    # Listes pour stocker les récompenses moyennes obtenues au cours de l'apprentissage
    avg_rew_list = []

    # On utilise la MSE pour la fonction cout du réseau Critic
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):

        # --- Affichage d'un épisode tous les 20 epochs ---
        if ((epoch + 1)%20 == 0):

            env_test = gym.make(env_name, render_mode="human")
            done = False
            obs = env_test.reset()[0]
            while not done:
                obs_tensor = torch.FloatTensor(obs).to(device)
                mu, std = policy(obs_tensor)
                dist = Normal(mu, std)
                act = dist.sample()
                act_clamped = nn.Tanh()(act)
                next_obs, rew, terminated, truncated, _ = env_test.step(act_clamped.cpu().detach().numpy())
                obs = next_obs
                done = terminated or truncated
                env_test.render()
            env_test.close()
        # ------------------------------

        # On récupère les trajectoires obtenues avec l'ancienne politique
        obs_list, act_list, rew_list, logp_list, val_list, done_list = collect_trajectories()
        advs, returns = compute_advantages(rew_list, val_list, done_list)
        
        obs_tensor = torch.FloatTensor(obs_list).to(device)
        act_tensor = torch.stack(act_list).to(device)
        old_logp_tensor = torch.stack(logp_list).to(device)
        
        # --- Mise à jour PPO ---
        for _ in range(10):  # 10 mini-epochs
            idx = np.random.permutation(len(obs_list))
            for start in range(0, len(obs_list), batch_size):
                # Création des batchs de données
                end = start + batch_size
                batch_idx = idx[start:end]

                obs_b = obs_tensor[batch_idx].to(device)
                act_b = act_tensor[batch_idx].to(device)
                adv_b = advs[batch_idx].to(device)
                ret_b = returns[batch_idx].to(device)
                old_logp_b = old_logp_tensor[batch_idx].to(device)

                # Policy
                # Calcul des paramètres de la distribution de la politique actuelle
                mu, std = policy(obs_b)
                # Distribution loi Normale en fonction de ces paramètres
                dist = Normal(mu, std)
                logp = dist.log_prob(act_b).sum(axis=-1)
                # Calcul de l'entropie de la loi Normale instanciée
                entropy = dist.entropy().sum(axis=-1).mean()
                # On applique une exponentielle aux log probs pour calculer le ratio
                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio*adv_b
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 +  clip_eps)*adv_b
                # Loss du réseau Acteur
                loss_policy = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

                # Value
                val_pred = value(obs_b)
                # La perte du réseau Critique est la MSE entre les valeurs prédites et les valeurs ciblées
                loss_value = loss_fn(val_pred.squeeze(), ret_b)

                # Backprop
                optimizer_policy.zero_grad()
                loss_policy.backward()
                optimizer_policy.step()

                optimizer_value.zero_grad()
                loss_value.backward()
                optimizer_value.step()

        # --- Stats ---
        avg_rew = np.mean(rew_list)
        print(f"Epoch {epoch+1} | Avg Reward = {avg_rew:.2f}")

        avg_rew_list.append(avg_rew)

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
    plt.figure()
    plt.plot(avg_rew_list, label="Avg Reward")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Values")
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

    return avg_rew_list, filename


# --- Entrainement de l'agent en faisant de la recherche d'hyper-paramètres ---

# Définition des HP à tester
gamma_list = [0.99]
lr_list = [3e-4, 3e-5]
clip_eps_list = [0.05, 0.2, 0.5]
epochs_list = [100]
steps_per_epoch_list = [4096]
batch_size_list = [128]
entropy_coef_list = [0.01, 0.1, 0.5]
hidden_dim_list = [256]


# Grid-search
all_trains = []

for gamma_i in gamma_list:
    for lr_i in lr_list:
        for clip_eps_i in clip_eps_list:
            for epochs_i in epochs_list:
                for steps_per_epoch_i in steps_per_epoch_list:
                    for batch_size_i in batch_size_list:
                        for entropy_coef_i in entropy_coef_list:
                            for hidden_dim_i in hidden_dim_list:
                                avg_rew, filename = train_agent(
                                    gamma_i, lr_i, clip_eps_i, epochs_i, steps_per_epoch_i,
                                    batch_size_i, entropy_coef_i, hidden_dim_i
                                )

                                all_trains.append((avg_rew, filename))


# --- Affichage des résultats du grid-search ---

plt.figure(figsize=(12, 7))

for avg_rew, filename in all_trains:
    label = filename.split("/")[-1].replace(".png", "")
    plt.plot(avg_rew, label=label)

plt.xlabel("Epochs")
plt.ylabel("Avg Reward")
plt.title("Comparaison des courbes d'apprentissage")
plt.legend(fontsize=8)
plt.grid(True)

plt.savefig(
    f"{gridsearch_dir}/global_comparison_{datetime.now().strftime('%d_%m_%Y_%Hh_%M_%S')}.png",
    bbox_inches="tight"
)

plt.show()