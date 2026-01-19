# **Reinforcement Learning: PPO & SAC Implementation**

This repository contains the implementation of two major Deep Reinforcement Learning algorithms: **Proximal Policy Optimization (PPO)** and **Soft Actor-Critic (SAC)**.

These implementations were developed to solve the **Hopper-v5** continuous control environment from Gymnasium. This project was carried out as part of the 5th-year Reinforcement Learning course at **Polytech Clermont**.

## **👥 Authors**

* **Nathan** (nathan7128)  
* **Khalid El Bazi**

## **📝 Project Overview**

The goal of this project is to implement, train, and compare on-policy (PPO) and off-policy (SAC) algorithms on a MuJoCo continuous control task.

### **Algorithms Implemented**

1. **PPO (Proximal Policy Optimization):**  
   * Implements a Gaussian Policy for continuous actions.  
   * Uses Generalized Advantage Estimation (GAE).  
   * Includes a Value network (Critic) and Policy network (Actor).  
   * Features a built-in Grid Search for hyperparameter tuning (Gamma, Learning Rate, Clipping epsilon, Entropy coefficient, etc.).  
2. **SAC (Soft Actor-Critic):**  
   * Implements the maximum entropy framework.  
   * Uses double Q-learning (two Critic networks) to reduce overestimation bias.  
   * Includes **Automatic Entropy Tuning** (alpha).  
   * Features a Replay Buffer for off-policy learning.

## **📂 Repository Structure**

* PPO\_ex.py: Complete source code for the PPO agent, training loop, and grid search.  
* SAC\_ex.py: Complete source code for the SAC agent, replay buffer, and grid search.  
* requirements.txt: Python dependencies required to run the project.  
* figures/: Directory where training curves and comparison plots are saved automatically.

## **🚀 Installation**

To run this project, you need Python installed along with the required libraries. We recommend using a virtual environment.

1. **Clone the repository:**  
   git clone \[https://github.com/nathan7128/sac-ppo-implementation.git\](https://github.com/nathan7128/sac-ppo-implementation.git)  
   cd sac-ppo-implementation

2. **Install dependencies:**  
   pip install \-r requirements.txt

   *Note: Ensure you have a working installation of MuJoCo for Gymnasium environments.*

## **⚙️ Usage**

Both scripts are designed to perform a **Grid Search** over specified hyperparameters by default. They will train the agent for a set number of epochs/steps and save the loss/reward curves in the figures/ folder.

### **Running PPO**

To train the PPO agent:

python PPO\_ex.py

*You can modify the gamma\_list, lr\_list, and other hyperparameters directly in the PPO\_ex.py file under the "Définition des HP à tester" section.*

### **Running SAC**

To train the SAC agent:

python SAC\_ex.py

*Similarly, hyperparameters for the grid search (LR, Batch Size, Tau) can be adjusted at the bottom of SAC\_ex.py.*

## **📊 Results**

The training scripts automatically generate:

1. **Training Curves:** Reward evolution over epochs/episodes for each specific run.  
2. **Global Comparison:** A combined plot comparing the performance of all hyperparameter combinations tested during the grid search.

Output files are stored in:

* figures/PPO/gridsearch\_\<timestamp\>/  
* figures/SAC/gridsearch\_\<timestamp\>/

## **🛠️ Built With**

* [Python](https://www.python.org/)  
* [PyTorch](https://pytorch.org/) \- Deep Learning Framework  
* [Gymnasium](https://gymnasium.farama.org/) \- RL Environments (Hopper-v5)  
* [Matplotlib](https://matplotlib.org/) \- Visualization
