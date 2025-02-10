# Distributional Point-Based Value Iteration (DPBVI)

This repository contains implementations of several reinforcement learning algorithms for solving Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs), including:
1. **Value Iteration (VI)** – [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)
2. **Distributional Value Iteration (DVI)** – [Bellemare et al., 2023](https://www.distributional-rl.org/)
3. **Point-Based Value Iteration (PBVI)** – [Pineau et al., 2003](http://www.cs.cmu.edu/~ggordon/jpineau-ggordon-thrun.ijcai03.pdf) 
4. **Distributional Point-Based Value Iteration (DPBVI)** – Our proposed method, extending PBVI to the distributional reinforcement learning setting.

## 📂 Repository Structure
```
distributional_point_based_value_iteration/
│── agents/
│   ├── dvi.py        # Distributional Value Iteration (DVI) implementation
│   ├── pbvi.py       # Point-Based Value Iteration (PBVI) implementation
│   ├── dpbvi.py      # Distributional PBVI (DPBVI) implementation (our work)
├── vi_cliff_walking_example.py   # Value Iteration on Cliff Walking environment
├── dvi_cliff_walking_example.py  # DVI evaluation on Cliff Walking
├── minigrid_eval.py              # PBVI & DPBVI evaluation on MiniGrid-DoorKey-5x5-v0
├── trivial_example.py            # PBVI & DPBVI evaluation on noisy two-state problem
├── minigrid_counters.pkl         # Recorded observations & transitions (used for extracting transition and sensor models)
│── README.md
│── requirements.txt
│── LICENSE
```

## 📝 Implemented Algorithms

### 1️⃣ **Value Iteration (VI)**
- Implements classical **value iteration** for MDPs.
- **File:** `vi_cliff_walking_example.py`
- **Reference:** Sutton & Barto, *Reinforcement Learning: An Introduction (2018)*.

### 2️⃣ **Distributional Value Iteration (DVI)**
- Implements **distributional RL** for solving MDPs.
- **Files:**
  - Implementation: `agents/dvi.py`
  - Evaluation: `dvi_cliff_walking_example.py`
- **Reference:** Bellemare et al., *Distributional Reinforcement Learning (2023)*.

### 3️⃣ **Point-Based Value Iteration (PBVI)**
- Implements **point-based** methods for solving POMDPs.
- **Files:**
  - Implementation: `agents/pbvi.py`
  - Evaluation: `minigrid_eval.py`, `trivial_example.py`
- **Reference:** Pineau et al., *Point-Based Value Iteration: An Anytime Algorithm for POMDPs (2003)*.

### 4️⃣ **Distributional Point-Based Value Iteration (DPBVI)**
- **Our proposed method**, extending PBVI with **distributional RL concepts**.
- **Files:**
  - Implementation: `agents/dpbvi.py`
  - Evaluation: `minigrid_eval.py`, `trivial_example.py`
- **Key Environments:**
  - **MiniGrid-DoorKey-5x5-v0** – Adapted from [MiniGrid](https://github.com/Farama-Foundation/Minigrid).
  - **Noisy Two-State Problem** – Adapted from [Norvig & Russell's Artificial Intelligence: A Modern Approach (Section 17.5)](http://aima.cs.berkeley.edu/).

## 🏗️ Installation (Using Conda)

### **1️⃣ Clone the repository**
```
git clone https://github.com/lpreuettUW/distributional_point_based_value_iteration.git
cd distributional_point_based_value_iteration
```

### **2️⃣ Create and activate a Conda environment**
Ensure you have Conda installed, then create an environment with Python 3.11:
```
conda create --name dpbvi-env python=3.11
conda activate dpbvi-env
```

### **3️⃣ Install dependencies**
```
pip install -r requirements.txt
```

## 🚀 Running Experiments

### Run Value Iteration (VI) on Cliff Walking:
```
python vi_cliff_walking_example.py
```

### Run Distributional Value Iteration (DVI) on Cliff Walking:
```
python dvi_cliff_walking_example.py
```

### Run PBVI / DPBVI on MiniGrid:
```
python minigrid_eval.py
```

### Run PBVI / DPBVI on Two-State Problem:
```
python trivial_example.py
```

## 📜 License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
