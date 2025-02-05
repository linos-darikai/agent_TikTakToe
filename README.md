# 🎮 TicTacToe Q-Learning AI 🤖

## 🌟 Overview
This project implements a Q-learning based AI that learns to play Tic-Tac-Toe through experience. The AI improves its strategy over time by learning from the outcomes of games it plays. It's like teaching a computer to play by letting it practice thousands of times! 

## 🧠 What is Q-Learning?

Q-Learning is a model-free reinforcement learning algorithm that learns to make optimal decisions by experiencing the outcomes of its actions. Think of it like training a pet - good actions get rewards, bad actions don't! The "Q" in Q-learning stands for "Quality" - representing how valuable a given action is in a particular state.

### 🎯 Core Concepts

1. **Q-Value** 📊: A measure that represents the quality of an action in a given state
2. **State** 🎲: A specific configuration or situation in the environment
3. **Action** ⚡: A possible move or decision the agent can make
4. **Reward** 🌟: Feedback received after taking an action
5. **Learning Rate (α)** 📚: How much new information overrides old information
6. **Discount Factor (γ)** ⏳: How much future rewards are valued compared to immediate rewards

### 🔬 The Q-Learning Formula

Q(s,a) = Q(s,a) + α * (R + γ * max(Q(s',a')) - Q(s,a))

Where:
- 📌 Q(s,a): Current Q-value for state s and action a
- 📚 α: Learning rate (set to 0.1 in this implementation)
- 🎁 R: Reward received
- ⏳ γ: Discount factor (set to 0.9 in this implementation)
- 🎯 max(Q(s',a')): Maximum Q-value for the next state

## 🛠️ Implementation Details

### 🧮 State Representation
- Each board state is converted to a unique numerical value using base-3 encoding
- Empty cells are represented as 0, AI moves as 1, and player moves as 2
- This creates a unique identifier for each possible board configuration

```python
def state_to_num(state):
    return sum(state[i] * (3**i) for i in range(9))
```

### 📚 Learning Process

1. **🔍 Exploration vs Exploitation**
   - Initial exploration rate (epsilon) = 1.0
   - Trained exploration rate = 0.1
   - Higher epsilon means more random moves for exploration
   - Lower epsilon means more exploitation of learned strategies

2. **🎯 Move Selection**
   - Random move if random number < epsilon
   - Otherwise, choose move with highest Q-value

3. **🏆 Rewards Structure**
   - Win: +1.0
   - Loss: -1.0
   - Tie: +0.5

4. **🎓 Training**
   - Two AI agents play against each other
   - States are stored in state_history during gameplay
   - Q-values are updated after each game using backward updates
   - Q-table is saved to CSV file for persistence

### ✨ Key Features

1. **📈 Progressive Learning**
   - Multiple AI models can be trained
   - Models automatically switch after MAX_LOSSES (7) consecutive losses
   - Each model maintains its own Q-table

2. **💾 Persistence**
   - Q-tables are saved to CSV files
   - Separate Q-table for each model number
   - Automatic loading/saving of Q-tables

3. **📊 Training Interface**
   - Progress bar shows training status
   - Training statistics displayed after completion
   - Option to train new models through UI

4. **🎮 Game Interface**
   - Graphical interface using tkinter
   - Status display showing current model and losses
   - Menu options for new game, training, and exit

## 🚀 Usage

1. 🆕 First run will prompt to train initial AI model
2. 🎮 Play against the AI by clicking empty cells
3. 🤖 AI automatically moves after player
4. 🎓 Train new models through Game menu
5. 💾 Models automatically save on exit

The AI will continuously learn from games played against human players, though at a slower rate than during training due to the lower exploration rate (0.1 vs 1.0 during training).

## 🤝 Contributing

Feel free to fork, submit PRs, or open issues! We love community contributions! 

## ⭐ Star Us!
If you find this project helpful, please give it a star! It helps others discover this project.

## 📜 License
This project is open source and available under the MIT License.
