import numpy as np
import random
import os
import csv
import tkinter as tk
from tkinter import messagebox, ttk
import atexit

BASE_Q_TABLE_FILE = "q_table_{}.csv"
INITIAL_EPSILON = 1.0
TRAINED_EPSILON = 0.1
MAX_LOSSES = 7
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9

class TicTacToeAI:
    def __init__(self, name="AI", model_num=1, epsilon=TRAINED_EPSILON):
        self.name = name
        self.model_num = model_num
        self.epsilon = epsilon
        self.values = {}
        self.losses = 0
        self.state_history = []  # Track states for learning
        self.load_q_table()
    
    def load_q_table(self):
        q_table_file = BASE_Q_TABLE_FILE.format(self.model_num)
        if os.path.exists(q_table_file):
            with open(q_table_file, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    self.values[int(row[0])] = float(row[1])
            print(f"Loaded {len(self.values)} states from Q-table {self.model_num}")
    
    def save_q_table(self):
        q_table_file = BASE_Q_TABLE_FILE.format(self.model_num)
        with open(q_table_file, "w", newline='') as f:
            writer = csv.writer(f)
            for key, value in self.values.items():
                writer.writerow([key, value])
        print(f"Saved {len(self.values)} states to Q-table {self.model_num}")
    
    def learn_from_game(self, reward):
        """Update Q-values based on the game outcome"""
        for state in reversed(self.state_history):
            self.values[state] = self.values.get(state, 0) + LEARNING_RATE * (reward - self.values.get(state, 0))
            reward *= DISCOUNT_FACTOR
        self.state_history = []  # Clear history after learning
        self.save_q_table()  # Save updated values

# [Previous functions remain the same: state_to_num, check_for_winner]
def state_to_num(state):
    return sum(state[i] * (3**i) for i in range(9))

def check_for_winner(board):
    wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != 0:
            return board[a]
    if all(board):
        return 'Tie'
    return None

def get_ai_move(ai, board):
    possible_moves = [i for i, spot in enumerate(board) if spot == 0]
    
    if np.random.rand() < ai.epsilon:
        return random.choice(possible_moves)
    
    best_value = -np.inf
    best_move = None
    for move in possible_moves:
        temp_state = board[:]
        temp_state[move] = 1
        state_num = state_to_num(temp_state)
        value = ai.values.get(state_num, 0)
        if value > best_value:
            best_value = value
            best_move = move
    return best_move if best_move is not None else random.choice(possible_moves)


def train_ai(model_num, episodes=100000):
    ai1 = TicTacToeAI("AI1", model_num, INITIAL_EPSILON)
    ai2 = TicTacToeAI("AI2", model_num, INITIAL_EPSILON)
    
    progress_window = tk.Toplevel()
    progress_window.title(f"Training AI Model {model_num}")
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(progress_window, length=300, variable=progress_var, mode='determinate')
    progress_bar.pack(pady=10)
    status_label = tk.Label(progress_window, text="Training in progress...")
    status_label.pack(pady=5)
    progress_window.update()
    
    win_count = {1: 0, 2: 0, 'Tie': 0}
    
    for episode in range(episodes):
        board = [0] * 9
        state_history = []
        current_ai = ai1 if random.random() < 0.7 else ai2
        
        while True:
            move = get_ai_move(current_ai, board)
            board[move] = 1 if current_ai == ai1 else 2
            state_num = state_to_num(board)
            state_history.append(state_num)
            
            winner = check_for_winner(board)
            if winner:
                reward = 1 if winner == (1 if current_ai == ai1 else 2) else -1
                if winner == 'Tie':
                    reward = 0.5
                
                for state in reversed(state_history):
                    current_ai.values[state] = current_ai.values.get(state, 0) + 0.1 * (reward - current_ai.values.get(state, 0))
                    reward *= 0.9
                
                win_count[winner] += 1
                break
                
            current_ai = ai2 if current_ai == ai1 else ai1
        
        progress_var.set((episode + 1) / episodes * 100)
        if (episode + 1) % 100 == 0:
            status_label.config(text=f"Episode {episode + 1}/{episodes}")
        progress_window.update()
    
    if win_count[1] >= win_count[2]:
        ai1.epsilon = TRAINED_EPSILON
        ai1.save_q_table()
        best_ai = ai1
    else:
        ai2.epsilon = TRAINED_EPSILON
        ai2.save_q_table()
        best_ai = ai2
    
    progress_window.destroy()
    messagebox.showinfo("Training Complete", 
                       f"Training Model {model_num} finished!\nWins: AI1: {win_count[1]}, AI2: {win_count[2]}, Ties: {win_count['Tie']}")
    return best_ai
   

class TicTacToeGame:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Tic-Tac-Toe vs Learning AI")
        
        self.current_model = 1
        self.ai = TicTacToeAI(model_num=self.current_model)
        self.board = [0] * 9
        self.buttons = []
        
        # Register save function on exit
        atexit.register(self.save_on_exit)
        
        # Create menu
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.reset_game)
        game_menu.add_command(label="Train New Model", command=self.train_new_model)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.quit_game)
        
        # Create status frame
        self.status_frame = tk.Frame(self.root)
        self.status_frame.grid(row=3, columnspan=3, pady=10)
        self.status_label = tk.Label(self.status_frame, 
                                   text=f"Current AI: Model {self.current_model} (Losses: {self.ai.losses})")
        self.status_label.pack()
        
        # Create game board
        self.create_board()
    
    def create_board(self):
        for i in range(9):
            btn = tk.Button(self.root, text='', font=('Arial', 24), width=5, height=2,
                          command=lambda i=i: self.make_move(i))
            btn.grid(row=i//3, column=i%3)
            self.buttons.append(btn)
    
    def make_move(self, index):
        if self.board[index] == 0:
            # Human move
            self.board[index] = 2
            self.buttons[index].config(text='O')
            
            # Record state after human move
            state_num = state_to_num(self.board)
            self.ai.state_history.append(state_num)
            
            winner = check_for_winner(self.board)
            if winner:
                self.end_game(winner)
                return
            
            # AI move
            ai_move = get_ai_move(self.ai, self.board)
            self.board[ai_move] = 1
            self.buttons[ai_move].config(text='X')
            
            # Record state after AI move
            state_num = state_to_num(self.board)
            self.ai.state_history.append(state_num)
            
            winner = check_for_winner(self.board)
            if winner:
                self.end_game(winner)
    
    def end_game(self, winner):
        msg = "It's a tie!" if winner == 'Tie' else f"{'AI' if winner == 1 else 'Human'} wins!"
        messagebox.showinfo("Game Over", msg)
        
        """# Update AI's learning based on game outcome
        if winner == 1:  # AI wins
            self.ai.learn_from_game(1.0)
        elif winner == 2:  # Human wins
            self.ai.learn_from_game(-1.0)
            self.ai.losses += 1
            if self.ai.losses >= MAX_LOSSES:
                self.switch_ai_model()
                self.ai.losses = 0
        else:  # Tie
            self.ai.learn_from_game(0.5)"""
        
        self.status_label.config(text=f"Current AI: Model {self.current_model} (Losses: {self.ai.losses})")
        self.reset_game()
    
    def save_on_exit(self):
        """Save the Q-table when the game exits"""
        if hasattr(self, 'ai'):
            self.ai.save_q_table()
    
    def quit_game(self):
        """Properly handle game exit"""
        self.save_on_exit()
        self.root.quit()
    def switch_ai_model(self):
        # Find next available model
        next_model = self.current_model + 1
        while not os.path.exists(BASE_Q_TABLE_FILE.format(next_model)):
            next_model += 1
            if next_model > self.current_model + 3:  # If no models found within next 3 numbers
                next_model = 1  # Wrap around to first model
                break
        
        self.current_model = next_model
        self.ai = TicTacToeAI(model_num=self.current_model)
        print(f"Switched to AI Model {self.current_model}")
    
    def reset_game(self):
        self.board = [0] * 9
        for button in self.buttons:
            button.config(text='')
        
        
        ai_move = get_ai_move(self.ai, self.board)
        self.board[ai_move] = 1
        self.buttons[ai_move].config(text='X')
        
        # Record initial state
        state_num = state_to_num(self.board)
        self.ai.state_history.append(state_num)
    
    def train_new_model(self):
        model_num = 1
        while os.path.exists(BASE_Q_TABLE_FILE.format(model_num)):
            model_num += 1
        
        self.ai = train_ai(model_num)
        self.current_model = model_num
        self.ai.losses = 0
        self.status_label.config(text=f"Current AI: Model {self.current_model} (Losses: {self.ai.losses})")
        self.reset_game()
    
    def run(self):
        if not any(os.path.exists(BASE_Q_TABLE_FILE.format(i)) for i in range(1, 4)):
            response = messagebox.askyesno("First Run", 
                "No trained AI models found. Would you like to train the first AI model?")
            if response:
                self.train_new_model()
            else:
                self.reset_game()
        else:
            self.reset_game()
        
        self.root.mainloop()
    
    

if __name__ == "__main__":
    game = TicTacToeGame()
    game.run()