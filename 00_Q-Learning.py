import pygame  
import random  
import numpy as np  

# Initialize pygame
pygame.init()

# Window and grid configuration
WIDTH = 300
HEIGHT = 300
GRID_SIZE = 5                   # Number of cells per row/column
CELL_SIZE = WIDTH // GRID_SIZE  # Size of each cell in pixels
FPS = 5                         # Frames per second (controls snake speed)

# Colors (RGB format)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Snake movement actions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
ACTIONS = [UP, DOWN, LEFT, RIGHT]


# Create game window
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Q-learning - Small Grid")
font = pygame.font.SysFont(None, 24)


# ------------------------------
# Food class
# ------------------------------
class Food:
    def __init__(self):
        # Randomly place food inside the grid
        self.position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

    def randomize(self):
        # Move food to a new random position
        self.position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        
        

# ------------------------------
# Snake class
# ------------------------------
class Snake:
    def __init__(self):
        self.positions = [(2, 2)]          # Start with snake at center
        self.direction = random.choice(ACTIONS)  # Random initial direction

    def move(self, action):
        # Move snake by updating head and shifting body
        self.direction = action
        head_x, head_y = self.positions[0]
        dir_x, dir_y = self.direction
        new_head = (head_x + dir_x, head_y + dir_y)
        self.positions = [new_head] + self.positions[:-1]  # Update positions list

    def grow(self):
        # Add a new segment to snake at the tail
        tail = self.positions[-1]
        self.positions.append(tail)

    def collision(self):
        # Check for collision with itself or walls
        head = self.positions[0]
        return (
            head in self.positions[1:] or  # Hits itself
            head[0] < 0 or head[0] >= GRID_SIZE or  # Hits wall (X-axis)
            head[1] < 0 or head[1] >= GRID_SIZE     # Hits wall (Y-axis)
        )
        
        

# ------------------------------
# Q-learning Setup
# ------------------------------
q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  
# Q-table shape: [x_position, y_position, action_index]

# Hyperparameters
alpha = 0.1   # Learning rate (how much new info overrides old)
gamma = 0.9   # Discount factor (importance of future rewards)
epsilon = 0.2 # Exploration rate (probability of random action)
num_episodes = 50  # Training episodes



# ------------------------------
# Helper functions
# ------------------------------
def get_state(snake):
    # State is defined as the snake's head position
    return snake.positions[0]


def get_reward(snake, food):
    # Reward system:
    if snake.positions[0] == food.position:   # Eats food
        return 10
    elif snake.collision():                  # Hits wall or itself
        return -100
    else:                                    # Every other move
        return -1
        
        
        
# ------------------------------
# Main Training Loop
# ------------------------------
clock = pygame.time.Clock()

for episode in range(1, num_episodes + 1): 
    snake = Snake() 
    food = Food()
    done = False 
    total_reward = 0 

    while not done:  
        clock.tick(FPS)  # Control game speed

        # Quit game if user closes window
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        # Get current state (snake head position)
        current_state = get_state(snake) 
        
        # ε-greedy strategy: explore or exploit
        if random.uniform(0, 1) < epsilon: 
            action_idx = random.randint(0, 3)   # Random action (explore)
        else:
            action_idx = np.argmax(q_table[current_state[0], current_state[1]])  # Best action (exploit)
            
        # Perform action
        action = ACTIONS[action_idx] 
        snake.move(action) 

        # Compute reward
        reward = get_reward(snake, food)  
        total_reward += reward 

        # Get next state
        next_state = get_state(snake) 

        # Update Q-table (if next state is valid)
        if 0 <= next_state[0] < GRID_SIZE and 0 <= next_state[1] < GRID_SIZE:
            q_table[current_state[0], current_state[1], action_idx] = (
                q_table[current_state[0], current_state[1], action_idx] +
                alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) -
                         q_table[current_state[0], current_state[1], action_idx])
            )

        # If food eaten → grow snake and randomize food position
        if snake.positions[0] == food.position: 
            snake.grow() 
            food.randomize() 

        # ---------------- Drawing ----------------
        win.fill(WHITE)  # Clear screen
        
        # Draw grid lines
        for i in range(GRID_SIZE): 
            for j in range(GRID_SIZE):
                pygame.draw.rect(win, BLACK, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        # Draw snake
        for pos in snake.positions:
            pygame.draw.rect(win, GREEN, (pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            label = font.render("Snake", True, BLACK)
            win.blit(label, (pos[0] * CELL_SIZE + 5, pos[1] * CELL_SIZE + 5))

        # Draw food
        pygame.draw.rect(win, RED, (food.position[0] * CELL_SIZE, food.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        label = font.render("Food", True, BLACK)
        win.blit(label, (food.position[0] * CELL_SIZE + 5, food.position[1] * CELL_SIZE + 5))

        # Update window
        pygame.display.update()

        # If collision → end episode
        if snake.collision(): 
            done = True

    # Print result for this episode
    print(f"Episode {episode} finished with Total Reward: {total_reward}")
