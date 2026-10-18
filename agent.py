try:
    import numpy as np
except ImportError:
    print("\nNumPy not installed. Please install it using 'pip install numpy'.\n")
    raise SystemExit(1)

try:
    import gymnasium as gym
except ImportError:
    print("\nGym not installed. Please install it using 'pip install \"gymnasium[toy-text]\"'.\n")
    raise SystemExit(1)

epsilon = 0.7 # Exploration rate
gamma = 1 # Discount factor
alpha = 0.1 # Learning rate
episodes = 10000 # Number of episodes for training
print(f"Agent is using Exploration rate (epsilon): {epsilon}, Discount factor (gamma): {gamma}, Learning rate (alpha): {alpha} for {episodes} episodes.")

env = gym.make(
    'FrozenLake-v1',
    map_name='8x8',
    is_slippery=True,
    render_mode='ansi',
)
state, info = env.reset(seed=67)  # Reset the environment and set a seed for reproducibility
np.random.seed(67)  
env.action_space.seed(67)  

num_states = env.observation_space.n # number of table rows for the Q-table
num_actions = env.action_space.n # number of table columns for the Q-table
q_table = np.zeros((num_states, num_actions)) # initialize the Q-table with zeros
print("Q-table initialized with shape:", q_table.shape)
np.set_printoptions(precision=4, suppress=True) # Set NumPy print options for better readability

def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample() # Explore: choose a random action
    else:
        action = np.argmax(q_table[state, :]) # Exploit: choose the action with the highest Q-value for the current state
        if np.where(q_table[state, :] == np.max(q_table[state, :]))[0].size > 1: # Check for ties in Q-values
            action = np.random.choice(np.where(q_table[state, :] == np.max(q_table[state, :]))[0]) # Randomly choose among the tied actions
    return action

def print_state_info(episode, reward, terminated):
    if terminated and reward == 1:
        print(f"Episode {episode + 1} terminated: Agent reached the goal!")

for episode in range(episodes):
    state, info = env.reset()  # Reset the environment for each episode

    while True:
        action = choose_action(state) # Choose an action based on the current state
        next_state, reward, terminated, truncated, info = env.step(action) # Take a step in the environment using the chosen action
        if terminated and reward == 1:
            print_state_info(episode, reward, terminated) # Print the state when agent reaches the goal

        # Update Q-table using the Q-learning formula
        target = reward
        if not terminated:
            target += gamma * np.max(q_table[next_state, :]) # Add the discounted maximum future reward if not terminated
        prev_q_value = q_table[state, action] # Store the previous Q-value for comparison
        q_table[state, action] = prev_q_value + alpha * (target - prev_q_value) # Update the Q-value for the current state-action pair

        state = next_state # Move to the next state

        if terminated or truncated: # Check if the episode has ended
            break
        
    if (episode + 1) % 1000 == 0: # Print the Q-table every 1000 episodes
        print(f"\nQ-table after episode {episode + 1}:")
        print(q_table) 
env.close()  # Close the environment after all episodes are completed