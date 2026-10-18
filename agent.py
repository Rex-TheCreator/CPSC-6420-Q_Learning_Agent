try:
    import numpy as np
    import gymnasium as gym
    import matplotlib.pyplot as plt
    from pathlib import Path
except ImportError:
    print("\nA dependency was not installed. Please install Gymnasium, NumPy, and Matplotlib using 'pip install numpy matplotlib \"gymnasium[toy-text]\"'.\n")
    raise SystemExit(1)

epsilon = 1 # Exploration rate
gamma = 1 # Discount factor
alpha = 0.1 # Learning rate
episodes = 5000 # Number of episodes for training
print(f"Agent is using Exploration rate (epsilon): {epsilon} with decay, Discount factor (gamma): {gamma}, Learning rate (alpha): {alpha} for {episodes} episodes.")

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

def choose_action(state, updated_epsilon):
    if np.random.uniform(0, 1) < updated_epsilon:
        action = env.action_space.sample() # Explore: choose a random action
    else:
        action = np.argmax(q_table[state, :]) # Exploit: choose the action with the highest Q-value for the current state
        if np.where(q_table[state, :] == np.max(q_table[state, :]))[0].size > 1: # Check for ties in Q-values
            action = np.random.choice(np.where(q_table[state, :] == np.max(q_table[state, :]))[0]) # Randomly choose among the tied actions
    return action

def print_state_info(episode, reward, terminated):
    if terminated and reward == 1:
        print(f"Episode {episode + 1} terminated: Agent reached the goal!")

episode_rewards = [] # store the reward for each episode
for episode in range(episodes):
    state, info = env.reset()  # Reset the environment for each episode
    episode_reward = 0

    while True:
        action = choose_action(state, epsilon * (episodes - episode) / episodes) # Choose an action based on the current state
        next_state, reward, terminated, truncated, info = env.step(action) # Take a step in the environment using the chosen action
        episode_reward += reward
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
            episode_rewards.append(episode_reward)
            break
        
    if (episode + 1) % episodes == 0: # Print the Q-table every 1000 episodes
        print(f"\n final Q-table:")
        print(q_table) 
        
evaluate_episodes = 500 # Number of episodes for evaluation
print(f"\nEvaluating the agent's performance over {evaluate_episodes} episodes...")
evaluation_rewards = [] # store the reward for each evaluation episode
episode_steps = [] # store the number of steps taken in each evaluation episode
for episode in range(evaluate_episodes):
    state, info = env.reset()  # Reset the environment for each evaluation episode
    episode_reward = 0
    step_count = 0
    while True:
        action = np.argmax(q_table[state, :]) # Choose the best action based on the learned Q-table
        next_state, reward, terminated, truncated, info = env.step(action) # Take a step in the environment using the chosen action
        episode_reward += reward
        step_count += 1
        state = next_state # Move to the next state
        if terminated or truncated: # Check if the episode has ended
            evaluation_rewards.append(episode_reward)
            episode_steps.append(step_count)
            break
print(f"Average reward over {episodes} training episodes: {np.mean(episode_rewards)}")
print(f"Average reward over {evaluate_episodes} evaluation episodes: {np.mean(evaluation_rewards)}")
print(f"Average steps over {evaluate_episodes} evaluation episodes: {np.mean(episode_steps)}")

env.close()  # Close the environment after all episodes are completed

episode_numbers = range(1, len(episode_rewards) + 1)  # Number the recorded episodes starting at 1.

Path("results").mkdir(exist_ok=True)  # Create results/ if that folder does not already exist.

plt.figure(figsize=(10, 4))  # Create a figure measuring 10 by 4 inches.
plt.plot(episode_numbers, episode_rewards, ".", markersize=2, label="Episode reward")  # Plot each episode as a dot.

window = 100  # Average the results of 100 consecutive episodes at a time.
moving_average = []  # Store one average for each complete window.

for end in range(window, len(episode_rewards) + 1):  # Move the window's ending position forward.
    recent_rewards = episode_rewards[end - window:end]  # Extract the preceding 100 episode rewards.
    moving_average.append(sum(recent_rewards) / window)  # Calculate and store their mean.

average_episodes = range(window, len(episode_rewards) + 1)  # Align each average with its window's final episode.
plt.plot(average_episodes, moving_average, label="100-episode moving average")  # Overlay the learning trend.

plt.xlabel("Training episode")  # x-axis
plt.ylabel("Total episode reward")  # y-axis
plt.title("Frozen Lake 8x8: Baseline")  # title
plt.ylim(-0.05, 1.05)  # Leave a little visual space around rewards of 0 and 1.
plt.legend()  # display legend
plt.tight_layout()  # Adjust spacing so labels fit within the figure.
plt.savefig("results/baseline.png", dpi=300)  # Save a high-resolution image for your report.
plt.show()  # Open the plot for viewing.