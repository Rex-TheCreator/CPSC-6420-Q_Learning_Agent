try:
    import gymnasium as gym
except ImportError:
    print("\nGym not installed. Please install it using 'pip install \"gymnasium[toy-text]\"'.\n")
    

env = gym.make(
    'FrozenLake-v1',
    map_name='8x8',
    is_slippery=True,
    render_mode='ansi',    
)

state, info = env.reset(seed=67)  # Reset the environment and set a seed for reproducibility

print("Frozen Lake number of states:", env.observation_space.n)
print("Frozen Lake number of actions:", env.action_space.n)
print("Initial state:", state)
print(env.render())  # Render the initial state of the environment

action = 2 # set an action (0: left, 1: down, 2: right, 3: up)
next_state, reward, terminated, truncated, info = env.step(action)  # Take step using action

print("Next state:", next_state)
print("Reward received:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print(env.render())  # Render the state after taking the action

env.close()  # Close the environment