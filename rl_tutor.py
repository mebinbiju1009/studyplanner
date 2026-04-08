import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

MODEL_PATH = "ppo_tutor_model"

class SimulatedStudentEnv(gym.Env):
    """
    A custom environment simulating a student's cognitive state.
    Observation comprises:
        [typing_delay, response_length, fatigue]
    Action comprises:
        0: Analogy
        1: Socratic Question
        2: Break
    """
    def __init__(self):
        super(SimulatedStudentEnv, self).__init__()
        # Actions: 0, 1, 2
        self.action_space = spaces.Discrete(3)
        # Observations: 
        # typing_delay (0.0 to 100.0) - in seconds
        # response_length (0.0 to 500.0) - in words
        # fatigue (0.0 to 1.0) - percentage
        low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([100.0, 500.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.state = None
        self.current_step = 0
        self.max_steps = 20

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Start with low delay, decent length, low fatigue
        self.state = np.array([5.0, 20.0, 0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        
        typing_delay, response_length, fatigue = self.state
        reward = 0.0
        
        # Simulate student dynamics based on the tutor's action
        if action == 0:  # Analogy
            # Usually helps understanding, reduces delay, slightly increases length, minor fatigue
            typing_delay = max(2.0, typing_delay - 2.0)
            response_length += 5.0
            fatigue += 0.05
            reward = 1.0
        elif action == 1:  # Socratic Question
            # Forces thinking, increases delay, drastically increases length, moderate fatigue
            typing_delay += 3.0
            response_length += 15.0
            fatigue += 0.1
            reward = 2.0 if fatigue < 0.6 else -2.0 # Good if not fatigued, bad if fatigued
        elif action == 2:  # Break
            # Resets fatigue, minimal response expected
            fatigue = max(0.0, fatigue - 0.4)
            response_length = 5.0
            typing_delay = 2.0
            reward = 3.0 if fatigue > 0.6 else -1.0 # Good if over-fatigued, bad if done too early
            
        # Natural drift
        fatigue += 0.02
        
        # Keep bounds
        typing_delay = np.clip(typing_delay, 0.0, 100.0)
        response_length = np.clip(response_length, 0.0, 500.0)
        fatigue = np.clip(fatigue, 0.0, 1.0)
        
        self.state = np.array([typing_delay, response_length, fatigue], dtype=np.float32)
        
        # End episode if max steps reached or fatigue extremely high
        terminated = bool(self.current_step >= self.max_steps or fatigue >= 0.95)
        
        # Penalty for episode ending due to extreme fatigue
        if fatigue >= 0.95:
            reward -= 5.0
            
        return self.state, float(reward), terminated, False, {}

def train_model(timesteps=10000):
    """Trains the PPO model on the simulated student environment."""
    env = SimulatedStudentEnv()
    model = PPO("MlpPolicy", env, verbose=0)
    print(f"Training PPO Tutor model for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATH)
    print("Model saved to", MODEL_PATH)

def get_teaching_strategy(typing_delay, response_length, fatigue):
    """
    Given observed metrics, predict the best teaching action.
    Loads the trained model if available, otherwise predicts randomly.
    Returns: strategy_index (0, 1, or 2), strategy_name
    """
    if not os.path.exists(f"{MODEL_PATH}.zip"):
        # If not trained, fallback to static heuristic.
        # But we will train it before usage.
        print("Warning: Model not found. Returning heuristic.")
        if fatigue > 0.7:
            action = 2
        elif typing_delay > 10.0:
            action = 0
        else:
            action = 1
    else:
        model = PPO.load(MODEL_PATH)
        obs = np.array([float(typing_delay), float(response_length), float(fatigue)], dtype=np.float32)
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)
        
    strategies = {
        0: "Analogy",
        1: "Socratic Question",
        2: "Break"
    }
    
    return action, strategies.get(action, "Analogy")

if __name__ == "__main__":
    # If run standalone, train the model
    train_model(20000)
