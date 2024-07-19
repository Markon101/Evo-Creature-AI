import pygame
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- Constants ---
WIDTH, HEIGHT = 1200, 600
GROUND_Y = 500
FPS = 60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EvoPy")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# --- Creature Components ---
class Bone:
    """
    Represents a single bone in the creature's structure.
    """
    def __init__(self, start_pos, length, thickness, angle=0.0, mass=1.0):
        """
        Initializes a Bone object.

        Args:
            start_pos (tuple): (x, y) coordinates of the starting point.
            length (float): Length of the bone.
            thickness (int): Thickness for drawing the bone.
            angle (float, optional): Angle in radians relative to horizontal. Defaults to 0.0.
            mass (float, optional): Mass of the bone (for future physics). Defaults to 1.0.
        """
        self.start_pos = start_pos
        self.length = length
        self.thickness = thickness
        self.angle = angle
        self.mass = mass

        # Calculate end position based on length and angle
        self.end_pos = (
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        )

    def update(self):
        """
        Updates the end position of the bone based on its current angle and start position.
        Called after any changes to angle or start_pos. 
        """
        self.end_pos = (
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        )

    def draw(self, screen):
        """
        Draws the bone on the Pygame screen.
        """
        pygame.draw.line(screen, WHITE, self.start_pos, self.end_pos, self.thickness)

class Joint:
    """
    Connects two bones together to form a joint.
    """
    def __init__(self, bone1, bone2, joint_type="hinge", angle_limit=(-math.pi, math.pi)):
        """
        Initializes a Joint object.

        Args:
            bone1 (Bone): The first bone connected to the joint.
            bone2 (Bone): The second bone connected to the joint.
            joint_type (str, optional): Type of joint (e.g., "hinge"). Defaults to "hinge".
            angle_limit (tuple, optional): (min_angle, max_angle) limits in radians. 
                                            Defaults to (-pi, pi).
        """
        self.bone1 = bone1
        self.bone2 = bone2
        self.joint_type = joint_type
        self.angle_limit = angle_limit

class Muscle:
    """
    Represents a muscle that can apply forces to bones.
    """
    def __init__(self, bone1, bone2, strength=1.0):
        """
        Initializes a Muscle.

        Args:
            bone1 (Bone): The bone where the muscle starts.
            bone2 (Bone): The bone where the muscle ends.
            strength (float, optional): The force the muscle can exert. Defaults to 1.0.
        """
        self.bone1 = bone1
        self.bone2 = bone2
        self.strength = strength
        self.activation = 0.0  # 0.0 (relaxed) to 1.0 (fully contracted)

    def contract(self, activation):
        """
        Sets the activation level of the muscle (how contracted it is).

        Args:
            activation (float): Activation level (0.0 to 1.0).
        """
        self.activation = max(0.0, min(1.0, activation))  # Clamp between 0 and 1


# --- Creature ---
class Creature:
    """
    Represents a creature made of bones, joints, and muscles.
    """
    def __init__(self, x, y):
        """
        Initializes a Creature.

        Args:
            x (float): Initial x-coordinate of the creature.
            y (float): Initial y-coordinate of the creature.
        """
        self.x = x
        self.y = y
        self.bones = []
        self.joints = []
        self.muscles = []
        self.alive = True

    def create_two_legged_creature(self):
        """
        Creates a simple two-legged creature structure.
        """
        # Body (starts vertically)
        body = Bone((self.x, self.y), 40, 3, angle=math.pi / 2) 

        # Legs
        leg1 = Bone(body.end_pos, 30, 3)
        leg2 = Bone(body.end_pos, 30, 3)

        self.bones.extend([body, leg1, leg2])

        # Joints (with angle limits for the legs)
        joint1 = Joint(body, leg1, angle_limit=(-math.pi/4, math.pi/4))  
        joint2 = Joint(body, leg2, angle_limit=(-math.pi/4, math.pi/4)) 
        self.joints.extend([joint1, joint2])

        # Muscles
        muscle1 = Muscle(body, leg1, strength=0.05)
        muscle2 = Muscle(body, leg2, strength=0.05)
        self.muscles.extend([muscle1, muscle2])

    def update(self, actions):
        """
        Updates the creature's state based on muscle activations and applies physics.

        Args:
            actions (list): A list of muscle activations (0.0 to 1.0) for each muscle.
        """
        for muscle, action in zip(self.muscles, actions):
            muscle.contract(action)

        self.apply_physics()

    def apply_physics(self):
        """
        Applies basic physics (gravity and ground collision) to the creature.
        """
        for bone in self.bones:
            bone.start_pos = (bone.start_pos[0], bone.start_pos[1] + 0.2)  # Gravity
            bone.update()

            if bone.start_pos[1] > GROUND_Y:
                bone.start_pos = (bone.start_pos[0], GROUND_Y)
            bone.update()

    def draw(self, screen):
        """
        Draws the creature on the Pygame screen.
        """
        for bone in self.bones:
            bone.draw(screen)

    def get_state(self):
        """
        Returns a representation of the creature's current state. 

        Returns:
            list: A list of state variables (example provided).
        """
        return [
            self.bones[0].angle,
            0.0,  # Placeholder for angular velocity
            self.y,
            0.0   # Placeholder for horizontal velocity 
        ]

# --- Neural Network ---
class NeuralNetwork(nn.Module):
    """
    Neural network for the DQN agent.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of output actions.
        """
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# --- DQN Agent ---
class DQNAgent:
    """
    DQN agent that interacts with and learns from the environment.
    """
    def __init__(self, state_size, action_size):
        """
        Initializes the DQN agent.

        Args:
            state_size (int): Dimensionality of the state space.
            action_size (int): Number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000) # Experience replay buffer
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.q_network = NeuralNetwork(state_size, 128, action_size)  # Q-Network
        self.target_network = NeuralNetwork(state_size, 128, action_size)  # Target network 
        self.target_network.load_state_dict(self.q_network.state_dict()) # Initially same weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores an experience tuple in the replay memory.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: True if episode ended, False otherwise.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Chooses an action using an epsilon-greedy policy.

        Args:
            state: The current state.

        Returns:
            int: The chosen action index.
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)  # Explore
        else:
            with torch.no_grad(): # No need to track gradients for action selection
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                # Get the action index correctly
                action = torch.argmax(q_values).item()  
                return action  # Exploit

    def train(self):
        """
        Samples from memory and performs a training step to update the Q-network.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to PyTorch tensors for batch processing
        states = torch.tensor(states, dtype=torch.float32).view(self.batch_size, -1)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32).view(self.batch_size, -1) # Reshaped here
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        # Q-Network Predictions
        q_values = self.q_network(states).gather(1, actions)

        # Target Network Calculations
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones).float()) 

        # Loss Calculation and Optimization 
        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decrease exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        """
        Updates the target network with the weights of the Q-network (done periodically).
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

# --- Environment ---
class Environment:
    """
    Simulates the environment where creatures interact.
    """
    def __init__(self):
        """
        Initializes the environment with a creature.
        """
        self.creatures = [Creature(WIDTH // 4, GROUND_Y - 60)]
        self.creatures[0].create_two_legged_creature()

    def step(self, actions):
        """
        Advances the environment by one time step.

        Args:
            actions (list): List of actions to apply to each creature.

        Returns:
            tuple: (observations, rewards, dones, info)
                   - observations: List of observations for each creature.
                   - rewards: List of rewards received by each creature.
                   - dones: List of booleans indicating if the episode is done for each creature.
                   - info: A dictionary for additional information.
        """
        for creature, action in zip(self.creatures, actions):
            muscle_activations = [0.0, 0.0]
            muscle_activations[action] = 1.0  
            creature.update(muscle_activations)

        # Placeholder return values - to be implemented
        observations = [0]
        reward = 0
        done = False
        info = {}
        return observations, reward, done, info

    def render(self, screen):
        """
        Renders the environment on the Pygame screen.

        Args:
            screen: The Pygame display surface.
        """
        screen.fill(BLACK)  # Clear screen
        pygame.draw.line(screen, WHITE, (0, GROUND_Y), (WIDTH, GROUND_Y), 2) # Draw ground

        for creature in self.creatures:
            creature.draw(screen)

# --- Main Game Loop ---
env = Environment()
agent = DQNAgent(4, 2)  

running = True
episode_count = 0
while running:
    # --- Event Handling ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Game Logic ---
    state = env.creatures[0].get_state()
    action = agent.act(state)

    next_state, reward, done, _ = env.step([action])

    agent.remember(state, action, reward, next_state, done)
    agent.train()

    # --- Rendering ---
    env.render(screen)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
