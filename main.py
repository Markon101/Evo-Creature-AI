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
    def __init__(self, start_pos, length, thickness, angle=0.0, mass=1.0):
        self.start_pos = start_pos
        self.length = length
        self.thickness = thickness
        self.angle = angle
        self.mass = mass

        self.end_pos = (
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        )

    def update(self):
        self.end_pos = (
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        )

    def draw(self, screen):
        pygame.draw.line(screen, WHITE, self.start_pos, self.end_pos, self.thickness)

class Joint:
    def __init__(self, bone1, bone2, joint_type="hinge", angle_limit=(-math.pi, math.pi)):
        self.bone1 = bone1
        self.bone2 = bone2
        self.joint_type = joint_type
        self.angle_limit = angle_limit

class Muscle:
    def __init__(self, bone1, bone2, strength=1.0):
        self.bone1 = bone1
        self.bone2 = bone2
        self.strength = strength
        self.activation = 0.0

    def contract(self, activation):
        self.activation = max(0.0, min(1.0, activation))

# --- Creature ---
class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bones = []
        self.joints = []
        self.muscles = []
        self.alive = True

    def create_simple_creature(self):
        bone1 = Bone((self.x, self.y), 30, 3)
        bone2 = Bone(bone1.end_pos, 30, 3)
        self.bones.extend([bone1, bone2])

        joint1 = Joint(bone1, bone2)
        self.joints.append(joint1)

        muscle1 = Muscle(bone1, bone2, strength=0.1)
        self.muscles.append(muscle1)

    def update(self, actions):
        for muscle, action in zip(self.muscles, actions):
            muscle.contract(action)
        self.apply_physics()

    def apply_physics(self):
        for bone in self.bones:
            bone.start_pos = (bone.start_pos[0], bone.start_pos[1] + 0.2)
            bone.update()

            if bone.start_pos[1] > GROUND_Y:
                bone.start_pos = (bone.start_pos[0], GROUND_Y)
            bone.update()

    def draw(self, screen):
        for bone in self.bones:
            bone.draw(screen)

    def get_state(self):
        return [
            self.bones[0].angle,
            0.0,  
            self.y,
            0.0 
        ]

# --- Neural Network ---
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        return out

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64

        self.q_network = NeuralNetwork(state_size, 128, action_size)
        self.target_network = NeuralNetwork(state_size, 128, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q_values = rewards + (self.gamma * next_q_values * (~dones).float())

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# --- Environment ---
class Environment:
    def __init__(self):
        self.creatures = [Creature(WIDTH // 4, GROUND_Y - 60)]
        self.creatures[0].create_simple_creature()

    def step(self, actions):
        for creature, action in zip(self.creatures, actions):
            muscle_activations = [0.0, 0.0]
            muscle_activations[action] = 1.0
            creature.update(muscle_activations)

        # Return some placeholder values for now 
        observations = [0] 
        reward = 0
        done = False
        info = {} 
        return observations, reward, done, info

    def render(self, screen):
        screen.fill(BLACK)
        pygame.draw.line(screen, WHITE, (0, GROUND_Y), (WIDTH, GROUND_Y), 2)

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
