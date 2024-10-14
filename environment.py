import pygame
import random
import math
from creature import Creature

class Environment:
    def __init__(self, width, height, ground_y):
        self.width = width
        self.height = height
        self.ground_y = ground_y
        self.creatures = []
        self.generate_random_terrain()
        self.time_step = 0
        self.max_steps = 1000  # Maximum steps per episode

    def generate_flat_terrain(self):
        self.terrain = [(0, self.ground_y), (self.width, self.ground_y)]

    def generate_random_terrain(self):
        terrain = [(0, self.ground_y)]
        x = 0
        while x < self.width:
            remaining_width = self.width - x
            if remaining_width < 50:
                # If the remaining width is less than the minimum step size, append the end point and break
                terrain.append((self.width, self.ground_y))
                break
            max_step = min(150, remaining_width)
            step = random.randint(50, max_step)
            x += step
            y_variation = random.randint(-50, 50)
            y = self.ground_y + y_variation
            y = max(0, min(self.height, y))  # Ensure y stays within screen
            terrain.append((x, y))
        self.terrain = terrain

    def generate_hilly_terrain(self):
        terrain = [(0, self.ground_y)]
        x = 0
        while x < self.width:
            remaining_width = self.width - x
            if remaining_width < 100:
                # If the remaining width is less than the minimum step size for hilly terrain, append the end point and break
                terrain.append((self.width, self.ground_y))
                break
            max_step = min(200, remaining_width)
            step = random.randint(100, max_step)
            x += step
            y_variation = random.randint(-100, 100)
            y = self.ground_y + y_variation
            y = max(0, min(self.height, y))
            terrain.append((x, y))
        self.terrain = terrain

    def add_creature(self, creature):
        # Ensure the creature has muscles before adding
        if len(creature.muscles) == 0:
            if len(creature.bones) < 2:
                bone1 = creature.add_bone((self.width // 2, self.ground_y), 50, 5)
                bone2 = creature.add_bone((bone1.end_pos[0], bone1.end_pos[1]), 50, 5)
                creature.add_muscle(bone1, bone2)
        if len(creature.muscles) == 0:
            raise ValueError("Creature must have at least one muscle before being added to the environment.")
        self.creatures.append(creature)

    def step(self, actions):
        observations = []
        rewards = []
        dones = []

        for creature, action in zip(self.creatures, actions):
            creature.update(action)

            # Reward is based on distance traveled to the right, stability, and energy efficiency
            initial_x = creature.x
            movement = creature.x - initial_x
            reward = movement

            # Stability Reward: Penalize if the lowest point of the creature is too low
            lowest_point = max(bone.start_pos[1] for bone in creature.bones)
            if lowest_point > self.ground_y:
                reward -= 10  # Penalty for touching the ground

            # Energy Efficiency: Penalize high muscle activation
            energy_penalty = sum(abs(muscle.activation) for muscle in creature.muscles)
            reward -= energy_penalty * 0.1

            # Check if creature is on the ground (fallen)
            if lowest_point > self.height:
                reward -= 100  # Heavy penalty for falling off the screen
                done = True
            else:
                done = False

            observations.append(creature.get_state())
            rewards.append(reward)
            dones.append(done)

        self.time_step += 1
        # Terminate episode if max steps reached
        if self.time_step >= self.max_steps:
            dones = [True for _ in dones]

        return observations, rewards, dones, {}

    def render(self, screen):
        # Draw terrain
        pygame.draw.lines(screen, (100, 100, 100), False, self.terrain, 2)

        # Draw creatures
        for creature in self.creatures:
            creature.draw(screen)

        # Highlight the best creature (if more than one creature)
        if len(self.creatures) > 1:
            best_creature = max(self.creatures, key=lambda c: c.x)
            for bone in best_creature.bones:
                pygame.draw.line(screen, (255, 215, 0), bone.start_pos, bone.end_pos, bone.thickness + 2)  # Highlight in gold

    def reset(self):
        self.creatures = []
        self.generate_random_terrain()  # Randomize terrain on reset
        self.time_step = 0