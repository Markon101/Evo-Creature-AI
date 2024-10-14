import pygame
import random
import math

class Environment:
    def __init__(self, width, height, ground_y):
        self.width = width
        self.height = height
        self.ground_y = ground_y
        self.creatures = []
        self.terrain = self.generate_flat_terrain()
        self.time_step = 0
        self.max_steps = 1000  # Maximum steps per episode

    def generate_flat_terrain(self):
        return [(0, self.ground_y), (self.width, self.ground_y)]

    def generate_random_terrain(self):
        terrain = [(0, self.ground_y)]
        x = 0
        while x < self.width:
            x += random.randint(50, 150)
            y = self.ground_y + random.randint(-50, 50)
            terrain.append((x, y))
        return terrain

    def generate_hilly_terrain(self):
        terrain = [(0, self.ground_y)]
        x = 0
        while x < self.width:
            x += random.randint(100, 200)
            y = self.ground_y + random.randint(-100, 100)
            terrain.append((x, y))
        return terrain

    def add_creature(self, creature):
        self.creatures.append(creature)

    def step(self, actions):
        observations = []
        rewards = []
        dones = []

        for creature, action in zip(self.creatures, actions):
            creature.update(action)

            # Reward is based on distance traveled to the right, stability, and energy efficiency
            reward = creature.x - creature.bones[0].start_pos[0]
            creature.x = creature.bones[0].start_pos[0]

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
        self.terrain = self.generate_flat_terrain()  # You can also randomize terrain here
        self.time_step = 0