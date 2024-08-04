import pygame
import random

class Environment:
    def __init__(self, width, height, ground_y):
        self.width = width
        self.height = height
        self.ground_y = ground_y
        self.creatures = []
        self.terrain = self.generate_flat_terrain()

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

    def add_creature(self, creature):
        self.creatures.append(creature)

    def step(self, actions):
        observations = []
        rewards = []
        dones = []

        for creature, action in zip(self.creatures, actions):
            creature.update(action)
            
            # Simple reward: distance traveled to the right
            reward = creature.x - creature.bones[0].start_pos[0]
            creature.x = creature.bones[0].start_pos[0]

            # Check if creature is on the ground
            lowest_point = max(bone.start_pos[1] for bone in creature.bones)
            if lowest_point > self.ground_y:
                reward -= 10  # Penalty for touching the ground

            observations.append(creature.get_state())
            rewards.append(reward)
            dones.append(False)  # For now, episodes don't end

        return observations, rewards, dones, {}

    def render(self, screen):
        # Draw terrain
        pygame.draw.lines(screen, (100, 100, 100), False, self.terrain, 2)

        # Draw creatures
        for creature in self.creatures:
            creature.draw(screen)
