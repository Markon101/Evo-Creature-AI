import pygame
import sys
import os
from creature_designer import CreatureDesigner
from environment import Environment
from ai_agent import DQNAgent
import itertools

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
GROUND_Y = 700
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EvolutionAI")
clock = pygame.time.Clock()

# Initialize components
designer = CreatureDesigner(WIDTH, HEIGHT)
env = Environment(WIDTH, HEIGHT, GROUND_Y)
agent = None  # Will initialize after creature design

# Game states
DESIGN = 0
SIMULATE = 1
TRAIN = 2
game_state = DESIGN

# Define activation levels per muscle
ACTIVATION_LEVELS = 3  # e.g., 0: off, 1: medium, 2: high

# Define UI Buttons
class Button:
    def __init__(self, rect, color, text, text_color=BLACK):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.text = text
        self.text_color = text_color
        self.font = pygame.font.Font(None, 24)
        self.text_surf = self.font.render(text, True, self.text_color)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        screen.blit(self.text_surf, self.text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

# Create buttons
reset_button = Button((WIDTH - 110, 10, 100, 40), GRAY, "Reset")
train_button = Button((WIDTH - 110, 60, 100, 40), GRAY, "Train")
pause_button = Button((WIDTH - 110, 110, 100, 40), GRAY, "Pause")
save_button = Button((WIDTH - 110, 160, 100, 40), GRAY, "Save Model")
load_button = Button((WIDTH - 110, 210, 100, 40), GRAY, "Load Model")

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # Save the model upon exiting if training
            if agent and game_state == TRAIN:
                agent.save("dqn_model.pth")
                print("Model saved upon exit.")
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_state == DESIGN:
                    # Switch to simulation mode
                    designed_creature = designer.get_designed_creature()
                    if designed_creature:
                        env.add_creature(designed_creature)
                        num_muscles = len(designed_creature.muscles)
                        if num_muscles == 0:
                            print("Creature has no muscles. Please design with at least one muscle.")
                            continue
                        action_size = ACTIVATION_LEVELS ** num_muscles
                        state_size = len(designed_creature.get_state())
                        agent = DQNAgent(state_size, action_size, ACTIVATION_LEVELS)
                        game_state = SIMULATE
                else:
                    # Switch back to design mode
                    game_state = DESIGN
                    env = Environment(WIDTH, HEIGHT, GROUND_Y)  # Reset environment
                    designer = CreatureDesigner(WIDTH, HEIGHT)
                    agent = None
            elif event.key == pygame.K_r and game_state == SIMULATE:
                # Reset simulation
                env = Environment(WIDTH, HEIGHT, GROUND_Y)
                designed_creature = designer.get_designed_creature()
                if designed_creature:
                    try:
                        env.add_creature(designed_creature)
                        num_muscles = len(designed_creature.muscles)
                        if num_muscles == 0:
                            print("Creature has no muscles. Please design with at least one muscle.")
                            continue
                        action_size = ACTIVATION_LEVELS ** num_muscles
                        state_size = len(designed_creature.get_state())
                        agent = DQNAgent(state_size, action_size, ACTIVATION_LEVELS)
                    except ValueError as e:
                        print(e)
                        continue

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                pos = event.pos
                if reset_button.is_clicked(pos):
                    env.reset()
                    if agent:
                        agent = DQNAgent(len(designed_creature.get_state()), ACTIVATION_LEVELS ** len(designed_creature.muscles), ACTIVATION_LEVELS)
                    print("Simulation reset.")
                elif train_button.is_clicked(pos):
                    if agent:
                        game_state = TRAIN
                        print("Training started.")
                    else:
                        print("No agent available. Please design and simulate a creature first.")
                elif pause_button.is_clicked(pos):
                    if game_state == SIMULATE:
                        game_state = DESIGN
                        print("Simulation paused.")
                    elif game_state == DESIGN:
                        game_state = SIMULATE
                        print("Simulation resumed.")
                elif save_button.is_clicked(pos):
                    if agent:
                        agent.save("dqn_model.pth")
                        print("Model saved.")
                    else:
                        print("No agent to save.")
                elif load_button.is_clicked(pos):
                    if os.path.exists("dqn_model.pth"):
                        try:
                            if agent:
                                agent.load("dqn_model.pth")
                            else:
                                num_muscles = len(designed_creature.muscles)
                                action_size = ACTIVATION_LEVELS ** num_muscles
                                state_size = len(designed_creature.get_state())
                                agent = DQNAgent(state_size, action_size, ACTIVATION_LEVELS)
                                agent.load("dqn_model.pth")
                            print("Model loaded.")
                        except Exception as e:
                            print("Failed to load model:", e)
                    else:
                        print("No saved model found.")

        if game_state == DESIGN:
            designer.handle_event(event)

    screen.fill(BLACK)

    if game_state == DESIGN:
        designer.draw(screen)
        # Instructions
        font = pygame.font.Font(None, 36)
        instruction_text = font.render("Use mouse to design creature. Press SPACE to simulate.", True, WHITE)
        screen.blit(instruction_text, (10, 10))
    elif game_state == SIMULATE:
        # Run simulation step
        states = [creature.get_state() for creature in env.creatures]
        actions_indices = [agent.act(state) for state in states]
        actions = [decode_action(action_index, len(creature.muscles), ACTIVATION_LEVELS)
                   for action_index, creature in zip(actions_indices, env.creatures)]
        next_states, rewards, dones, _ = env.step(actions)

        # Training step (if in TRAIN state)
        if game_state == TRAIN:
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                # Convert action list back to a single index for storage
                action_index = 0
                for a in action:
                    action_index = action_index * ACTIVATION_LEVELS + a
                agent.remember(state, action_index, reward, next_state, done)
            agent.replay(32)  # Train on a batch of 32 samples

            # Optionally, save the model periodically or based on certain conditions
            # Here, we'll save every 1000 steps
            if agent.step_counter % 1000 == 0:
                agent.save("dqn_model.pth")
                print("Model saved at step", agent.step_counter)

        # Render environment
        env.render(screen)

        # Draw UI
        font = pygame.font.Font(None, 36)
        mode_text = font.render("Mode: Simulation", True, WHITE)
        screen.blit(mode_text, (10, 10))
        reset_text = font.render("Press 'R' to reset simulation", True, WHITE)
        screen.blit(reset_text, (10, 50))
        if agent:
            epsilon_text = font.render(f"Epsilon: {agent.epsilon:.2f}", True, WHITE)
            screen.blit(epsilon_text, (10, 90))

    # Draw buttons
    reset_button.draw(screen)
    train_button.draw(screen)
    pause_button.draw(screen)
    save_button.draw(screen)
    load_button.draw(screen)

    pygame.display.flip()
    clock.tick(FPS)

    # Optionally, save the model when exiting
    if not running and agent is not None:
        agent.save("dqn_model.pth")

pygame.quit()
sys.exit()

def decode_action(action_index, num_muscles, activation_levels):
    """
    Decodes the action index into a list of muscle activations.
    """
    activations = []
    for _ in range(num_muscles):
        activations.append(action_index % activation_levels)
        action_index = action_index // activation_levels
    return activations[::-1]  # Reverse to maintain order