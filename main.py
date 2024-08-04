import pygame
import sys
from creature_designer import CreatureDesigner
from environment import Environment
from ai_agent import DQNAgent

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

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("EvolutionAI")
clock = pygame.time.Clock()

# Initialize components
designer = CreatureDesigner(WIDTH, HEIGHT)
env = Environment(WIDTH, HEIGHT, GROUND_Y)
agent = None  # We'll initialize this after creature design

# Game states
DESIGN = 0
SIMULATE = 1
game_state = DESIGN

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if game_state == DESIGN:
                    # Switch to simulation mode
                    designed_creature = designer.get_designed_creature()
                    env.add_creature(designed_creature)
                    state_size = len(designed_creature.get_state())
                    action_size = len(designed_creature.muscles)
                    agent = DQNAgent(state_size, action_size)
                    game_state = SIMULATE
                else:
                    # Switch back to design mode
                    game_state = DESIGN
                    env = Environment(WIDTH, HEIGHT, GROUND_Y)
                    agent = None
        
        if game_state == DESIGN:
            designer.handle_event(event)

    screen.fill(BLACK)

    if game_state == DESIGN:
        designer.draw(screen)
    elif game_state == SIMULATE:
        # Run simulation step
        states = [creature.get_state() for creature in env.creatures]
        actions = [agent.act(state) for state in states]
        next_states, rewards, dones, _ = env.step(actions)
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            agent.remember(state, action, reward, next_state, done)
        
        agent.replay(32)  # Train on a batch of 32 samples
        
        # Render environment
        env.render(screen)

    # Draw UI
    font = pygame.font.Font(None, 36)
    mode_text = font.render("Mode: " + ("Design" if game_state == DESIGN else "Simulate"), True, WHITE)
    screen.blit(mode_text, (10, 10))
    instruction_text = font.render("Press SPACE to switch modes", True, WHITE)
    screen.blit(instruction_text, (10, 50))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
sys.exit()
