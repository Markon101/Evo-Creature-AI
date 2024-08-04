import pygame
import math
from creature import Creature, Bone, Joint, Muscle

class CreatureDesigner:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_size = 20
        self.creature = Creature(screen_width // 2, screen_height // 2)
        self.selected_bone = None
        self.drawing_bone = False
        self.start_pos = None

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                self.start_drawing_bone(event.pos)
            elif event.button == 3:  # Right click
                self.select_bone(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.finish_drawing_bone(event.pos)
        elif event.type == pygame.MOUSEMOTION:
            if self.drawing_bone:
                self.update_drawing_bone(event.pos)

    def start_drawing_bone(self, pos):
        self.drawing_bone = True
        self.start_pos = self.snap_to_grid(pos)

    def update_drawing_bone(self, pos):
        # This method would update a preview of the bone being drawn
        pass

    def finish_drawing_bone(self, pos):
        if self.drawing_bone:
            end_pos = self.snap_to_grid(pos)
            length = math.dist(self.start_pos, end_pos)
            angle = math.atan2(end_pos[1] - self.start_pos[1], end_pos[0] - self.start_pos[0])
            new_bone = Bone(self.start_pos, length, 3, angle)
            self.creature.bones.append(new_bone)
            
            if self.selected_bone:
                # Create a joint between selected bone and new bone
                joint = Joint(self.selected_bone, new_bone)
                self.creature.joints.append(joint)
                
                # Create a muscle between selected bone and new bone
                muscle = Muscle(self.selected_bone, new_bone)
                self.creature.muscles.append(muscle)

            self.drawing_bone = False
            self.start_pos = None

    def select_bone(self, pos):
        for bone in self.creature.bones:
            if self.point_on_bone(pos, bone):
                self.selected_bone = bone
                return
        self.selected_bone = None

    def point_on_bone(self, point, bone):
        # Check if point is close to the bone (simplified)
        dist = math.dist(point, bone.start_pos) + math.dist(point, bone.end_pos)
        return abs(dist - bone.length) < 5  # 5 pixel tolerance

    def snap_to_grid(self, pos):
        return (round(pos[0] / self.grid_size) * self.grid_size,
                round(pos[1] / self.grid_size) * self.grid_size)

    def draw(self, screen):
        # Draw grid
        for x in range(0, self.screen_width, self.grid_size):
            pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.grid_size):
            pygame.draw.line(screen, (50, 50, 50), (0, y), (self.screen_width, y))

        # Draw creature
        self.creature.draw(screen)

        # Draw selected bone highlight
        if self.selected_bone:
            pygame.draw.line(screen, (255, 255, 0), self.selected_bone.start_pos, 
                             self.selected_bone.end_pos, self.selected_bone.thickness + 2)

        # Draw bone being created
        if self.drawing_bone and self.start_pos:
            end_pos = pygame.mouse.get_pos()
            pygame.draw.line(screen, (100, 100, 255), self.start_pos, end_pos, 2)

    def get_designed_creature(self):
        return self.creature
