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
        # Optional: Implement visual preview of the bone being drawn
        pass

    def finish_drawing_bone(self, pos):
        if self.drawing_bone:
            end_pos = self.snap_to_grid(pos)
            length = math.dist(self.start_pos, end_pos)
            if length == 0:
                print("Cannot create a bone of length 0.")
                self.drawing_bone = False
                self.start_pos = None
                return
            angle = math.atan2(end_pos[1] - self.start_pos[1], end_pos[0] - self.start_pos[0])
            new_bone = self.creature.add_bone(self.start_pos, length, 3, angle)
            
            if new_bone is None:
                self.drawing_bone = False
                self.start_pos = None
                return

            if self.selected_bone:
                # Create a joint between selected bone and new bone
                joint = self.creature.add_joint(self.selected_bone, new_bone)
                
                # Create a muscle between selected bone and new bone
                muscle = self.creature.add_muscle(self.selected_bone, new_bone)
                if joint is None or muscle is None:
                    print("Failed to create joint or muscle.")
            
            self.drawing_bone = False
            self.start_pos = None

    def select_bone(self, pos):
        for bone in self.creature.bones:
            if self.point_on_bone(pos, bone):
                self.selected_bone = bone
                print("Bone selected:", bone)
                return
        self.selected_bone = None
        print("No bone selected.")

    def point_on_bone(self, point, bone):
        # Check if point is close to the bone (simplified)
        # Calculate distance from point to the line segment
        x, y = point
        x1, y1 = bone.start_pos
        x2, y2 = bone.end_pos
        if x1 == x2 and y1 == y2:
            return False
        # Compute the projection of point onto the bone
        A = x - x1
        B = y - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1

        if param < 0 or param > 1:
            return False

        xx = x1 + param * C
        yy = y1 + param * D

        distance = math.dist((x, y), (xx, yy))
        return distance < 10  # 10 pixel tolerance

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
            end_pos = self.snap_to_grid(end_pos)
            pygame.draw.line(screen, (100, 100, 255), self.start_pos, end_pos, 2)

    def get_designed_creature(self):
        # Ensure the creature has at least one muscle before returning
        if len(self.creature.muscles) == 0 and len(self.creature.bones) >=2:
            bone1 = self.creature.bones[0]
            bone2 = self.creature.bones[1]
            self.creature.add_muscle(bone1, bone2)
        return self.creature