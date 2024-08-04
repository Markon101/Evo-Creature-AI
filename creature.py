import pygame
import math

class Bone:
    def __init__(self, start_pos, length, thickness, angle=0.0, mass=1.0):
        self.start_pos = list(start_pos)
        self.length = length
        self.thickness = thickness
        self.angle = angle
        self.mass = mass
        self.end_pos = self.calculate_end_pos()

    def calculate_end_pos(self):
        return [
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        ]

    def update(self):
        self.end_pos = self.calculate_end_pos()

    def draw(self, screen):
        pygame.draw.line(screen, (255, 255, 255), self.start_pos, self.end_pos, self.thickness)

class Joint:
    def __init__(self, bone1, bone2, angle_limit=(-math.pi, math.pi)):
        self.bone1 = bone1
        self.bone2 = bone2
        self.angle_limit = angle_limit

class Muscle:
    def __init__(self, bone1, bone2, strength=1.0):
        self.bone1 = bone1
        self.bone2 = bone2
        self.strength = strength
        self.activation = 0.0

    def contract(self, activation):
        self.activation = max(0.0, min(1.0, activation))

class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bones = []
        self.joints = []
        self.muscles = []
        self.alive = True

    def add_bone(self, start_pos, length, thickness, angle=0.0):
        bone = Bone(start_pos, length, thickness, angle)
        self.bones.append(bone)
        return bone

    def add_joint(self, bone1, bone2, angle_limit=(-math.pi, math.pi)):
        joint = Joint(bone1, bone2, angle_limit)
        self.joints.append(joint)
        return joint

    def add_muscle(self, bone1, bone2, strength=1.0):
        muscle = Muscle(bone1, bone2, strength)
        self.muscles.append(muscle)
        return muscle

    def update(self, actions):
        for muscle, action in zip(self.muscles, actions):
            muscle.contract(action)
        self.apply_physics()

    def apply_physics(self):
        # Apply gravity
        for bone in self.bones:
            bone.start_pos[1] += 0.5  # Simple gravity

        # Apply muscle forces
        for muscle in self.muscles:
            force = muscle.activation * muscle.strength
            angle = math.atan2(muscle.bone2.start_pos[1] - muscle.bone1.start_pos[1],
                               muscle.bone2.start_pos[0] - muscle.bone1.start_pos[0])
            muscle.bone1.start_pos[0] += force * math.cos(angle)
            muscle.bone1.start_pos[1] += force * math.sin(angle)
            muscle.bone2.start_pos[0] -= force * math.cos(angle)
            muscle.bone2.start_pos[1] -= force * math.sin(angle)

        # Enforce joint constraints
        for joint in self.joints:
            dx = joint.bone2.start_pos[0] - joint.bone1.end_pos[0]
            dy = joint.bone2.start_pos[1] - joint.bone1.end_pos[1]
            angle = math.atan2(dy, dx)
            angle = max(joint.angle_limit[0], min(joint.angle_limit[1], angle))
            joint.bone2.start_pos[0] = joint.bone1.end_pos[0] + joint.bone2.length * math.cos(angle)
            joint.bone2.start_pos[1] = joint.bone1.end_pos[1] + joint.bone2.length * math.sin(angle)

        # Update all bones
        for bone in self.bones:
            bone.update()

    def draw(self, screen):
        for bone in self.bones:
            bone.draw(screen)

    def get_state(self):
        # Return a list of all bone angles and positions
        state = []
        for bone in self.bones:
            state.extend([bone.angle, bone.start_pos[0], bone.start_pos[1]])
        return state
