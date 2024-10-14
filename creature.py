import pygame
import math

class Bone:
    def __init__(self, start_pos, length, thickness, angle=0.0, mass=1.0):
        self.start_pos = list(start_pos)
        self.length = length
        self.thickness = thickness
        self.angle = angle
        self.mass = mass
        self.velocity = [0.0, 0.0]  # Velocity attribute
        self.end_pos = self.calculate_end_pos()

    def calculate_end_pos(self):
        return [
            self.start_pos[0] + self.length * math.cos(self.angle),
            self.start_pos[1] + self.length * math.sin(self.angle)
        ]

    def apply_force(self, force):
        # Update velocity based on force and mass (simple physics)
        acceleration = [force[0] / self.mass, force[1] / self.mass]
        self.velocity[0] += acceleration[0]
        self.velocity[1] += acceleration[1]

    def update(self):
        # Update position based on velocity
        self.start_pos[0] += self.velocity[0]
        self.start_pos[1] += self.velocity[1]
        self.end_pos = self.calculate_end_pos()

    def draw(self, screen):
        pygame.draw.line(screen, (255, 255, 255), self.start_pos, self.end_pos, self.thickness)

class Joint:
    def __init__(self, bone1, bone2, angle_limit=(-math.pi / 4, math.pi / 4)):
        self.bone1 = bone1
        self.bone2 = bone2
        self.angle_limit = angle_limit

    def enforce_constraints(self):
        # Calculate current angle between bones
        dx = self.bone2.start_pos[0] - self.bone1.end_pos[0]
        dy = self.bone2.start_pos[1] - self.bone1.end_pos[1]
        current_angle = math.atan2(dy, dx)

        # Clamp the angle within the joint limits
        clamped_angle = max(self.angle_limit[0], min(self.angle_limit[1], current_angle))
        angle_diff = clamped_angle - current_angle

        # Apply correction to bone2 position
        self.bone2.start_pos[0] = self.bone1.end_pos[0] + self.bone2.length * math.cos(clamped_angle)
        self.bone2.start_pos[1] = self.bone1.end_pos[1] + self.bone2.length * math.sin(clamped_angle)

    def draw(self, screen):
        # Draw a visual representation of the joint
        if isinstance(self.bone1.end_pos, (list, tuple)) and isinstance(self.bone2.start_pos, (list, tuple)):
            try:
                center = (int(self.bone1.end_pos[0]), int(self.bone1.end_pos[1]))
                pygame.draw.circle(screen, (0, 255, 0), center, 5)
            except (TypeError, ValueError):
                print("Invalid joint coordinates:", self.bone1.end_pos, self.bone2.start_pos)

class Muscle:
    def __init__(self, bone1, bone2, strength=1.0, damping=0.1):
        self.bone1 = bone1
        self.bone2 = bone2
        self.strength = strength
        self.damping = damping
        self.activation = 0.0

    def contract(self, activation):
        # PD control for smoother muscle contraction
        self.activation = max(0.0, min(1.0, activation))
        desired_length = self.bone1.length  # Assume resting length is initial length
        current_length = math.dist(self.bone1.start_pos, self.bone2.start_pos)
        length_error = desired_length - current_length

        # Apply spring force proportional to length error
        force_magnitude = self.strength * length_error - self.damping * self.activation
        angle = math.atan2(self.bone2.start_pos[1] - self.bone1.start_pos[1],
                           self.bone2.start_pos[0] - self.bone1.start_pos[0])
        force = [force_magnitude * math.cos(angle), force_magnitude * math.sin(angle)]

        # Apply forces to bones
        self.bone1.apply_force(force)
        self.bone2.apply_force([-f for f in force])

    def draw(self, screen):
        # Draw a visual representation of the muscle
        pygame.draw.line(screen, (255, 0, 0), self.bone1.start_pos, self.bone2.start_pos, 2)

class Creature:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.bones = []
        self.joints = []
        self.muscles = []
        self.alive = True

    def add_bone(self, start_pos, length, thickness, angle=0.0):
        if length <= 0:
            print("Cannot create a bone of length 0.")
            return None
        bone = Bone(start_pos, length, thickness, angle)
        self.bones.append(bone)
        return bone

    def add_joint(self, bone1, bone2, angle_limit=(-math.pi / 4, math.pi / 4)):
        if bone1 and bone2:
            joint = Joint(bone1, bone2, angle_limit)
            self.joints.append(joint)
            return joint
        else:
            print("Cannot create a joint without two valid bones.")
            return None

    def add_muscle(self, bone1, bone2, strength=1.0, damping=0.1):
        if bone1 and bone2:
            muscle = Muscle(bone1, bone2, strength, damping)
            self.muscles.append(muscle)
            return muscle
        else:
            print("Cannot create a muscle without two valid bones.")
            return None

    def update(self, actions):
        if len(self.muscles) == 0:
            raise ValueError("Creature must have at least one muscle to update.")
        if len(actions) != len(self.muscles):
            raise ValueError(f"Expected {len(self.muscles)} actions, but got {len(actions)} actions.")
        for muscle, action in zip(self.muscles, actions):
            muscle.contract(action)
        self.apply_physics()

    def apply_physics(self):
        # Apply gravity
        for bone in self.bones:
            gravity_force = [0, 0.5 * bone.mass]
            bone.apply_force(gravity_force)

        # Enforce joint constraints
        for joint in self.joints:
            joint.enforce_constraints()

        # Update all bones
        for bone in self.bones:
            bone.update()

    def draw(self, screen):
        # Draw all bones
        for bone in self.bones:
            bone.draw(screen)
        # Draw all joints
        for joint in self.joints:
            joint.draw(screen)
        # Draw all muscles
        for muscle in self.muscles:
            muscle.draw(screen)

    def get_state(self):
        # Return a list of all bone angles, positions, and muscle activations
        state = []
        for bone in self.bones:
            state.extend([bone.angle, bone.start_pos[0], bone.start_pos[1]])
        for muscle in self.muscles:
            state.append(muscle.activation)
        return state