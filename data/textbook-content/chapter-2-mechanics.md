# Chapter 2: Mechanical Systems and Kinematics

## Section 2.1: Robot Anatomy and Structure

A humanoid robot consists of several interconnected mechanical systems that mimic human anatomy:

**Head and Neck**: Contains cameras, microphones, and sensors for perception. The neck provides degrees of freedom for looking around.

**Torso**: Houses the main computation unit, power systems, and serves as the central rigid body.

**Arms and Hands**: Equipped with multiple joints to provide dexterity. The shoulder has three degrees of freedom, the elbow has one, and the wrist has three degrees of freedom.

**Legs and Feet**: Provide locomotion and balance. Each leg typically has hip, knee, and ankle joints. The feet contain pressure sensors for feedback during walking.

## Section 2.2: Forward and Inverse Kinematics

Kinematics is the study of motion without considering forces. Forward kinematics describes the position and orientation of the robot's end-effector (like a hand) given the joint angles.

Inverse kinematics is the reverse problem: given a desired position and orientation, calculate the joint angles needed to reach that position. This is a fundamental problem in robotics and often has multiple solutions or no solution for certain target positions.

The mathematical representation uses Denavit-Hartenberg (DH) parameters and transformation matrices to describe the robot's geometry.

## Section 2.3: Actuators and Drive Systems

Humanoid robots use different types of actuators to move their joints:

- **Electric Motors**: Most common, offer good control and efficiency
- **Hydraulic Actuators**: Provide high power density but are heavy and require fluid systems
- **Pneumatic Actuators**: Light and backdrivable, but less precise
- **Series Elastic Actuators (SEA)**: Allow force control and compliance, important for safe human-robot interaction

Modern humanoid robots increasingly use electric motors with reduction gears and force-torque sensors in the joints.

## Section 2.4: Compliance and Impedance Control

Unlike rigid manipulators, humanoid robots benefit from having some compliance in their joints. This allows them to:

1. Absorb impact and vibration
2. Adapt to contact forces from the environment
3. Interact safely with humans
4. Distribute forces across multiple joints

Impedance control allows the robot to behave like a virtual mass-spring-damper system, enabling natural interaction with the environment.

