# Evo-Creature-AI

Evo-Creature-AI is an open-source game that reimagines the concept of Evolution.io with modern machine learning techniques. It offers a unique platform for designing and evolving 2D creatures in a physics-based environment, powered by cutting-edge AI algorithms.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [AI Techniques](#ai-techniques)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Reporting Issues](#reporting-issues)
- [License](#license)

## Overview

Evo-Creature-AI allows players to design simple 2D creatures with animated bones, muscles, and joints on a grid. These creatures are then dropped into a game world as agents, learning to tackle various challenges such as running, navigating rough terrain, or achieving specific goals. The game utilizes modern ML/DL algorithms to evolve and optimize creature behavior over time.

## Features

- Interactive creature designer with intuitive bone and muscle creation
- Physics-based 2D environment with customizable terrains
- Deep Q-Learning Network (DQN) for creature behavior learning
- Multiple challenge environments (running, rough terrain navigation, etc.)
- Batched parallel background training for improved performance
- Experimental KAN (Knowledge-Augmented Neural) models integration (coming soon)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Evo-Creature-AI.git
   cd Evo-Creature-AI
   ```

2. Install the required dependencies:
   ```
   pip install pygame torch numpy
   ```

### System Requirements

- Python 3.11+
- 4GB RAM (8GB recommended for larger simulations)
- Graphics card with OpenGL 3.3+ support

Compatibility: Windows 10+, macOS 10.14+, Linux (major distributions)

## Usage

Run the main script to start the game:

```
python main-py.py
```

- Use the mouse to design your creature in the design mode.
- Left-click and drag to create bones.
- Right-click to select bones and create joints.
- Press the spacebar to switch between design and simulation modes.
- In simulation mode, watch your creature learn and evolve!

## Technical Details

Evo-Creature-AI is built using:

- Python 3.11+
- Pygame for graphics and user interface
- PyTorch for neural network implementation
- Custom physics engine for 2D creature simulation

The project structure includes:

- `main-py.py`: Main game loop and mode switching
- `creature_designer.py`: Interface for designing creatures
- `creature.py`: Classes for Creature, Bone, Joint, and Muscle
- `environment.py`: Simulation environment and terrain generation
- `ai_agent.py`: Implementation of the DQN agent

## AI Techniques

Evo-Creature-AI employs several advanced AI techniques:

1. Deep Q-Learning Network (DQN): Used for decision-making in creature behavior.
2. Experience Replay: Improves learning stability and efficiency.
3. Batched Parallel Training: Allows for faster learning across multiple instances.
4. Knowledge-Augmented Neural (KAN) Models: (Upcoming) Will incorporate domain knowledge into the learning process.

## Roadmap

- Implement advanced terrain generation algorithms
- Add more complex challenges and environments
- Introduce creature species and inter-species competition
- Develop a web-based version for easier access
- Integrate more sophisticated evolutionary algorithms
- **Actually work (Claude got ahead of itself there but I think this is a fun roadmap)**

## Contributing

We welcome contributions to Evo-Creature-AI! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

Please ensure your code adheres to the project's coding standards and include tests for new features.

## Reporting Issues

If you encounter any bugs or have feature requests, please open an issue on our GitHub repository. When reporting issues, please include:

- A clear and descriptive title
- A detailed description of the issue or feature request
- Steps to reproduce the issue (for bugs)
- Your environment details (OS, Python version, etc.)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

Evo-Creature-AI is a work in progress. We're excited to see how it evolves with community contributions and new AI techniques!
