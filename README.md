# Evo-Creature-AI

Evo-Creature-AI is an open-source game that aims to provide an experience similar to the excellent Evolution.io, but with a modern twist. It incorporates cutting-edge machine learning and deep learning algorithms, including batched parallel background training and experimental KAN (Knowledge-Augmented Neural) models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

Evo-Creature-AI allows players to design simple 2D creatures with animated bones, muscles, and joints on a grid. These creatures are then dropped into a game world as agents, learning to tackle various challenges such as running, navigating rough terrain, or achieving specific goals. The game utilizes modern ML/DL algorithms to evolve and optimize creature behavior over time.

## Features

- Interactive creature designer
- Physics-based 2D environment
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

## Usage

Run the main script to start the game:

```
python main-py.py
```

- Use the mouse to design your creature in the design mode.
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

## Contributing

We welcome contributions to Evo-Creature-AI! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

Please ensure your code adheres to the project's coding standards and include tests for new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Evo-Creature-AI is a work in progress. We're excited to see how it evolves with community contributions and new AI techniques!
