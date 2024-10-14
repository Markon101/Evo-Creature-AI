# EvoGame: Evolutionary Creature Simulation

![EvoGame Banner](https://your-image-link.com/banner.png)

EvoGame is an innovative simulation game that combines evolutionary algorithms with artificial intelligence to create, train, and evolve intelligent creatures. Users can design creatures with customizable bones, joints, and muscles, and watch as AI agents learn to navigate dynamically generated terrains. Whether you're a developer, researcher, or enthusiast, EvoGame offers a unique platform to explore the fascinating interplay between biology-inspired design and machine learning.

## Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Controls](#controls)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Creature Designer:** Intuitive interface to create and customize creatures by adding bones, joints, and muscles.
- **Dynamic Terrain Generation:** Automatically generates varied terrains (flat, random, hilly) for creatures to navigate.
- **AI Integration:** Utilizes Deep Q-Networks (DQN) for training AI agents to control creature movements.
- **User Interface Controls:** Easy-to-use buttons for resetting simulations, initiating training, pausing, and managing AI models.
- **Model Persistence:** Save and load trained AI models to continue training or for inference in future sessions.
- **Visual Feedback:** Real-time visualization of creature structures, joint connections, muscle activations, and training metrics.
- **Performance Optimization:** Efficient rendering and physics calculations to ensure smooth simulation experiences.

## Demo

![EvoGame Demo](https://your-image-link.com/demo.gif)

*Watch as a creature navigates through a randomly generated terrain, controlled by an AI agent.*

## Installation

### Prerequisites

- **Python 3.11** or higher
- **Pip** package manager

### Clone the Repository

```bash
git clone https://github.com/yourusername/evogame.git
cd evogame
```

### Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv evocreature
source evocreature/bin/activate  # On Windows: evocreature\Scripts\activate
```

### Install Dependencies

Ensure you have `pip` updated:

```bash
pip install --upgrade pip
```

Install required packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```plaintext
pygame==2.6.1
torch==2.0.1
numpy==1.25.0
```

*Note: Ensure that your system meets the requirements for PyTorch. Visit [PyTorch Installation](https://pytorch.org/get-started/locally/) for more details.*

## Usage

### Running the Simulation

Start the EvoGame simulation by executing the `main.py` script:

```bash
python main.py
```

### Creature Design Mode

1. **Adding Bones:**
   - **Left Click:** Click and drag on the grid to create bones. Ensure that the start and end points are distinct to avoid zero-length bones.
   
2. **Selecting Bones:**
   - **Right Click:** Click near a bone to select it. Selected bones are highlighted in yellow.

3. **Connecting Bones:**
   - After selecting a bone, create a new bone to automatically connect it via a joint and muscle.

### Simulation Mode

1. **Switching Modes:**
   - **Press `SPACE`:** Toggle between Design and Simulation modes.

2. **Training AI Agents:**
   - **Click "Train" Button:** Initiate the training process where AI agents learn to control the creatures.
   - **Monitor Epsilon:** Observe the exploration rate of the AI agent displayed on the screen.

3. **Controls:**
   - **Reset Simulation:** Click the "Reset" button to restart the simulation environment.
   - **Pause/Resume Simulation:** Click the "Pause" button to toggle between pausing and resuming the simulation.
   - **Save Model:** Click the "Save Model" button to persist the current AI model.
   - **Load Model:** Click the "Load Model" button to load a previously saved AI model.

## Controls

| Action                | Description                                        |
|-----------------------|----------------------------------------------------|
| **Left Click**        | Start drawing a bone by clicking and dragging.     |
| **Right Click**       | Select a bone to connect with a new bone.          |
| **Press `SPACE`**     | Toggle between Design and Simulation modes.        |
| **Click "Reset"**     | Reset the simulation environment.                  |
| **Click "Train"**     | Start training the AI agent.                       |
| **Click "Pause"**     | Pause or resume the simulation.                    |
| **Click "Save Model"**| Save the current AI model to disk.                  |
| **Click "Load Model"**| Load a saved AI model from disk.                    |
| **Press `R`**         | Reset the simulation during Simulation mode.       |

## Troubleshooting

### Common Errors

1. **Cannot Create a Bone of Length 0**

   ```
   Cannot create a bone of length 0.
   ```

   **Cause:** Attempting to create a bone where the start and end positions are identical.

   **Solution:**
   - Ensure that when designing creatures, bones have distinct start and end points.
   - Adjust the grid size if necessary to reduce the likelihood of zero-length bones.

2. **TypeError: center argument must be a pair of numbers**

   ```
   TypeError: center argument must be a pair of numbers
   ```

   **Cause:** Passing invalid coordinates to `pygame.draw.circle`.

   **Solution:**
   - Verify that joints are connected between valid bones with numerical coordinates.
   - Ensure that bones are correctly initialized with positive lengths.

3. **RuntimeError: Size Mismatch in Neural Network**

   ```
   RuntimeError: Error(s) in loading state_dict for NeuralNetwork:
           size mismatch for layer1.weight: copying a param with shape torch.Size([24, 10]) from checkpoint, the shape in current model is torch.Size([24, 13]).
   ```

   **Cause:** The architecture of the current `NeuralNetwork` model differs from the one used when saving `dqn_model.pth`.

   **Solution:**
   - **Option 1:** Delete the existing `dqn_model.pth` file to allow the creation of a new model matching the current architecture.
     ```bash
     rm dqn_model.pth
     ```
   - **Option 2:** Ensure that the model architecture remains consistent between training sessions.

4. **Pygame AVX2 Warning**

   ```
   RuntimeWarning: Your system is avx2 capable but pygame was not built with support for it...
   ```

   **Cause:** Pygame was not compiled with AVX2 optimizations.

   **Solution:**
   - **Option 1:** Rebuild Pygame from source with AVX2 support.
   - **Option 2:** Ignore the warning if performance is acceptable.

### Additional Tips

- **Virtual Environment:** Use a virtual environment to manage dependencies and prevent conflicts.
- **Dependencies:** Ensure all dependencies listed in `requirements.txt` are installed correctly.
- **Model Saving:** Regularly save your AI models to prevent loss of training progress.

## Contributing

Contributions are welcome! If you'd like to improve EvoGame, please follow these steps:

1. **Fork the Repository:**
   - Click the "Fork" button at the top-right corner of the repository page.

2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/yourusername/evogame.git
   cd evogame
   ```

3. **Create a New Branch:**
   ```bash
   git checkout -b feature/YourFeatureName
   ```

4. **Make Changes and Commit:**
   ```bash
   git add .
   git commit -m "Add feature: YourFeatureName"
   ```

5. **Push to Your Fork:**
   ```bash
   git push origin feature/YourFeatureName
   ```

6. **Submit a Pull Request:**
   - Navigate to your fork on GitHub and click "New Pull Request".

### Guidelines

- **Code Quality:** Ensure your code follows Python best practices and is well-documented.
- **Testing:** Include tests for new features and ensure existing tests pass.
- **Commit Messages:** Write clear and descriptive commit messages.
- **Issue Reporting:** If you encounter bugs or have feature requests, please open an issue.

## License

Distributed under the Apache 2 License

## Acknowledgments

- **[Pygame](https://www.pygame.org/news)** - For providing a robust library for game development.
- **[PyTorch](https://pytorch.org/)** - For enabling powerful machine learning capabilities.
- **Inspiration Sources:** Thanks to the open-source community for inspiring tools and frameworks that made this project possible.
