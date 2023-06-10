# MechComposite

This project implements a melody generation algorithm using a genetic algorithm approach. The goal is to evolve melodies that have a high harmony score, structural similarity to a reference melody, low repetition, and high similarity to a reference melody.

## Dependencies

The following dependencies are required to run the project:

- Python 3.6 or higher
- numpy
- matplotlib
- pydsmusic

## Usage

1. Clone the repository:

```shell
git clone <repository-url>
```

2. Install the required dependencies:

```shell
pip install numpy matplotlib
```

3. Run the main script:

```shell
python All_cp.py
```

4. The generated audio file, `output.wav`, will be saved in the project directory.

## Configuration

The genetic algorithm parameters can be modified in the `melody_generation.py` script:

- `population_size`: The size of the population (default: 100).
- `melody_length`: The length of the melodies (default: length of the reference melody).
- `num_generations`: The number of generations to run the genetic algorithm (default: 500).

The reference melody can be modified by changing the `reference_melody` list in the `melody_generation.py` script.

## Results

The best melody found by the genetic algorithm, along with its fitness score, will be displayed in the console output. The convergence plot of the best fitness scores over generations will be shown after the algorithm completes.

The generated audio file, `output.wav`, contains the synthesized version of the best melody.

## License

This project is licensed under the [MIT License](LICENSE).
```
