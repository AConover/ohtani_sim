# Ohtani Sim

This repository contains a Python script designed to simulate the remainder of Shohei Ohtani's 2024 MLB season. The simulation estimates the probabilities of Ohtani reaching various performance milestones, such as joining the 30-30, 40-40, or 50-50 clubs for home runs and stolen bases. The results are visualized in a GIF that shows the progression of these probabilities throughout the season.

## Features

- **Bayesian Updating**: The simulation employs a Bayesian framework with a Gamma-Poisson model to update the probability distribution of Ohtani's home runs and stolen bases based on his performance to date.
- **High-Fidelity Simulations**: The code runs 1 million simulations per game to provide a robust estimate of Ohtani's performance.
- **Visualization**: The code generates a GIF that visualizes the simulated progression of Ohtani's home runs and stolen bases, along with the probabilities of achieving the 30-30, 40-40, and 50-50 milestones.

## Requirements

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)

## Acknowledgements
 Statistics were pulled from ESPN at https://www.espn.com/mlb/player/gamelog/_/id/39832/shohei-ohtani.
