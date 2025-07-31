#  EvoTrace



---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Folder Structure](#folder-structure)

---

## Overview

**EvoTrace** is designed to track and log various operations in a genetic algorithm (GA), 
providing insights into how the algorithm evolves and behaves over generations.

---

## Installation

Follow these steps to install and set up the **Genetic Algorithm Trace Generator** on your local machine:

### 1. Clone the Repository
First, clone the repository to your local machine using Git:

```bash
git clone https://github.com/matteobonani/EvoTrace.git
```
### 2. Navigate to the Project Directory
Once the repository is cloned, navigate into the project directory `cd EvoTrace`.

### 3. Set Up a Virtual Environment (Optional but Recommended)

I suggest to use a virtual environment to manage project dependencies.

### 4. Install the Dependencies
Once the virtual environment is activated, install the required dependencies using pip:

```bash
pip install -r requirements.txt
```
---

## Usage

### 1. Running the Genetic Algorithms (GAs)

You can execute the genetic algorithm using either a Python script or a Jupyter notebook, depending on your preferred workflow.
#### Option A: Python Script

Modify the configuration parameters directly inside the script `test/GA_runner.py`.
Then, run the script using:
```bash
cd test
python GA_runner.py
```
You will be prompted to enter the number of optimization runs. Results will be saved in a timestamped folder inside the results/ directory.
#### Option B: Jupyter Notebook (Recommended for Exploration)

For a more interactive experience, open the notebook `test/GA_runner_demo.ipynb`.
You can modify all configuration parameters directly in the notebook cells (e.g., population size, mutation operator, Declare model). This option is ideal for experimentation and quick debugging.

---

### 2. Configuration Guidelines

#### 2.1 Problem Types

Two types of optimization problems are currently supported:

- ProblemSingle: for single-objective optimization.

- ProblemMulti: for multi-objective optimization (e.g., NSGA-II).

Available problem definitions can be found in:

- `ga_objects/problems/single_objective_problems.py`

- `ga_objects/problems/multi_objective_problems.py`

#### 2.2 Trace Lengths

The number of events per trace (trace_length) must be selected from the following predefined lengths:
30, 50, 70, 90
Each length corresponds to a pre-generated initial population CSV file, stored in the respective model folder:
`declare_models/model*/initial_pop_*.csv`

For example: `declare_models/model2/initial_pop_50.csv`.
The first * represents the model number (e.g., model1, model2, etc.), and the second * represents the trace length.

#### 2.3 Custom Trace Lengths (Optional)
To use a custom trace length not included in the default set (30, 50, 70, 90), follow the steps in section 3 below.

---

### 3. Generating Initial Populations Using ASP (Optional)
Initial populations used by the GA are generated via an Answer Set Programming (ASP) based trace generator. This step is required only if you want to use a custom trace length that is not pre-generated.
#### Option A: Python Script

Edit the configuration in `test/ASP_runner.py`, then run the script:
```bash
cd test
python ASP_runner.py
```
#### Option B: Jupyter Notebook

To generate custom populations interactively, open `test/ASP_runner_demo.ipynb`.
This notebook allows you to configure the number of events, Declare model, and population size directly from within the interface.
> **Note:**  For GA initialization, it is sufficient to generate a small ASP population of size 10. The genetic algorithm will use this as the seed population.

---

## Folder Structure

```text
.
├── declare_models/                   # models for experiments
│   ├── model1/
│   ├── model2/
│   ├── model3/
│   └── model4/
│
├── ga_objects/                       # core GA logic and components
│   ├── operators/                    # GA operators (e.g., selection, crossover, mutation)
│   ├── problems/                     # problem definitions for optimization
│   └── utils/                        # helper utilities and support functions
│
├── test/                             # test scripts and experiments
│   ├── results/                      # output results from test runs
│   ├── ASP_runner.py                 # runner script for ASP-based trace generation
│   ├── GA_runner.py                  # runner script for Genetic Algorithm
│   ├── ASP_runner_demo.ipynb         # notebook version of ASP runner
│   ├── GA_runner_demo.ipynb          # notebook version of GA runner
│   ├── mainTest.ipynb                # main testing notebook
│   └── singleTest.ipynb              # notebook for a single test scenario
│
├── .gitignore
├── README.md                         # project documentation
└── requirements.txt                  # python dependencies
```
