# Qubo formulation of Feature Selection Problem

This repository contains a Qubo formulation problem to assess **Feature Selection** in classical Machine Learning frameworks.
It contians files to handle a structured dataset (*breast cancer* dataset provided by *Scikit-Learn* is used), to create and evaluate **Redundancy** and **Importance** matrices originated from the **Mutual and Conditional Mutual Information** between features and labels. Subsequently, the symmetric $n x n$ Qubo matrix $Q$ is constructed. 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)

## Background

Quantum computing aims of assessing unsolved or hard-to-be-solved in a polynomial time (NP-hard) computational problems such as combinatorial optimization problems (**COPs**) as *Maximum Cut Problem (MaxCut)*, *Travel Salesman Problem (TSP)*, *Knapstack Problem* and so on. One of these COPs which has a large impact on Machine Learning frameworks is the **Feature Selection Problem**. It consist of selecting a maximum number of features in a given dataset by minimizing and maximizing the Redundancy and Importance matrices respectively. One of the technique we could use to address this exponentially large problem (which depends on the number of features present in the dataset and by the number of samples in it) is o use *Quantum Annealer* approach, by mapping the problem into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.
Going directly to the mathematical formulation of the problem, we have a dataset $S$ which contains $N$ samples characterized by $M$ features each as it follows $S = {x^j}$ 
$Q = $


- What was your motivation?
- Why did you build this project? (Note: the answer is not "Because it was a homework assignment.")
- What problem does it solve?
- What did you learn?
