# Qubo formulation of Feature Selection Problem

<img src="/imgs/feature_sel_ml.jpg" width="1024"/>

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


Going directly to the mathematical formulation of the problem, we start from a dataset $S$ which contains $N$ samples characterized by $M$ features each as follows $S = \{x_{i=1, ..., N}^{j=1, ..., M}\}$,
then we can construct a symmetric Qubo matrix as


$Q = -R(k) - I(k) + \lambda(\sum_{i}^{M} x_i - k )^2$ where $k \in M$,

$R_{i,j} = H(X|Z) + H(Y|Z) - H(X,Y|Z)$ for $i \neq j$,

$I_i = H(X,y) - H(X|y) - H(y|X)$ for $i = j$,


where $R$ is known as Redundancy matrix and computes the conditional mutual information between a feature $X_i$ and a label $y$ knowing a feature $Z$, while the Importance matrix is actually a vector that represents the mutual information between a feature $X_i$ and the label $y$. 

Before getting the solution out of the Qubo matrix, we decided to add an extra term, known as **penalty term**, which takes into account that the solution must contain exactly *k* features for being considered a valid solution. This penalty term can be formulated as

$P = \lambda(\sum_i x_i - k)^2 = \lambda(\sum_i x_i - 2k\sum_{i \neq j} x_i x_j)$ 

This is how the Qubo matrix looks like with a tuning parameter $\lambda = 2.8$


<img width="512" src="/imgs/qubo_matrix.png" align="left" />

<div align="right">
As we can see, the Qubo matrix has negative values on the diagonal corresponding to the Importance matrix while values approaching 0 to the off-diagonal elements which results in a way of penalizing Redundancy while maximizing the Importance.
</div>

Afterwards, once we have constructed the Qubo matrix $Q$ we encode the problem into a Dwave quantum annealing or a classical simulated annealing and we solve this last equation:

$$ \arg\min_x \mathbf{x}^TQ\mathbf{x} $$

which will lead to a bitstring $\mathbf{x}$ that is our solution. In this context, selected features are represented as $1$ in the solution, while the discarded features are labeled as $0$.

### Motivation
- What was your motivation?
- Why did you build this project? (Note: the answer is not "Because it was a homework assignment.")
- What problem does it solve?
- What did you learn?
