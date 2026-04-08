# Added mass force

### Abstract

The prediction of added mass force during fluid–structure interaction problems, such as water entry, is computationally expensive using traditional numerical methods. This work presents a machine learning approach using neural networks to model and predict added mass forces based on input flow and geometric parameters. A deep neural network (multi-layer perceptron) is trained on simulated or experimental data to learn the nonlinear relationship between input features and resulting hydrodynamic forces. The model achieves high accuracy and demonstrates the potential of data-driven approaches for fast and efficient prediction in complex fluid dynamics problems.

### Motivation

In problems such as water entry, slamming, and fluid-structure interaction (FSI), the concept of added mass plays a crucial role. Added mass represents the additional inertia a body experiences due to the surrounding fluid.

#### However:

Traditional CFD simulations (e.g., Navier–Stokes solvers) are:
  1) Computationally expensive
  2) Time-consuming for parametric studies
Experimental measurements:
  1) Are difficult for transient and high-speed events
  2) Require sophisticated setups

#### Therefore, there is a strong need for:

  1) Fast surrogate models
  2) Real-time prediction capability
  3) Reduced computational cost

Neural networks provide:

  1) Nonlinear mapping capability
  2) Ability to learn complex fluid dynamics behavior
  3) Fast inference once trained
