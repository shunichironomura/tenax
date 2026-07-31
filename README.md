# Tenax

Tenax is a software tool for robust decision making analyses.

## Vision

The ivsion for Tenax is that it will become a software toolbox composed of:

- Rust library/toolbox that implements core, common analyses in robust decision-making such as PRIM, CART, etc.
- Rust CLI binary that runs the analyses against a model server that speaks the protocol (specifics of the protocol are to be determined)
- Python library that wraps the core Rust library/toolbox using PyO3 that users can easily run analysis in Python scripts and notebooks.

The differenciator from the existing libraries such as EMA workbench includes:

- Performance due to the Rust implementation
- Incremental, parallel, anytime algorithm for faster iteration of decision makers

## Roadmap

1. Implement the conventional analyses methods such as PRIM and ensure the correctness of the implementation by comparing the results with existing solutions such as EMA workbench. This enables the decoupling of analyses methods implementation and the rest such as the protocol
2. Define the protocol that will be the foundation of the future anytime algorithm implementation but also is compatibile with batch analyses like conventional PRIM analysis.
3. Implement the template protocol server implementation for Python and Rust.
4. Implement the analyzer client that performs the conventional analyses methods that were implemented in the step 1.
5. Implement the anytime algorithm.

## What is NOT in scope

- Fast visualization of millions of data points. We can resort to dedicated libraries like `datashader` and `xy` (from Reflex dev).
