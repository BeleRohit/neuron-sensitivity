# Project: Neuron Sensitivity — Weight-Space Perturbations

Build a small, research-oriented PyTorch project investigating how modifying individual neurons' weights affects a trained neural network's predictions.

**Objective:** Understand neuron-level sensitivity, robustness, and the relationship between parameter magnitude and functional importance. This is an educational experiment, not a claim of novel research.

### 1. Model and dataset
- Use MNIST with a simple MLP: `784 → 128 → 64 → 10`.
- Train using PyTorch with cross-entropy loss and Adam.
- Support CPU and Apple Silicon MPS.
- Save the trained model, baseline test accuracy, and training metrics.
- Use fixed random seeds for reproducibility.

### 2. Weight-space experiments
Without retraining the model:

- Perturb the incoming weight vector of one hidden neuron at a time.
- Test multiple perturbation strengths.
- Compare Gaussian noise, weight scaling, and neuron ablation.
- Repeat across both hidden layers.
- Restore the original model weights before every independent experiment.
- Normalize perturbations relative to the original weight norm for fair comparisons.
- Keep biases unchanged for noise/scaling experiments; document that ablation must also eliminate the neuron's effective contribution.

### 3. Measurements
For every perturbation, record:

- Test accuracy before and after modification.
- Number and percentage of prediction flips.
- Mean absolute change in predicted-class probability.
- Perturbation magnitude.
- Neuron weight norm.
- Layer and neuron index.

Repeat stochastic perturbations across multiple seeds. Use the same unmodified test set for every comparison.

### 4. Analysis
Investigate:

1. Which neurons are most sensitive to perturbations?
2. Are first-layer or second-layer neurons more sensitive?
3. Does sensitivity increase with perturbation magnitude?
4. Does a neuron's weight norm predict its functional importance?
5. Do neurons with similar weight norms produce different prediction changes?

Generate:
- Neuron-sensitivity heatmaps.
- Accuracy versus perturbation strength.
- Weight norm versus sensitivity scatter plots.
- Layer-wise sensitivity comparisons.

### 5. Engineering requirements
- Use a clean, modular Python project.
- Include unit tests for perturbation, restoration, reproducibility, and metric calculations.
- Never modify the baseline checkpoint in place.
- Keep experiment configurations and results reproducible.
- Use NumPy, PyTorch, pandas, and Matplotlib.
- Avoid unnecessary frameworks, cloud infrastructure, and complex abstractions.
- Run experiments locally on my MacBook Air M2.

### 6. Deliverables
- Complete runnable implementation.
- `README.md` with setup and execution instructions.
- `EXPERIMENT_REPORT.md` explaining methodology, results, limitations, and findings.
- Saved experimental metrics in CSV format.
- Publication-quality plots.
- A Jupyter notebook explaining the experiments step by step.

**Important:** Implement and execute the experiments, not just create a project skeleton. Report actual measured results, including negative findings. Do not fabricate results or claim that weight magnitude alone determines neuron importance.

Keep the project small, understandable, and focused on developing first-principles intuition about neural-network weight space.