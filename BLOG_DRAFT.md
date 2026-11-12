# What does it mean for a neuron to matter?

*A small MNIST experiment on why weight size alone cannot tell the whole story.*

When we talk about a neural network neuron as “important,” we often reach for one number: the size of its weights. That is convenient, but a parameter norm says only how long a vector is. It does not say which inputs activate the neuron, how often it fires, which downstream paths use its output, or whether its contribution matters to the model’s current decisions.

To make that distinction concrete, I trained five small MNIST classifiers with the same architecture: 784 inputs, hidden layers of 128 and 64 ReLU units, and 10 output logits. Each model was trained for five epochs with Adam and cross-entropy. I then removed one hidden neuron at a time by setting its incoming weights and bias to zero, and measured how many of the same 10,000 test predictions changed.

## Similar-sized vectors, different interventions

The first model offered a clear example. First-layer neuron 104 had an incoming weight norm of 2.987; neuron 122’s was 3.006. Their norms differed by less than one percent. Yet ablating neuron 104 changed 342 test predictions (3.42%), while ablating neuron 122 changed 61 (0.61%).

The difference came from the network’s behavior, not the length of the input vectors. Neuron 104’s ablation affected many predictions for the digit 1: 295 of the model’s 1,141 baseline predictions for that class changed. For those examples, removing the neuron sharply reduced the class-1 decision margin. The norm alone did not describe that relationship between inputs, activation, downstream computation, and the final decision.

This one-network example could have been a fluke, so I repeated the training and ablation process with five independent seeds. The numeric index 104 was not consistently the most sensitive neuron. Hidden units can take on different roles, and their indices are not identities shared across separately trained networks. But the broader pattern did recur: within each model, many neuron pairs had incoming norms within one percent while their ablation flip rates differed by more than half a percentage point.

## Predicting before removing

The next question was whether any pre-ablation properties could predict the damage an ablation would cause. I measured incoming and outgoing weight norms, activation frequency, mean activation, and a local estimate of each neuron’s effect on output logits. I kept the target separate: prediction-flip rate came from an exact forward pass after ablation.

For a second-layer neuron, its activation multiplied by its outgoing output weights gives its additive contribution to each logit at the baseline activation. For a first-layer neuron, another ReLU sits between it and the output. I therefore used a local Jacobian that includes the second layer’s active ReLU gates. That first-layer measure is a local approximation: a finite ablation can change those gates. It is useful as a pre-ablation feature, but it is not the same as the exact ablation result.

Across the five models, incoming norm was positively related to sensitivity, but it was not the strongest single feature. The local logit contribution had the highest average Spearman correlation with flip rate: 0.837 in the first layer and 0.820 in the second. Activation frequency mattered more in the second layer (0.593) than in the first (0.075), another sign that the same feature need not describe both layers equally well.

I also tested a simple five-feature ridge model. To check whether it generalized beyond neurons in the same network, I held out an entire trained model at a time. The combined predictor achieved mean held-out Spearman correlations of 0.835 for the first layer and 0.820 for the second. A norm-only predictor reached 0.588 and 0.442. The combined model also lowered mean absolute error in both layers.

## What this does and doesn’t show

The result is not that weight norms are useless. They carried predictive signal in both layers. The result is that weight length by itself omits information. Activation patterns and downstream pathways helped rank which neurons would change predictions when removed.

This remains a bounded experiment: five seeds, one small MLP, five training epochs, and one fixed dataset. The local logit contribution is approximate in the first layer, and the predictor was evaluated on new model seeds—not on a new dataset or a shifted distribution. Neural units can also be redundant, so an ablation measures the effect of removing one unit in one trained network, not an intrinsic value that survives retraining.

For this setting, “importance” is best treated as a relationship between a neuron, the inputs it sees, the paths that use its activation, and the decision boundary. A parameter norm is one part of that picture. It is not the whole picture.

*Methods and fold-level measurements: [`PHASE2_REPORT.md`](PHASE2_REPORT.md). Code, checkpoints, CSVs, and figures are in the project repository.*
