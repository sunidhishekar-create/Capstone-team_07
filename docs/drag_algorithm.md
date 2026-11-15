\# DRAG Algorithm — Implementation Documentation



\## 1. Overview



This project implements the original DRAG algorithm described in \*Divergence-Based Adaptive Aggregation for Byzantine Robust Federated Learning\* (arXiv: 2601.06903).



DRAG modifies federated client updates using a reference direction and a divergence-based mixing coefficient before server aggregation.



\*\*Scope:\*\* This implementation covers original DRAG only. BR-DRAG is not implemented.



\## 2. Federated Learning Workflow



At each communication round:



1\. The server maintains the current global model.

2\. A subset of clients is selected uniformly at random.

3\. Each selected client trains its local model using local data.

4\. Each client returns its parameter displacement.

5\. The server calculates the divergence-based mixing coefficient.

6\. Each client update is modified using the reference direction.

7\. The server averages the modified updates and updates the global model.

8\. The reference direction is updated for the next round.



\## 3. Client Update



Let θᵗ denote the global model parameters at round t, and let θₘᵗ denote the parameters obtained by client m after local training.



The client displacement is:



gₘᵗ = θₘᵗ − θᵗ



This represents the change in model parameters caused by local training.



In the implementation, this displacement is returned by `local\_update` in `src/fl/client.py`.



\## 4. Reference Direction



The reference direction represents the direction used to compare client updates.



The implementation initializes the reference direction from the mean of the selected clients' initial local updates:



r⁰ = mean(gₘ⁰)



The subsequent reference direction is updated using:



rᵗ = (1 − α)rᵗ⁻¹ + αΔᵗ⁻¹



where:

\- α controls the reference update.

\- Δᵗ⁻¹ is the previous aggregated update.



The corresponding implementation is `ReferenceDirection.initialize` and `ReferenceDirection.advance` in `src/algorithms/drag.py`.



\## 5. Divergence-Based Mixing Coefficient



DRAG measures the directional difference between a client update g and reference direction r using cosine similarity.



The implementation computes:



λ = c(1 − cos(g, r))



where c is a scaling parameter.



Cosine similarity is clamped to the interval \[-1, 1]. The implementation also handles near-zero vector norms to avoid numerical errors.



The mixing coefficient determines the relative contribution of the client update and the reference direction.



Implementation: `divergence\_of\_directions` in `src/algorithms/drag.py`.



\## 6. Modified Client Update



The implementation modifies each client update using:



v = (1 − λ)g + λ(||g|| / ||r||)r



Here:

\- g is the client displacement.

\- r is the reference direction.

\- λ is the divergence-based mixing coefficient.

\- ||g|| and ||r|| are the corresponding vector norms.



The reference direction is scaled to account for the client update's norm. If the reference norm is near zero, the implementation falls back to the original client update.



Implementation: `drag\_modify` in `src/algorithms/drag.py`.



\## 7. Server Aggregation



For S selected clients, the server calculates the mean of their modified updates:



Δᵗ = (1/S) Σ vₘᵗ



The global model is updated as:



θᵗ⁺¹ = θᵗ + Δᵗ



Because client updates are represented as parameter displacements, the aggregated displacement is added to the global parameters.



Implementation: `aggregate` in `src/algorithms/drag.py` and the training loop in `src/fl/trainer.py`.



\## 8. Experimental Configuration



The completed experiments used the following configuration:



\- Total clients: 40

\- Selected clients per round: 10

\- Local SGD steps: 5

\- Batch size: 10

\- Learning rate: 0.01

\- Reference update parameter α: 0.25

\- Datasets: EMNIST-Balanced, CIFAR-10, CIFAR-100

\- Tested β values: 0.1 and 0.5



The configured c values were 0.25 for β = 0.1 and 0.1 for β = 0.5.



EMNIST experiments ran for 3,000 rounds, while CIFAR-10 and CIFAR-100 experiments ran for 4,000 rounds.



\## 9. Experimental Results



Six experiment configurations completed. Their final test accuracies were:



| Dataset | β | Final test accuracy |

|---|---:|---:|

| EMNIST | 0.1 | 87.16% |

| EMNIST | 0.5 | 88.07% |

| CIFAR-10 | 0.1 | 74.59% |

| CIFAR-10 | 0.5 | 73.53% |

| CIFAR-100 | 0.1 | 45.26% |

| CIFAR-100 | 0.5 | 40.97% |



The best observed final accuracy for EMNIST was obtained with β = 0.5. For CIFAR-10 and CIFAR-100, the higher final accuracy was obtained with β = 0.1.



These values represent one run per configuration and do not establish statistical significance.



The detailed metrics, learning curves, and checkpoints are stored in the project's Google Drive folder.



\## 10. Implementation Assumptions and Limitations



The implementation documents several assumptions where the paper does not fully specify the experimental details:



\- The training-round counts follow the x-axis extents of the paper's figures.

\- The initial reference-direction pass is separate from the subsequent training pass in the first round.

\- Near-zero vector norms receive special handling for numerical stability.

\- Dataset preprocessing and model architecture choices are documented in the project README.

\- Each configuration was run with one random seed.



The results therefore represent an experimental evaluation of the current implementation, not proof of exact reproduction of every paper setting.



\## 11. Related Work



The implementation is based on:



\*Divergence-Based Adaptive Aggregation for Byzantine Robust Federated Learning.\*



Refer to the original paper for the complete algorithm, equations, and theoretical analysis.



