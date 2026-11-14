# BR-DRAG: Byzantine-robust divergence-based adaptive aggregation

Code and notes for the BR-DRAG part of the reproduction (Phase 1).

Base paper: Xiao et al., *Divergence-Based Adaptive Aggregation for Byzantine Robust Federated
Learning*, arXiv 2601.06903, Section IV, eqs. (12)-(16). Settings follow Section VI-B.

## 1. Idea

Plain FedAvg averages whatever the clients send, so a few malicious clients can push the model
anywhere. BR-DRAG gives the server a small trusted dataset, the **root set** `D_root`. Every
round the server trains briefly on it to get a **reference direction** `r`, then pulls each
client update towards `r`. The further an update points away from `r`, the more it gets pulled.

Two things make this robust:

- **Norm control.** Every update is rescaled to the length of `r` before mixing, so an attacker
  cannot win by sending a huge vector.
- **Direction control.** The mixing weight `λ` grows with the angle between the update and `r`,
  so updates that disagree with the trusted direction lose most of their own direction.

Unlike DRAG, the correction happens on the **server**. Clients upload their raw update.

## 2. Notation

| Symbol | Meaning | Default |
|---|---|---|
| `M` | number of clients | 40 |
| `S` | clients selected per round | 10 |
| `U` | local SGD steps (clients and root trainer) | 5 |
| `B` | minibatch size | 10 |
| `η` | learning rate | 0.01 |
| `θ^t` | global model at round `t` (flat parameter vector) | |
| `g_m` | raw update of client `m`, `θ_m^(t,U) − θ^t` | |
| `D_root` | root set, class-balanced | 3000 samples |
| `r^t` | reference direction from `D_root` | |
| `c_t` | divergence scale | 0.5 |

## 3. One round

**Step 1: reference direction (server), eqs. (12)-(13).**
Start from `θ^t`, run `U` plain SGD steps on minibatches from `D_root`:

```
θ^(t,0) = θ^t
θ^(t,u+1) = θ^(t,u) − η ∇f(θ^(t,u); batch from D_root)      u = 0 .. U−1
r^t = θ^(t,U) − θ^t
```

**Step 2: broadcast.** The server sends `θ^t` and `r^t` to the selected clients.

**Step 3: local training (clients).** Each selected client runs `U` SGD steps on its own data and
uploads the raw update `g_m = θ_m^(t,U) − θ^t`. Malicious clients may replace it (Section 6).

**Step 4: divergence weight (server), eq. (16).**

```
λ_m = c_t · (1 − cos(g_m, r^t))          λ_m ∈ [0, 2 c_t]
```

`λ = 0` when `g_m` points exactly along `r`, `λ = c_t` when it is orthogonal, `λ = 2 c_t` when
it points the opposite way.

**Step 5: corrected update (server), eq. (15).**

```
v_m = (1 − λ_m) · (‖r^t‖ / ‖g_m‖) · g_m  +  λ_m · r^t
```

The first term is `g_m` rescaled to the length of `r`; the second pulls it towards `r`. For
`c_t ≤ 0.5`, `λ ≤ 1`, so `v_m` is a convex mix of two vectors of length `‖r‖` and `‖v_m‖ ≤ ‖r‖`.

**Step 6: aggregate and apply, eq. (14).**

```
Δ^t = (1/S) Σ_m v_m
θ^(t+1) = θ^t + Δ^t
```

### Pseudocode

```python
for t in range(T):
    S_t = sample S of M clients
    r = root_sgd(theta, D_root, U, B, lr) - theta          # eqs. (12)-(13)
    g = {m: local_sgd(theta, D_m, U, B, lr) - theta for m in S_t}
    g = attack(g)                                           # simulator only
    v = []
    for m in S_t:
        lam = c_t * (1 - cos(g[m], r))                      # eq. (16)
        v.append((1 - lam) * norm(r) / norm(g[m]) * g[m] + lam * r)   # eq. (15)
    theta = theta + mean(v)                                 # eq. (14)
```

### Edge cases handled in code

- If `‖g‖` or `‖r‖` is below `1e-8`, `cos` is taken as 0, so `λ = c_t`.
- `cos` is clamped to `[−1, 1]` so rounding can never push `λ` outside `[0, 2 c_t]`.
- `g = 0` gives `v = c_t · r`.

## 4. BR-DRAG vs DRAG

| | DRAG | BR-DRAG |
|---|---|---|
| Reference `r` | momentum of past global updates, `r ← (1−α) r + α Δ` | `U` SGD steps on the trusted root set, recomputed every round |
| Where the correction runs | client | server |
| Rescaling | `r` rescaled to `‖g‖` | `g` rescaled to `‖r‖` |
| Main goal | client drift on non-IID data | Byzantine robustness |
| Needs a warm-up round | yes (`r^0` = mean of first uploads) | no |

## 5. Baselines

- **FedAvg**, eq. (4): `θ^(t+1) = θ^t + (1/S) Σ g_m`.
- **FLTrust, paper version** (`fltrust_paper`, base paper p. 9):
  `g̃_m = max(0, cos(g_m, r)) · ‖g_m‖ · r / ‖r‖`, then the unweighted mean.
- **FLTrust, original** (`fltrust_cao`, Cao et al., NDSS 2021):
  `trust_m = ReLU(cos(g_m, r))`, `ĝ_m = (‖r‖ / ‖g_m‖) g_m`,
  `Δ = Σ trust_m ĝ_m / Σ trust_m` (zero if all trust is 0).

Both FLTrust versions use the same root set and the same `r` as BR-DRAG.

## 6. Attacks

A fixed set of `⌊ρ M⌉` clients is malicious for the whole run (`ρ` = 0.3 or 0.6).

| Attack | Level | What the malicious client uploads |
|---|---|---|
| Noise | update | `p · g`, scalar `p ~ N(0, 3)` (read as variance 3) |
| Sign flip | update | `−g` |
| Label flip | data | trains honestly on data where half the labels are changed to `L − l − 1` |
| Min-Max | update | `ĝ = ḡ_b + γ p`, `p = −ḡ_b / ‖ḡ_b‖`, largest `γ` with `max_i ‖ĝ − b_i‖ ≤ max_{i,j} ‖b_i − b_j‖` |
| Min-Sum | update | same `ĝ`, largest `γ` with `Σ_i ‖ĝ − b_i‖² ≤ max_k Σ_i ‖b_k − b_i‖²` |

`ḡ_b` and `b_i` are the benign updates selected in that round (the attacker is omniscient). All
attackers send the same `ĝ` in Min-Max and Min-Sum. `γ` is found by bisection and logged.

## 7. Root set

`floor(3000 / L)` samples per class are drawn at random from the training split **before** the
data is split across clients, and removed from client data. For EMNIST balanced (`L = 47`) this
gives 2961 samples. The paper vets the pool with confident learning; that step is available
(`--vet-root`, needs cleanlab) but was off in all our runs.

## 8. Where it is in the code

| File | Contents |
|---|---|
| `src/fedrag/strategies/br_drag.py` | `cosine`, `divergence` (eq. 16), `correct_update` (eq. 15), `aggregate` / `apply` (eq. 14), `BRDRAG` strategy class |
| `src/fedrag/strategies/root_trainer.py` | `root_reference`: eqs. (12)-(13) |
| `src/fedrag/strategies/fltrust.py` | both FLTrust versions |
| `src/fedrag/attacks/` | `noise`, `sign_flip`, `label_flip`, `min_max`, `min_sum` |
| `src/fedrag/data/root.py` | class-balanced root set |
| `src/fedrag/runner/run.py` | experiment loop, metrics, checkpoints |
| `tests/strategies/test_br_drag.py` | equations checked against hand-worked examples |

`Downlink` / `Uplink` in `br_drag.py` are temporary stand-ins until `core/messages.py` is merged.

Per-round diagnostics logged to `metrics.csv`: `lambda_mean/min/max`, `cos_g_r_mean`, `r_norm`,
`g_norm_mean`, `rho_min/max` (`‖r‖ / ‖g‖`), and server timings.

## 9. Running

```bash
pip install -r requirements.txt
pytest
PYTHONPATH=src python -m fedrag.runner.run --dataset emnist_balanced --beta 0.5 \
    --method br_drag --attack sign_flip --malicious 0.6 --rounds 1000 --seed 0
```

Methods: `fedavg`, `br_drag`, `fltrust_paper`, `fltrust_cao`.
Attacks: `none`, `noise`, `sign_flip`, `label_flip`, `min_max`, `min_sum`.
Datasets: `emnist_balanced`, `cifar10`, `cifar100`. `--c-t` sets `c_t`.

## 10. Choices the paper leaves open

- CNN widths (we use the classic FedAvg CNNs; EMNIST 601k, CIFAR-10 2.16M, CIFAR-100 1.19M params).
- `N(0, 3)` read as variance 3.
- Min-Max / Min-Sum direction `p = −ḡ_b / ‖ḡ_b‖`; Min-Sum uses squared distances.
- Root set removed from client data; no vetting by default.

## 11. What we observed (Phase 1)

- BR-DRAG stays usable where FedAvg collapses (e.g. EMNIST, sign flip at 60%: about 70% vs 2%).
- Final accuracies are below the paper's, most on CIFAR-100, and the original FLTrust beats
  BR-DRAG in most settings.
- In the logs `cos(g, r)` stays close to 0, so `λ ≈ c_t` every round, and `‖r‖` shrinks late in
  training. Since every update is rescaled to `‖r‖`, progress slows and test loss rises.
- Raising `c_t` to 1.0 recovered most of the gap on EMNIST Min-Max at 60% (65.5% to 78.5% at 1000 rounds).

## 12. Note for the DP phase

In the current architecture, update-level attackers skip the DP stage. Under DP, honest uploads
are noisy while attacker uploads are clean, and Min-Max / Min-Sum are built from an average of
honest uploads in which most of the noise cancels. BR-DRAG could then trust attackers more than
honest clients. Proposed: a config flag `attack.applies_dp: true | false`, with both settings
tested once the DP stage exists.
