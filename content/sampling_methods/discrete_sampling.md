(chap:sampling_methods_discrete_sampling)=
# Discrete sampling

+++
Many tasks within deep learning involve sampling over discrete spaces. Such sampling tasks are distinct from sampling over continuous spaces: while the primary frameworks of sampling—Markov chain Monte Carlo (MCMC), rejection sampling, and importance sampling—introduced in Chapter [](#chap:sampling:intro) can be applied to discrete spaces by adjusting the proposal distribution, this adjustment is often highly non-trivial. In this chapter, we focus on high-dimensional targets whose unnormalised log density can be evaluated but whose state spaces are too large to enumerate. Such targets arise in binary energy-based models, categorical latent-variable models, and neural models over combinatorial structures.

## Introduction

+++
Discrete sampling is notably difficult due to the sparse, discontinuous, and combinatorial nature of discrete spaces. If a state consists of {math}`d` categorical variables with {math}`K` possible values each, its state space may contain {math}`K^d` configurations. In principle, one could evaluate the unnormalised density at every configuration, normalise the resulting weights, and sample from the corresponding categorical distribution. This enumeration-based approach becomes infeasible as the dimension grows. Standard local MCMC methods avoid full enumeration, but target-independent proposals can mix slowly because they do not distinguish promising moves from moves into low-density regions.

As a running example, consider a binary energy-based model with {math}`x\in\{0,1\}^d` and unnormalised log density {math}`U(x)`. The model assigns a score to each of its {math}`2^d` possible configurations, making enumeration infeasible for large {math}`d`. Nevertheless, {math}`U(x)`, and often the gradient of a differentiable extension of {math}`U`, can be evaluated at any given configuration. We use this binary model to illustrate the methods below, although the framework also applies to the broader categorical and combinatorial settings described above.

Gradients are a staple of sampling within deep learning, as illustrated by the continuous-space Langevin methods in [Section {name}](#sec:mala). Although a function defined only on a discrete space has no intrinsic gradient, many deep-learning targets are represented through a differentiable function on a continuous ambient space and evaluated at binary, categorical, or other discrete states. The gradient of this extension can provide a cheap source of local information about promising discrete moves.

We begin with an uninformed local MCMC proposal and show how pointwise-informed and locally balanced proposals incorporate target-specific information. We then use this framework to develop gradient-based discrete samplers, first for single-coordinate moves and then for parallel multi-coordinate moves. Finally, we discuss methods designed for larger-scale exploration.

## Notation

+++
We assume a discrete state space {math}`\mathcal{X}`. We denote a state by {math}`x=(x_1,\dots,x_d)\in\mathcal{X}`, where each coordinate can take one of {math}`K` possible values. Throughout the chapter, {math}`\pi` denotes the normalised target distribution and {math}`U` its unnormalised log density. Thus our target is

```{math}
:label: eq:disc-sampling:target

\pi(x) = \frac{1}{Z}\exp(U(x)), \ \ Z = \sum_{x' \in \mathcal{X}} \exp U(x')
```

The normalising constant or partition function {math}`Z` in Equation [](#eq:disc-sampling:target) generally requires a sum over as many as {math}`K^d` states. This makes enumeration-based direct sampling computationally infeasible in high dimensions. MCMC can nevertheless operate on differences such as {math}`U(x')-U(x)`, in which {math}`Z` cancels.

## Locally Balanced Proposals

+++
A simple Metropolis–Hastings approach proposes a nearby state using a target-independent kernel {math}`K_\sigma(x'\mid x)`. For example, the kernel might choose a coordinate uniformly and replace its value. Such a proposal is inexpensive, but it assigns the same proposal probability to moves towards high- and low-density states. Large moves may explore quickly but often reach low-density states and are rejected; small moves are more likely to be accepted but can produce highly correlated samples and traverse the state space slowly.

For the running binary example, the simplest local kernel selects one of the {math}`d` bits uniformly and flips it, without using the values of {math}`U` to prioritise the available moves.

{cite:t}`zanella2020informed` introduces a framework that addresses this trade-off by weighting the candidate moves from {math}`K_\sigma` using target-density information. The resulting pointwise-informed proposal is

```{math}
:label: eq:disc-sampling:pointwise-informed

P_{g,\sigma}(x'\mid x)
=\frac{g\!\left(\frac{\pi(x')}{\pi(x)}\right)K_\sigma(x'\mid x)}
{Z_{g,\sigma}(x)}.
```

The pointwise-informed proposal in Equation [](#eq:disc-sampling:pointwise-informed) consists of the target-density ratio, a symmetric base kernel or weight {math}`K_\sigma` whose scale controls the size of candidate moves, a balancing function {math}`g` that weights moves using the density ratio, and the state-dependent normaliser {math}`Z_{g,\sigma}(x)`. The base kernel may be specified only up to a state-independent multiplicative constant, because that constant cancels through {math}`Z_{g,\sigma}(x)`. We assume {math}`K_\sigma(x'\mid x)=K_\sigma(x\mid x')`. The limit {math}`\sigma\to0` represents a family of kernels that becomes increasingly concentrated at the current state.

The roles of invariance and ergodicity in MCMC are reviewed in [Section {name}](#sec:mcmc), while detailed balance and the Metropolis–Hastings construction are discussed in [Section {name}](#sec:mh). In the present setting, {math}`P_{g,\sigma}` is reversible with respect to {math}`\pi_{g,\sigma}` if it satisfies detailed balance,

```{math}
\pi_{g,\sigma}(x)P_{g,\sigma}(x'\mid x)
=\pi_{g,\sigma}(x')P_{g,\sigma}(x\mid x').
```

Detailed balance is sufficient for {math}`\pi_{g,\sigma}` to be invariant under the transition. Invariance alone does not ensure convergence from an arbitrary initial state. For a discrete chain, irreducibility means that every state with positive target probability can eventually be reached from every admissible initial state, while aperiodicity means that the chain is not constrained to return according to a fixed cycle. Together with invariance and the relevant regularity conditions, these properties yield an ergodic chain. The general role of ergodicity in MCMC is reviewed in [Section {name}](#sec:mcmc).

A pointwise-informed proposal is called *locally balanced* if the transition kernel {math}`P_{g,\sigma}` is reversible with respect to some distribution {math}`\pi_{g,\sigma}` such that {math}`\pi_{g,\sigma}` converges weakly to {math}`\pi` as {math}`\sigma \to 0` {cite:p}`zanella2020informed`. In plain terms, as the proposal becomes increasingly local, the invariant distribution of the uncorrected informed proposal approaches the desired target. {cite:t}`zanella2020informed` demonstrates that a pointwise-informed proposal is locally balanced if and only if the balancing function satisfies {math}`g(t) = t g(1/t)`.

If this property holds for the function {math}`g`, then the locally balanced proposal will be reversible with respect to the following distribution without any correction:

```{math}
\pi_{g,\sigma}(x)
=\frac{\pi(x)Z_{g,\sigma}(x)}
{\sum_{x'\in\mathcal X}\pi(x')Z_{g,\sigma}(x')}.
```

Here, {math}`Z_{g,\sigma}` is defined as follows:

```{math}
Z_{g,\sigma}(x)
=\sum_{x'\in\mathcal X}
g\!\left(\frac{\pi(x')}{\pi(x)}\right)K_\sigma(x'\mid x).
```

For a local kernel satisfying {math}`K_\sigma(x'\mid x)\to\mathbf 1\{x'=x\}` and the conventional normalisation {math}`g(1)=1`, we have {math}`Z_{g,\sigma}(x)\to1`. Consequently, {math}`\pi_{g,\sigma}` converges to {math}`\pi` as {math}`\sigma\to0` {cite:p}`zanella2020informed`.

At a non-zero scale, {math}`\pi_{g,\sigma}` need not equal {math}`\pi`, so a Metropolis–Hastings correction is still required for exact sampling. Local balance makes the uncorrected proposal close to the target in the small-scale regime, which can improve the acceptance rate, while a non-zero {math}`\sigma` is required to produce non-degenerate movement across the state space.

{cite:t}`zanella2020informed` shows that locally balanced proposals can improve efficiency over uninformed local proposals in the asymptotic regimes considered there. Constructing an exact informed proposal nevertheless requires evaluating the density ratio for every candidate. Full enumeration can involve {math}`K^d` states; even a Hamming-one neighbourhood contains {math}`d(K-1)` candidates. The gradient-based methods below replace these repeated neighbouring-state evaluations with one gradient evaluation at the current state.

## Gradient-Based Discrete Sampling

+++
A function specified only on a discrete state space does not have an intrinsic gradient. Gradient-based discrete samplers therefore assume that the unnormalised log density {math}`U` is supplied with, or represented through, a differentiable extension to a continuous ambient space. For example, a neural network may accept a continuous vector representation even though the sampler evaluates it only at binary or one-hot encoded states. This extension need not be unique, and its gradient is useful only insofar as it predicts changes between the discrete states of interest.

Given such an extension, the log-density difference for a nearby state can be approximated by

```{math}
U(x')-U(x)\approx\nabla U(x)^\top(x'-x).
```

This approximation is most reliable when the extension is sufficiently smooth and {math}`x'` is close to {math}`x`; differentiability alone does not guarantee accuracy if the function changes sharply between neighbouring states. Since {math}`\log\pi(x)=U(x)-\log Z`, their gradients are identical and the unknown normalising constant is not required. One forward and backward pass at {math}`x` then supplies approximate scores for all local candidate moves, at the expense of Taylor-approximation error.

### Gibbs with Gradient

+++
{cite:t}`grathwohl2021oops` first introduces this notion of gradient-based discrete sampling through the Gibbs-With-Gradient algorithm. This algorithm can be seen as a version of the standard Gibbs algorithm where the updated coordinate is selected based on the gradient.

Returning to the general locally balanced sampler, we can derive the Gibbs-With-Gradient update by choosing an uninformed kernel that uniformly selects a state differing from the current state in exactly one coordinate. Define the Hamming-one neighbourhood by

```{math}
H(x)=\left\{x'\in\mathcal X:
\sum_{i=1}^d\mathbf 1\{x'_i\neq x_i\}=1\right\}.
```

In the running binary example, {math}`H(x)` contains exactly the {math}`d` states obtained by flipping one bit of {math}`x`.

For this fixed-scale update, the uniform kernel is

```{math}
K_{\sigma}(x'\mid x)
=\frac{\mathbf 1\{x'\in H(x)\}}{|H(x)|}.
```

When every coordinate has {math}`K` possible categories, {math}`|H(x)|=d(K-1)`. This factor is constant across candidate moves and is therefore absorbed into the proportionality below.

Next, we choose the balancing function {math}`g(t) = \sqrt{t}`. Before approximation, this yields the exact informed proposal

```{math}
q_{\mathrm{exact}}(x' \mid x) \propto \mathbf{1}\{x' \in H(x)\}  \sqrt{ \frac{\pi(x')}{\pi(x)}}
```

Using the unnormalised log density {math}`U` from Equation [](#eq:disc-sampling:target), this proposal can be expressed without evaluating {math}`Z`:

```{math}
q_{\mathrm{exact}}(x' \mid x) \propto \mathbf{1}\{x' \in H(x)\}  \exp \left(\frac{1}{2}( U(x') - U(x))\right).
```

Replacing the exact log-density difference by its first-order Taylor approximation defines the Gibbs-With-Gradient proposal:

```{math}
q_{\mathrm{GWG}}(x' \mid x) \propto \mathbf{1}\{x' \in H(x)\}  \exp \left(\frac{1}{2} \nabla U(x)^\top (x' - x)\right).
```

[](#fig:disc-sampling:gradient-discrete-space) illustrates this Gibbs-With-Gradient construction. The gradient at {math}`x` assigns different approximate weights to the states in {math}`H(x)`, and {math}`x'` is then sampled from the resulting proposal over those discrete neighbours.

:::{figure} assets/discrete_sampling/figure_gradient_discrete_space.png
:label: fig:disc-sampling:gradient-discrete-space
:align: center
:width: 65%

Gradient-informed sampling on a discrete state space. The gradient $\nabla U(x)$ of a differentiable extension provides local geometric information at the current state $x$, which induces different proposal weights over its Hamming-one neighbourhood $H(x)$. The starred state $x'$ represents a draw from this proposal. Adapted from @grathwohl2021oops.
:::

+++
This can be computed quickly by viewing the update as the selection of a coordinate to modify. In the binary case, this update is always a bit flip. Here, {math}`\operatorname{Softmax}` normalises a vector of real-valued scores into probabilities, and {math}`\operatorname{Cat}` denotes the categorical distribution with those probabilities. The resulting proposal is

```{math}
q_{\mathrm{GWG}}(\cdot\mid x)
=\operatorname{Cat}\left(
\operatorname{Softmax}\left(
-\frac{1}{2}(2x-1)\odot\nabla U(x)
\right)
\right).
```

The categorical case requires selecting both a coordinate and its proposed category. Let {math}`c_i` denote the current category of coordinate {math}`i`. We represent each categorical coordinate by a one-hot vector when evaluating the gradient. For every alternative category {math}`j \neq c_i`, define the approximate log-probability difference

```{math}
\widetilde d(x)_{i,j}
=
\nabla_{x_{i,j}}U(x)
-
\nabla_{x_{i,c_i}}U(x).
```

The first term measures the local effect of assigning category {math}`j` to coordinate {math}`i`, while the second corresponds to its current category.

There are {math}`d(K-1)` possible coordinate–category updates. We define a categorical proposal over these updates by

```{math}
q_{\mathrm{GWG}}(i,j\mid x)
=
\operatorname{Softmax}_{i\in[d],\,j\neq c_i}
\left(\frac{1}{2}\widetilde d(x)_{i,j}\right).
```

After sampling {math}`(i,j)\sim q_{\mathrm{GWG}}(\cdot,\cdot\mid x)`, the proposed state {math}`x'` is obtained by changing coordinate {math}`i` from {math}`c_i` to {math}`j` and leaving all other coordinates unchanged. The full categorical procedure is given in [](#alg:disc-sampling:gibbs-with-gradient).

:::{prf:algorithm} Categorical Gibbs With Gradient Sampler
:label: alg:disc-sampling:gibbs-with-gradient

- **Inputs:** initial categorical state {math}`x=(x_1,\dots,x_d)\in\mathcal{X}`; unnormalised log-density {math}`U`
- **Output:** sample sequence {math}`\mathcal{S}`
1. {math}`\mathcal{S} \gets ()`
1. **For** each iteration:
    1. compute {math}`\widetilde d(x)` and {math}`q_{\mathrm{GWG}}(\cdot,\cdot \mid x)`
    1. sample {math}`(i,j) \sim q_{\mathrm{GWG}}(\cdot,\cdot \mid x)`
    1. {math}`c_i \gets` current category of coordinate {math}`i`
    1. {math}`x' \gets \operatorname{replace}(x,i,j)`
    1. compute the reverse-move probability {math}`q_{\mathrm{GWG}}(i,c_i \mid x')`
    1. accept {math}`x'` with probability
    $$
\min\left\{1,
      \exp\left(U(x') - U(x)\right)
      \frac{q_{\mathrm{GWG}}(i,c_i \mid x')}{q_{\mathrm{GWG}}(i,j \mid x)}
      \right\}
    $$
    1. update {math}`x \gets x'` if accepted
    1. append {math}`x` to {math}`\mathcal{S}`
1. **Return** {math}`\mathcal{S}`
:::

+++
Because the gradient-based proposal uses a first-order approximation and is generally asymmetric, it does not by itself preserve the target distribution. We therefore apply the standard Metropolis–Hastings acceptance correction, described in [Section {name}](#sec:mh), which ensures that the target distribution is invariant. Under the ergodicity conditions stated above, the resulting Markov chain converges to the target distribution.

### Discrete Langevin Proposal

+++
Gibbs-With-Gradient can outperform uninformed single-coordinate proposals by using the gradient to prioritise promising updates. Its Hamming-one restriction also keeps the Taylor approximation local, but only one coordinate can change per iteration, which may produce strongly correlated samples in high dimensions. {cite:t}`zhang2022langevin` addresses this limitation with the Discrete Langevin Proposal (DLP), which retains gradient-based directionality while adding a distance penalty that permits multiple coordinates to be proposed simultaneously.

In the running binary example, Gibbs-With-Gradient proposes at most one bit flip, whereas DLP constructs a proposal for every bit and may change several bits in the same iteration.

To derive the proposal, consider the discrete analogue of a Gaussian random-walk kernel,

```{math}
K_{\sigma}(x' \mid x) \propto
\exp\left(-\frac{1}{2\sigma}\lVert x'-x\rVert_2^2\right).
```

Combining this kernel with a first-order approximation of the locally balanced weight gives the DLP

```{math}
:label: eq:disc-sampling:dlp

q(x'\mid x) \propto
\exp\left(
\frac{1}{2}\nabla U(x)^\top(x'-x)
-\frac{1}{2\sigma}\lVert x'-x\rVert_2^2
\right).
```

The linear term in Equation [](#eq:disc-sampling:dlp) favours moves predicted to increase {math}`U`, while the quadratic term penalises moves far from the Taylor expansion point. The step size {math}`\sigma` controls the strength of this penalty.

Because the state space factorises as {math}`\mathcal{X}=\prod_{i=1}^d\mathcal{X}_i` and the exponent in Equation [](#eq:disc-sampling:dlp) is a sum of coordinate-wise terms, the proposal factorises as {math}`q(x'\mid x)=\prod_{i=1}^d q_i(x'_i\mid x)`. Each factor is the categorical distribution

```{math}
q_i(x'_i\mid x)
=\operatorname{Cat}\left(
\operatorname{Softmax}_{x'_i\in\mathcal{X}_i}
\left[
\frac{1}{2}\nabla_i U(x)(x'_i-x_i)
-\frac{(x'_i-x_i)^2}{2\sigma}
\right]
\right).
```

Thus, larger values of {math}`\sigma` permit more distant proposals, whereas smaller values concentrate the proposal near the current state.

While this may look similar to the proposal formula for Gibbs with Gradient, it is important to note that in DLP, the softmax is computed per coordinate over the potential classes that can be realised. In Gibbs with Gradient, the softmax is over the entire sample to pick only one coordinate. This allows for the parallel update of all coordinates.

The factorisation makes it possible to propose all coordinates in parallel after one gradient evaluation. Although the proposal factors coordinate-wise conditional on the current state, the gradient of the joint log density still guides each coordinate using information from the full current configuration. A Metropolis–Hastings correction can then account for the remaining approximation error and dependence structure. [](#fig:disc-sampling:gwg-versus-dlp) contrasts this parallel update with the single-coordinate update used by Gibbs-With-Gradient.

:::{figure} assets/discrete_sampling/figure_gwg_vs_dlp.png
:label: fig:disc-sampling:gwg-versus-dlp
:align: center

Comparison of gradient-based discrete proposals. Gibbs-With-Gradient samples one coordinate from $q_{\mathrm{GWG}}(i\mid x)$ and proposes a state that differs in that coordinate. The Discrete Langevin Proposal constructs coordinate-wise distributions $\{q_i\}_{i=1}^d$ and can propose several changes in parallel. The displayed Metropolis–Hastings correction gives the adjusted variants; omitting it yields DULA in the discrete Langevin case. Based on @grathwohl2021oops and @zhang2022langevin.
:::

+++
Using DLP yields two samplers: the discrete Metropolis-adjusted Langevin algorithm (DMALA), which includes a Metropolis–Hastings correction, and the discrete unadjusted Langevin algorithm (DULA), which omits it. DULA generally has asymptotic bias. For log-quadratic targets, however, {cite:t}`zhang2022langevin` show that this bias vanishes as {math}`\sigma\to0`; they also bound the bias for targets whose gradients are close to those of a log-quadratic distribution.

In terms of implementation, the cost difference between DULA and DMALA is negligible: while DMALA incorporates a Metropolis–Hastings correction that requires the computation of both {math}`q(\cdot\mid x)` and {math}`q(\cdot\mid x')`, caching means that only one new proposal distribution is computed per step after initialisation. The cached version of DMALA is given in [](#alg:disc-sampling:discrete-langevin-sampler).

:::{prf:algorithm} DMALA with Cache
:label: alg:disc-sampling:discrete-langevin-sampler

- **Inputs:** initial state {math}`x=(x_1,\dots,x_d)\in\mathcal{X}`; unnormalised log-density {math}`U`
- **Output:** sample sequence {math}`\mathcal{S}`
1. {math}`\mathcal{S} \gets ()`
1. compute {math}`q(\cdot \mid x)` and cache proposal
1. **For** each iteration:
    1. sample {math}`x' \sim q(\cdot \mid x)`
    1. compute {math}`q(\cdot \mid x')` using DLP proposal
    1. accept {math}`x'` with probability
    $$
\min\left\{1,
      \exp\left(U(x') - U(x)\right)
      \frac{q(x \mid x')}{q(x' \mid x)}
      \right\}
    $$
    1. **If** accepted:
        1. update cached proposal to {math}`q(\cdot \mid x')`
        1. {math}`x \gets x'`
    1. append {math}`x` to {math}`\mathcal{S}`
1. **Return** {math}`\mathcal{S}`
:::

+++
The core logic lies in the fact that the {math}`x'` of the current iteration is either rejected or becomes the {math}`x` for the next sampling step. In the case of rejection, the proposal function {math}`q(\cdot| x)` does not change as {math}`x` remains the same. In the case of acceptance, the proposal step simply gets updated to match the updated {math}`x'` by using {math}`q(\cdot | x') \to q(\cdot | x)`, since this proposal must be computed anyway for the accept/reject step. Thus at each step, only one proposal function must be computed.

## Beyond Locally Balanced Proposals

+++
Locally balanced proposals offer a convenient framework for incorporating gradients into discrete samplers, but their defining approximation concerns the small-scale limit {math}`\sigma\to0`. At finite scales, local proposals may take a long time to move between separated modes, necessitating longer chains or multiple chains.

For example, if the binary target places high probability on two configurations separated by many low-probability intermediate states, a sequence of single-bit proposals may struggle to move from one mode to the other.

This motivates proposals designed for non-local movement. {cite:t}`zanella2020informed` already considers {math}`g(t)=t`, calling the resulting proposal the *naively informed* choice because it has the form {math}`P_{g,\sigma}(x'\mid x)\propto K_\sigma(x'\mid x)\pi(x')`. Building on this observation, {cite:t}`sun2023any` formalise the large-scale interpretation: they call a proposal family *globally balanced* if the invariant distribution of its uncorrected informed proposal converges to {math}`\pi` as {math}`\sigma\to\infty`. They assume that the base kernel becomes uniform in this limit. Since a state-independent constant cancels in the normalisation of the informed proposal, they write this limit as {math}`K_\sigma(x'\mid x)\to1`; for a normalised kernel on a finite state space, the corresponding limit is {math}`1/|\mathcal X|`. For {math}`g(t)=t`, the limiting proposal is then

```{math}
\lim_{\sigma\to\infty}P_{g,\sigma}(x'\mid x)
=\frac{\pi(x')/\pi(x)}
{\sum_{z\in\mathcal X}\pi(z)/\pi(x)}
=\pi(x').
```

Thus the proposal becomes an independent draw from the target itself and is globally balanced in the sense of {cite:t}`sun2023any`. Notice that {math}`g(t)=t` does not satisfy the locally balanced identity {math}`g(t)=t g(1/t)`; local and global balance refer to different limiting regimes.

### Any-scale Balanced Sampler

+++
The any-scale balanced sampler introduced by {cite:t}`sun2023any` makes two distinct modifications for proposals outside the local regime. First, it adjusts the density-ratio weighting to match the proposal scale. The family {math}`g(t)=t^\beta`, with {math}`\beta\in[1/2,1]`, interpolates between the locally balanced choice {math}`g(t)=\sqrt{t}` as {math}`\sigma\to0` and the globally balanced choice {math}`g(t)=t` as {math}`\sigma\to\infty`. This scale-dependent relationship is illustrated schematically in [](#fig:disc-sampling:scale-balance-cycle)(a).

Adjusting {math}`\beta` changes how strongly the proposal favours high-density candidates, but it does not reduce the error in a first-order approximation of {math}`U(x')-U(x)`. The method therefore makes a second modification: it introduces a quadratic approximation intended to remain informative over larger moves. The log-density difference is approximated by

```{math}
:label: eq:disc-sampling:any-scale-approximation

U(x')-U(x)
\approx (x'-x)^\top\nabla U(x)
+\frac{1}{2}(x'-x)^\top W(x'-x),
```

where {math}`W` is a symmetric matrix shared across states. Setting {math}`W=\nabla^2U(x)` would recover the usual second-order Taylor approximation locally, but repeatedly computing a Hessian is expensive. Instead, the sampler estimates a global {math}`W` from pairs {math}`(x_n,x'_n)` collected during burn-in:

```{math}
\widehat W
=\operatorname*{arg\,min}_{W=W^\top}
\sum_n
\left\lVert
W(x'_n-x_n)-\bigl(\nabla U(x'_n)-\nabla U(x_n)\bigr)
\right\rVert_2^2.
```

Thus {math}`W` is fitted to explain observed changes in the gradient and acts as a global curvature approximation. Scale-dependent balancing and curvature approximation address different problems: {math}`\beta` controls the density-ratio weighting, whereas {math}`W` improves the approximation used to score non-local candidates.

Substituting the quadratic approximation in Equation [](#eq:disc-sampling:any-scale-approximation) and {math}`g(t)=t^\beta` into the informed proposal gives

```{math}
:label: eq:disc-sampling:any-scale-proposal

q(x'\mid x)\propto
\exp\left(
\beta\left[(x'-x)^\top\nabla U(x)
+\frac{1}{2}(x'-x)^\top W(x'-x)\right]
-\frac{1}{2\sigma}\lVert x'-x\rVert_2^2
\right).
```

The quadratic term captures cross-coordinate interactions, but it also prevents direct coordinate-wise sampling. {cite:t}`sun2023any` therefore introduce an augmented proposal with a Gaussian auxiliary variable. The augmented construction is used directly with the Metropolis–Hastings correction below.

Choose a diagonal positive semi-definite matrix {math}`D` such that {math}`W+D` is positive semi-definite, and draw

```{math}
Y\mid x\sim
\mathcal N\left((W+D)^{1/2}x,I\right).
```

The matrix square root can be computed by an eigendecomposition or Cholesky factorisation. Conditional on {math}`Y`, define

```{math}
q(x'\mid x,Y)\propto
\exp\left(
\beta{x'}^\top
\left[\nabla U(x)-Wx+(W+D)^{1/2}Y\right]
-\frac{1}{2}{x'}^\top
\left(\beta D+\frac{1}{\sigma}I\right)x'
\right),
```

and sample {math}`x'` from this distribution. Because {math}`D` is diagonal and the state space factorises, this conditional proposal factorises across coordinates. Equations [](#eq:disc-sampling:any-scale-proposal) and the augmented proposal express, respectively, the motivating quadratic approximation and the auxiliary-variable transition implemented by the sampler. The Metropolis–Hastings step accounts for the resulting augmented proposal exactly.

The Metropolis–Hastings acceptance probability for the augmented proposal is

```{math}
A(x,x',Y)
=\min\left\{1,
\exp\bigl(U(x')-U(x)\bigr)
\frac{q(Y\mid x')q(x\mid x',Y)}
{q(Y\mid x)q(x'\mid x,Y)}
\right\}.
```

The diagonal matrix {math}`D` controls the conditional proposal variance. {cite:t}`sun2023any` consider two semidefinite programs for choosing it. The AB-max variant minimises the largest eigenvalue of {math}`W+D`:

```{math}
(D^*,\lambda^*)=\operatorname*{arg\,min}_{D,\lambda}\lambda
\quad\text{subject to}\quad
D\succeq0,\qquad 0\preceq W+D\preceq\lambda I.
```

This criterion focuses on the worst spectral direction. The AB-trace variant instead solves

```{math}
D^*=\operatorname*{arg\,min}_{D}\operatorname{tr}(D)
\quad\text{subject to}\quad
D\succeq0,\qquad W+D\succeq0.
```

The motivation for this objective comes from the continuous quadratic surrogate studied by {cite:t}`sun2023any` in the special case {math}`\sigma=\infty` and {math}`\beta=1`. In that surrogate, the covariance of the proposal increment is {math}`2D^{-1}`, so minimising the trace of {math}`D` encourages dispersion across coordinates. This covariance identity does not hold literally for the general discrete categorical proposal, whose dispersion also depends on {math}`\beta`, {math}`\sigma`, and the state space. The authors report that AB-trace performs better empirically and use it as their preferred variant.

The proposal parameters {math}`\beta` and {math}`\sigma` are correlated and target-dependent. {cite:t}`sun2023any` therefore adapt them by alternating coordinate-wise updates that seek to maximise the empirical average jump distance. For one parameter type {math}`a\in\{\sigma,\beta\}`, let {math}`\phi` denote its current value and define the feasible sets {math}`\mathcal I_\sigma=(0,\infty)` and {math}`\mathcal I_\beta=[1/2,1]`. With the other parameter held fixed, the adaptation block compares {math}`\phi` with feasible multiplicative perturbations of it over three consecutive chain segments, as shown in [](#alg:disc-sampling:ab-adaptive-algorithm).

:::{prf:algorithm} Any-scale Adaptation Block
:label: alg:disc-sampling:ab-adaptive-algorithm

- **Inputs:** parameter type {math}`a\in\{\sigma,\beta\}`; current value {math}`\phi\in\mathcal I_a`; adaptation rate {math}`\gamma\in(0,1)`; initial state {math}`x^{(0)}`
- **Output:** updated parameter {math}`\phi'`; samples {math}`x^{(1)},\dots,x^{(3N)}`
1. {math}`\phi_0\gets\phi`
1. {math}`\phi_+\gets\operatorname{proj}_{\mathcal I_a}\!\left(\phi(1+\gamma)\right)`
1. {math}`\phi_-\gets\operatorname{proj}_{\mathcal I_a}\!\left(\phi(1-\gamma)\right)`
1. use {math}`\phi_0` to compute the proposal and collect {math}`x^{(1)}, x^{(2)}, \dots, x^{(N)}`
1. compute {math}`d_0 \gets \sum_{i=1}^N \lVert x^{(i)} - x^{(i-1)} \rVert_1`
1. use {math}`\phi_+` to sample {math}`x^{(N+1)}, \dots, x^{(2N)}` using the proposal
1. compute {math}`d_{+} \gets \sum_{i=1}^N \lVert x^{(N+i)} - x^{(N+i-1)} \rVert_1`
1. use {math}`\phi_-` to sample {math}`x^{(2N+1)}, \dots, x^{(3N)}` using the proposal
1. compute {math}`d_{-} \gets \sum_{i=1}^N \lVert x^{(2N+i)} - x^{(2N+i-1)} \rVert_1`
1. **If** {math}`\max(d_0, d_{+}, d_{-}) = d_{+}`:
    1. {math}`\phi' \gets \phi_+`
1. **Else if** {math}`\max(d_0, d_{+}, d_{-}) = d_{-}`:
    1. {math}`\phi' \gets \phi_-`
1. **Else:**
    1. {math}`\phi' \gets \phi_0`
1. **Return** {math}`\phi'` and {math}`x^{(1)},\dots,x^{(3N)}`
:::

+++
The three segments in [](#alg:disc-sampling:ab-adaptive-algorithm) are consecutive: the second begins from {math}`x^{(N)}` and the third from {math}`x^{(2N)}`. The full adaptation alternates a block for {math}`\phi=\sigma` and a block for {math}`\phi=\beta`. If neither parameter changes during an outer iteration, the update rate is reduced geometrically, {math}`\gamma\gets\rho\gamma` for a decay factor {math}`\rho\in(0,1)`; otherwise the selected parameter values are retained for the next iteration. Because adaptation based on short-run jump distances can be noisy during early mixing, this procedure is intended for burn-in rather than continued adaptation after samples are retained for inference.

Through the combination of a quadratic approximation of the log-probability difference {math}`U(x') - U(x)` and the adaptive scheme for {math}`\beta` and {math}`\sigma`, the any-scale sampler interpolates between locally and globally balanced proposals.

### Automatic Cyclical Sampling

+++
One common issue with gradient-based samplers for continuous spaces is exploration: as the gradient points in the direction of the closest mode, it gives no information regarding further, more distant modes. This can hinder the accuracy of the sampler in terms of approximating the target distribution.

The same problem applies to gradient-based discrete samplers. Unlike a continuous state space, a discrete space has no arbitrarily small non-trivial perturbations: even the most local move must change at least one coordinate. This granularity can make the trade-off between conservative, high-acceptance moves and larger exploratory moves especially pronounced. Interpolating between locally and globally balanced proposals is therefore a natural way to vary the proposal scale when large movements may be required to escape a local mode.

In the continuous case, many approaches have been proposed for handling the mode-escape problem. One popular approach is cyclical SG-MCMC, discussed in Chapter [](#chap:sampling_methods_sg_mcmc). A cyclical schedule gradually reduces the step-size of a gradient-based sampler from very large magnitudes to small magnitudes and repeats for several cycles. The large steps at the beginning allow for more exploratory movements, which enable mode discovery; and the smaller step-sizes at the end of the cycle allow for exploitation of the mode. The repetition of the cycles allows for the discovery and exploitation of many modes, which is required for effective characterisation of multi-modal distributions.

Building on this intuition, {cite:t}`pynadath2024gradient` introduce a sampler that cycles jointly through step sizes and balancing exponents. Varying {math}`\sigma` alone can produce poorly balanced proposals and low acceptance probabilities, so the method uses the generalised DLP

```{math}
:label: eq:disc-sampling:cyclical-dlp

q_i(x'_i\mid x)
=\operatorname{Cat}\left(
\operatorname{Softmax}_{x'_i\in\mathcal X_i}
\left[
\beta\nabla_iU(x)(x'_i-x_i)
-\frac{(x'_i-x_i)^2}{2\sigma}
\right]
\right).
```

Setting {math}`\beta=1/2` in Equation [](#eq:disc-sampling:cyclical-dlp) recovers the standard DLP. Within each cycle, large values of {math}`\sigma` are paired with values of {math}`\beta` closer to 1 to encourage global exploration; as {math}`\sigma` decreases, {math}`\beta` moves towards {math}`1/2` and the proposal becomes more local.

The automatic tuning procedure begins with user-specified endpoint exponents {math}`\beta_{\min}` and {math}`\beta_{\max}` and a target acceptance rate. Short burn-in runs determine corresponding endpoint step sizes {math}`\sigma_{\min}` and {math}`\sigma_{\max}`. These endpoints define the cyclical step-size schedule, following the continuous-space construction of {cite:t}`zhang2020cyclical`.

For each step size {math}`\sigma_k`, the corresponding {math}`\beta_k` is selected to maximise an empirical acceptance-rate objective subject to {math}`\beta_k\in[1/2,\beta_{k-1}]`. This monotonic constraint reflects the transition from large moves with more globally balanced weighting at the beginning of a cycle to small, locally balanced moves at its end, while reducing the cost of searching for the full schedule. [](#fig:disc-sampling:scale-balance-cycle)(b) shows the coupled schedules used to move repeatedly from global exploration to local characterisation.

:::{figure} assets/discrete_sampling/figure_scale_balance_cycle.png
:label: fig:disc-sampling:scale-balance-cycle
:align: center

Proposal scale, balancing, and cyclical scheduling. Panel (a) schematically illustrates that the preferred balancing exponent increases with the proposal scale: small-scale proposals favour $\beta$ near $1/2$, whereas larger-scale proposals favour values closer to $1$. The curves show normalised illustrative efficiency rather than empirical measurements. Panel (b) shows coupled schedules for $\sigma_k$ and $\beta_k$ over two cycles, progressing from large proposals with more globally balanced weighting to small, locally balanced proposals within each cycle. Panel (a) is adapted conceptually from @sun2023any; panel (b) is adapted from @pynadath2024gradient.
:::

## Conclusion

+++
Gradient information can make sampling in high-dimensional discrete spaces substantially more efficient, even though the state space itself is not differentiable. Locally balanced proposals provide the organising principle: they combine an uninformed local kernel with target-specific information and use a Metropolis–Hastings correction to retain the desired target distribution. Gibbs-With-Gradient uses a first-order approximation to choose an informed single-coordinate update, whereas DLP factorises a Langevin-like proposal so that many coordinates can be proposed in parallel.

The appropriate proposal depends on the scale of movement required. Small, locally balanced moves can characterise a mode accurately but may mix slowly between separated modes. Any-scale balancing adapts the balancing exponent and adds a global curvature approximation for more effective non-local moves. Automatic cyclical sampling instead repeats a schedule from large exploratory proposals to small exploitative proposals, pairing each step size with a suitable balancing exponent.

In the running binary example, this progression moves from informed single-bit proposals, through parallel multi-bit updates, to methods that adapt the scale and weighting of those updates to improve movement between separated modes.

These methods trade additional structure and tuning for improved exploration. Their practical performance depends on the quality of the differentiable extension of {math}`U`, the accuracy of the local or quadratic approximation, and the cost of evaluating reverse proposals in the Metropolis–Hastings correction. The adjusted variants provide asymptotically exact sampling under the usual irreducibility and aperiodicity conditions; unadjusted variants can be cheaper, but generally introduce stationary bias.