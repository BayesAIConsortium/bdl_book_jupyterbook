(cha:laplace:analytic_functionality)=
# Analytic, low-cost functionality from Laplace

+++
In the previous sections, we have seen how the Laplace approximation endows a neural network with a Gaussian weight-space uncertainty (cf. Chapter [](#cha:laplace:introduction)), and how this uncertainty can be transformed into a closed-form expression for predictive uncertainty (cf. Chapter [](#cha:laplace:linearized)) informed by the training data (cf. Chapter [](#cha:laplace:diffgeo)). By combining the linearised pushforward with analytic approximations for the softmax integral in classification, we obtain a versatile analytic toolbox for uncertainty-driven decision making.

Classical Bayesian statistical methods – such as probabilistic prediction, out-of-distribution detection, active learning, Bayesian optimisation, continual learning, and model selection – become directly accessible in deep learning under the Gaussian assumption. This setting also enables further approximations for inference at various computational budgets {cite:p}`daxberger2021laplace,weber2025laplax`.

In this chapter, we examine two case studies demonstrating how such uncertainty estimates can be leveraged to address open challenges in deep learning. The first focuses on how to mitigate predictive overconfidence in ReLU networks. The second addresses continual learning: we will see that the Laplace approximation provides a natural regularisation term, which alleviates catastrophic forgetting. We then conclude with additional observations on how the Laplace-Gaussian Process connection can be leveraged for optimal data selection under different objectives. For a discussion on computational techniques for scaling these approaches to large networks, such as KFAC and subnetwork approximations, we refer to [Section {name}](#sec:laplace:kfac) and Chapter [](#chap:scalability_scalable_laplace).

:::{admonition} Takeaways

+++
Since linearised Laplace turns virtually every deep neural network into a Gaussian process predictive distribution, the powerful practical functionalities of Gaussian process regression – confidence and uncertainty quantification; model adaptation; active learning and design; continual learning; etc. – all become directly available in deep learning. Of course, since this is an approximate framework, these functionalities can only be expected to work to varying degrees of approximation.
:::

(sec:laplace:calibrated_uncertainty)=
## Fixing overconfidence with calibrated uncertainty

### Provable overconfidence in ReLU-like neural networks

+++
In supervised classification, deep neural networks are trained to map an input {math}`x \in \mathbb{R}^d` to a vector of class logits {math}`f_\theta(x) \in \mathbb{R}^C`, which are converted into categorical probabilities via the softmax function [](#eq:laplace:softmax). Ideally, these probabilities should reflect the true uncertainty of the model’s predictions. In practice, however, modern deep networks tend to be *overconfident*: they assign very high probabilities to certain classes even when the input is far from the training distribution. This behaviour is particularly problematic in safety-critical settings, such as medical diagnosis or autonomous driving, where unwarranted certainty can have serious consequences {cite:p}`hein2019why`.

A precise theoretical explanation for this behaviour in ReLU-like networks was given by {cite:t}`hein2019why`. Such networks consist of linear layers (fully connected or convolutional) combined with piecewise-linear activations such as ReLU or LeakyReLU. Consequently, they implement a piecewise affine mapping:

```{math}
f(x) = \mathbf{V}^{(l)}x + \mathbf{a}^{(l)}, \quad x \in \mathcal{Q}_l, \quad l = 1, \dots, R
```

where the polytopes {math}`\mathcal{Q}_1, \dots, \mathcal{Q}_R` partition the input space {math}`\mathbb{R}^d` into regions of constant activation pattern. Within each such region, the logits grow linearly with {math}`\| x \|`. As the input moves far away from the training data, the behaviour is dominated by a single affine component whose largest logit can grow without bound, while all other logits remain comparatively smaller. Since the softmax function magnifies differences between logits, this causes the predicted probability for a single class to approach one.

:::{prf:theorem} cf. Theorem 3.1, @hein2019why
:label: thm:reluproblem

+++
Suppose that {math}`\mathbf{V}^l` does not contain identical rows for all {math}`l = 1, \dots, R`, then for almost any non-zero input {math}`x` and any {math}`\varepsilon > 0`, there exists a scaling factor {math}`\alpha > 0` such that for some class {math}`k`,

```{math}
\operatorname{softmax}_k(f(\alpha x)) \geq 1 - \varepsilon,
```

and

```{math}
\lim_{\alpha \rightarrow \infty} \operatorname{softmax}_k(f(\alpha x)) = 1.
```
:::

+++
In other words, deterministic ReLU networks extrapolate with *increasing certainty* as we move away from training data, regardless of the lack of evidence.

(sec:laplace:bayesianoverconfidence)=
### Bayesian perspective as a counter measure

+++
The problem identified in the previous section is an artefact directly caused by the use of point estimates. The natural way to solve it is to assign a probability measure over the weights {math}`\theta` instead of committing to a single value. As we will see, this naturally leads to increasing variance away from the data, which tempers the confidence after the softmax is applied.

The Laplace approximation provides a tractable route to introduce this behaviour into neural networks. To recall, we approximate for a given trained network with parameters {math}`\theta_*` (the maximum a posteriori estimate), the posterior over parameters by a Gaussian:

```{math}
p(\theta \vert \mathcal{D}) \approx \mathcal{N}(\theta_*, \Psi(\theta_*)^{-1}),
```

where {math}`\Psi` describes the curvature of the negative log posterior evaluated at {math}`\theta_*`. With linearising the pushforward around {math}`\theta_*`, the logits become Gaussian as well (cf. [Section {name}](#sec:laplace:linearization)):

```{math}
z(x) \sim \mathcal{N}(\mu(x), k(x, x))
```

with

```{math}
:label: eq:laplace:pushforward

\mu(x) = f(x, \theta_*), \quad k(x, x) = \jaclogit(x, \theta_*) \Psi(\theta_*)^{-1} \jaclogit(x, \theta_*)^\top.
```

For ReLU networks, the components of {math}`\jaclogit(x, \theta_*)` grow at most linearly with {math}`\| x \|`, so the logit variance {math}`k(x, x)` grows quadratically with {math}`\| x \|` (equivalently, the standard deviation grows linearly). This means that, far from the training set, the variance in each logit matches the mean’s linear growth. We will now see that this leads to finite asymptotic confidence in each class, bounded away from 1 and 0.

The predictive class probabilities are obtained by marginalising over the Gaussian posterior

```{math}
p(y = k \vert x, \mathcal{D} ) = \mathbb{E}_{z \sim \mathcal{N}(\mu(x), k(x, x))}\left[ \operatorname{softmax}_k (z) \right]\,.
```

While exact evaluation is infeasible, accurate analytic approximations exist (cf. [Section {name}](#sec:laplace:predictive)). A well-known approximation in the binary case is due to {cite:t}`mackay1992evidence`:

```{math}
\mathbb{E}[\sigma(z)] \approx \sigma \left( \frac{\mu}{\sqrt{1 + \frac{\pi}{8} s^2 }}\right),
```

where {math}`z \sim \mathcal{N}(\mu, s^2)` is the Gaussian logit with mean {math}`\mu` and variance {math}`s^2 = k(x, x)`, and {math}`\sigma(t) = 1 / (1 + e^{-t})` is the logistic sigmoid.

This solution was established by {cite:t}`kristiadi2020being` in their paper *Being Bayesian, Even Just A Bit, Fixes Overconfidence In ReLU Networks*, where *Even Just A Bit* refers to the observation that the following theorem holds even when only the weights of the last layer are treated probabilistically, and *even* if the covariance {math}`\Psi(\theta_*)^{-1}` in the Laplace approximation is chosen in an *arbitrary* way (say, equal to the identity {math}`\mathbf{I}`).

:::{prf:theorem} cf. Proposition 2.5, @kristiadi2020being
:label: thm:fixingoverconfidence

+++
Consider a binary classification ReLU network {math}`f_\theta` with Laplace posterior {math}`\theta \sim \mathcal{N}(\theta_*, \Psi(\theta_*)^{-1})`. Then, the predictive confidence at input {math}`x` decreases monotonically as prior variance {math}`\sigma_0^2` increases, satisfying

```{math}
\lim_{\sigma_0^2 \rightarrow \infty} \sigma(\vert z(x) \vert) \leq \sigma \left(
		\frac{\vert \mu(x) \vert}{1 + \sqrt{\frac{\pi}{8}\lambda_{\text{max}}(H)\,\|\jaclogit(x, \theta_*)\|^2}}\right), \quad \lim_{\sigma_0^2 \rightarrow 0} \sigma( \vert z(x) \vert ) = \sigma(\vert \mu(x) \vert).
```
:::

+++
Here, {math}`\lambda_{\text{max}}(H)` denotes the largest eigenvalue of the negative log-likelihood Hessian evaluated at the posterior mean {math}`\theta_*`, while {math}`\mu(x) = f(x, \theta_*)` and {math}`\jaclogit(x, \theta_*)` are the mean logit and network Jacobian at {math}`\theta_*`, exactly as in Equation [](#eq:laplace:pushforward). Note that {math}`H` differs from the posterior precision {math}`\Psi(\theta_*)` only by the Gaussian-prior term, {math}`\Psi(\theta_*) = H + \sigma_0^{-2}\mathbf{I}`, and this prior contribution vanishes as {math}`\sigma_0^2 \to \infty`, so only the likelihood Hessian survives in the bound.

Of course, this theorem provides only an asymptotic statement. In words: If we assign *any* Gaussian measure to the weights of the network (whether explicitly constructed as a Laplace approximation or simply imposed) then asymptotic overconfidence is “healed” in the weak sense that it is bounded away from 1 and 0. Despite its weak form, this result is a great argument for balancing the cost of Laplace approximations with their goals. If all you want is to fix asymptotic confidence, then you can do so at virtually zero computational cost: Just impose an arbitrary confidence on the parameters. This is still better than using a point estimate instead.

:::{figure} assets/laplace_analytic_functionality/relu_confidence_comparison.png
:label: fig:laplace:overconfidence_fix
:align: center
:width: 95%

**Mitigating overconfidence in a ReLU network with the Laplace approximation.** The figure shows the predictive confidence on a two-moons binary classification task for three different models. Darker background shading indicates higher confidence, while the solid black line marks the decision boundary at $p=0.5$. **(Left)** The standard MAP-trained ReLU network exhibits severe overconfidence. As described in [](#thm:reluproblem), its predictive confidence incorrectly approaches 1 (maximum certainty) for inputs far away from the training data. **(Middle)** Applying the Laplace approximation introduces uncertainty over the network's weights. The resulting logit variance grows quadratically with distance from the data, which tempers the softmax and reduces predictive confidence in out-of-distribution regions. **(Right)** Adding a ReLU-GP residual term (RGPR) to the Laplace-approximated logits introduces an additional source of variance that grows cubically. This ensures that the predictive confidence correctly converges to the uniform distribution (0.5) for inputs far from the training data, providing better calibration. **(Far Right)** This panel quantitatively confirms the visual results by plotting the maximum confidence of each model as an input point is scaled away from the origin.
:::

(sec:laplace:role_prior_variance)=
### The role of the prior variance

+++
[](#thm:fixingoverconfidence) highlights the special role of the prior variance {math}`\sigma_0^2`: increasing {math}`\sigma_0^2` broadens the posterior over weights, which in turn accelerates variance growth in the logits and reduces predictive confidence away from the data. On the other hand, letting {math}`\sigma_0^2 \to 0` recovers the deterministic prediction. Although {math}`\sigma_0^2` is traditionally motivated by the weight decay used during training, in the Laplace setting it can be tuned *post hoc* to improve the calibration.

Hyperparameters such as the prior variance or additional factors entering the curvature can be tuned with techniques often motivated by the literature on Gaussian processes {cite:p}`Rasmussen2006Gaussian`. This includes downstream objectives (e.g., negative log-likelihood, out-of-distribution entropy) or model fit criteria (e.g. log marginal likelihood). Here, the Gaussian structure of the Laplace approximation allows for closed-form objectives suitable for gradient-based tuning, which is discussed in more detail in addition to other applications in the next chapter (cf. Chapter [](#cha:laplace:model_selection)).

For the particular setting of calibrating {math}`\sigma_0^2` such that the resulting uncertainty structure reflects out-of-distribution well, {cite:t}`kristiadi2020being` suggest using

```{math}
:label: eq:laplace:prior_variance_objective

\operatorname{arg min}_{\sigma_0^2} - \tfrac{1}{n} \sum_{i=1}^n \log p(y = \hat{t}_i \mid \hat{x}_i, \mathcal{D}) - \gamma \ShanEntSymb[p(y \mid \tilde{x}_i, \mathcal{D})],
```

where the first term is the standard cross-entropy loss over an in-distribution validation set {math}`\{\hat{x}_i, \hat{t}_i\}`, while the second is the predictive entropy over a set of out-of-distribution inputs {math}`\{\tilde{x}_i\}`, which the objective encourages to be large through the leading minus sign. The parameter {math}`\gamma \in [0, 1]` controls the trade-off between these two terms.

### How to mitigate the problem even further

+++
A simplified summary of [](#thm:fixingoverconfidence) is that assigning a Gaussian measure to weights bounds asymptotic confidence away from the pathological values 1 and 0. But it will not drive asymptotic confidence down to the ideal uniform (maximum entropy) limit {math}`1/C` where {math}`C` is the number of classes. And of course the Theorem says nothing about confidence “close to” or “in-between” the training data. To address either problem, {cite:t}`kristiadi2021learnable,kristiadi2021infinite` consider two extensions to tune the overall confidence structure: one reshapes the local curvature for better calibration close to the data {cite:p}`kristiadi2021learnable`, the other combines the weight-space uncertainty with some additional output-space uncertainty to ensure the ideal uniform confidence limit {cite:p}`kristiadi2021infinite`.

The approach *Learnable Uncertainty under Laplace approximations* (LULA) {cite:p}`kristiadi2021learnable` augments a trained network with additional parameters – *uncertainty units* – inserted into the architecture in a way that preserves the MAP prediction {math}`f_{\theta_*}(x)` but modifies the Hessian {math}`H`, e.g. for three linear layers without nonlinearities and biases for simplicity, we have,

```{math}
\begin{align}
	f(x) = \begin{bmatrix} W^{(3)}_{\text{MAP}} \end{bmatrix} & \begin{bmatrix} W^{(2)}_{\text{MAP}} \end{bmatrix} \begin{bmatrix} W^{(1)}_{\text{MAP}} \end{bmatrix} x                                                                                                                                                                       \\
	                                                                       & \Rightarrow f^{(\text{LULA})}(x) = \begin{bmatrix} W^{(3)}_{\text{MAP}} & 0 \end{bmatrix} \begin{bmatrix} W^{(2)}_{\text{MAP}} & 0 \\ \hat{W}_1^{(2)} & \hat{W}_2^{(2)} \end{bmatrix} \begin{bmatrix} W^{(1)}_{\text{MAP}} \\ \hat{W}_1^{(1)} \end{bmatrix} x\,.
\end{align}
```

Therefore, the final variance changes while the mean prediction stays the same, which motivates the name *uncertainty units*. These units are then trained using an uncertainty-aware loss that encourages high predictive entropy on out-of-distribution (OOD) inputs and low entropy on in-distribution (ID) inputs, with gradients masked so that the original prediction remains unchanged. It can then be shown that the resulting Laplace approximation with diagonal curvature approximation has higher variance over the output compared to the diagonal Laplace for the network without the uncertainty units (cf. Proposition 2, {cite:t}`kristiadi2021learnable`).

However, residual overconfidence can still be left for inputs extremely far from the training data. This is a consequence of the fact that ReLU networks have only finitely many effective features, whereas a Gaussian process (GP) with an infinite set of ReLU features exhibits a growth of cubic variance, forcing the softmax confidence toward the uniform distribution as {math}`\| x \| \rightarrow \infty`. The *ReLU-GP residual* (RGPR) approach {cite:p}`kristiadi2021infinite` adds such a GP as an independent residual term {math}`\hat{f} \sim \mathcal{GP}(0, \hat{k})` to the Bayesian neural network:

```{math}
\tilde{f}(x) = f(x) + \hat{f}(x)
```

with a specific double-sided cubic spline kernel {math}`\hat{k}` chosen so that {math}`\hat{k}(x, x) \approx 0` near the origin but grows cubically with {math}`\| x \|`. The resulting logit distribution is

```{math}
z(x) \sim \mathcal{N}\big( \mu(x),  k(x, x) + \hat{k}(x, x)\,\mathbf{I}\big)\,
```

preserving the mean while adding variance that dominates in the tails. Applying a softmax approximation to this Gaussian results in a predictive distribution that converges to the uniform distribution in the limit (cf. Theorem 4, {cite:t}`kristiadi2021infinite`).

Importantly, this Gaussian process term can be added in such a way that its evaluation has constant and negligible cost, independent of the training data. This is because it is explicitly constructed to not be affected by data within the training region, only contributing prior uncertainty in the OOD regime. It is thus another example of a Laplace approximation adding specific functionality, at extremely minimal cost overhead. The double-sided cubic spline kernel chosen in this work also has a pleasing interpretation as a limit process arising from an infinite number of the ReLU uncertainty units from above, all distributed in a regular fashion around the training domain, pointing “outwards” so that they are not affected by the training data. The implied intuition is that these units “complete” or complement the finitely many ReLUs that make up the deep network that we actually train on the available (finite) data, and provide a nonparametric sea of ReLUs that cover the domain towards the horizon. To speak even more metaphorically, this is arguably an example of the Bayesian perspective on deep learning providing an intuition – and a concrete, practical solution! – to a pathology of point-estimated deep models. [](#fig:laplace:overconfidence_fix) compares the predictive confidence of the MAP network, its Laplace approximation, and the RGPR extension on a two-moons task, illustrating how each successive step tempers overconfidence away from the data.

:::{admonition} Takeaways

+++
Asymptotic and OOD overconfidence are examples of pathologies arising from the use of point estimates in deep learning. The framework of Laplace approximations (or more generally, assigning a Gaussian measure to the weights and using linearisation to push forward to the output domain) can fix these issues to some tangible, quantifiable degree, while adding, occasionally negligible, or at least very manageable computational overhead. This is arguably because these problems really just arise discretely from a non-Bayesian treatment, and vanish as soon as one assigns *a* probability measure, *any* kind of posterior on the parameters.
:::

(sec:laplace:continuallearning)=
## Fixing catastrophic forgetting in continual learning

+++
In *continual learning*, a single neural network is trained on a sequence of tasks

```{math}
t = 1, \dots, T, \qquad \text{with datasets } \mathcal{D}_t,
```

with the goal of maintaining good performance on earlier tasks. This setting is increasingly relevant in real-world deployments: modern large-scale models are trained once, but are then confronted with a stream of evolving data and objectives. A key obstacle to adaptation is *catastrophic forgetting*, the loss of previously acquired abilities when learning new tasks.

A naïve approach is to fine-tune the current parameters {math}`\theta_t` by minimising the loss on the next dataset {math}`D_{t+1}`:

```{math}
\begin{align}
	\theta_{t+1}^* = \operatorname{arg min}_{\substack \theta}\ \mathcal{L}(\theta; \mathcal{D}_{t+1}).
\end{align}
```

However, this discards all information about earlier tasks: parameters drift toward the new optimum for adapting to the new data {math}`\mathcal{D}_{t+1}`, and end up erasing information by neglecting critical weights for previous performance.

(sec:laplace:continuallearning:bayesiansolution)=
### A Bayesian solution

+++
From a Bayesian perspective, training on {math}`\mathcal{D}_{t+1}` should not start from an uninformative prior. Instead, the posterior after task {math}`t` should become the prior for task {math}`t + 1`:

```{math}
\begin{align}
	p(\theta_{t+1} \mid D_{1:t+1}) \propto p(\mathcal{D}_{t+1} \mid \theta_{t+1})\ p(\theta_t \mid \mathcal{D}_{1:t}),
\end{align}
```

instead of the original uninformative isotropic Gaussian {math}`p(\theta)` in form of weight decay. Taking the negative log turns this into a regularised optimisation problem, where the regulariser encodes knowledge from earlier tasks.

If {math}`p(\theta_t \mid D_{1:t})` is approximated by a Gaussian {math}`\mathcal{N}(\theta_t^*, \Psi_t^{-1})` via the Laplace approximation {cite:p}`sliwa2024efficient`, the next-task objective becomes:

```{math}
\begin{align}
	\mathcal{L}_{t+1}^{\mathrm{reg}}(\theta) = \mathcal{L}(\theta; \mathcal{D}_{t+1}) + \tfrac{\lambda}{2} \| \theta - \theta_t^* \|_{\Psi_t}^2,
\end{align}
```

where {math}`\| v \|_{\Psi_t}^2 := v^\top \Psi_t\, v` denotes the squared Mahalanobis norm induced by the precision {math}`\Psi_t`. The precision matrix {math}`\Psi_t` determines how strongly each direction in parameter space is protected: large precision (high curvature) means a stiff penalty in that direction, since the loss on the previous task would change rapidly, while low precision allows for adaptation. This yields an *anisotropic* weight-space prior that selectively resists change where it would harm past performance. The regularisation strength {math}`\lambda` controls the trade-off between retaining performance on past tasks and adapting to new ones by scaling the precision of the prior: higher values express greater confidence in past parameters, lower values signal greater uncertainty and openness to change.

(subsec:laplace:bayesianfiltering)=
### Bayesian filtering in weight space

+++
Starting from the previously motivated interpretation and following {cite:t}`sliwa2024efficient`, we can formalise continual learning as Bayesian state estimation in weight space:

```{math}
\begin{align}
	\theta_0 \rightarrow \theta_1 \rightarrow \dots \rightarrow \theta_T
\end{align}
```

with a Gaussian *state-transition* model and a likelihood defined by the task loss

```{math}
\begin{align}
	p(\theta_{t+1} \mid \theta_t) & = \mathcal{N}(\theta_t, Q) \qquad                                           &  & \text{(process model)}      \\
	p(D_t \mid \theta_t)\,         & \propto \exp\big( - \frac{1}{\tau} \mathcal{L}(\theta_t; D_t) \big), \qquad &  & \text{(observation model)}.
\end{align}
```

Continual learning becomes in this interpretation a problem of tracking a moving target in weight space: the process model predicts where the target might go next, and the observation model corrects this prediction based on new evidence. The additional process noise covariance {math}`Q` models the expected drift between tasks: large {math}`Q` corresponds to high *plasticity* (parameters may change substantially), while small {math}`Q` enforces *stability* (parameters remain close).

The Bayesian filtering recursion consists of:

1.  *Predict step*: propagate the posterior through the process model to obtain a prior for {math}`\theta_t`, (potentially) inflating the uncertainty by {math}`Q`.

2.  *Update step*: incorporate the likelihood from {math}`\mathcal{D}_t`, pulling the mean toward the new optimum and shrinking uncertainty according to curvature.

This is analogous to a Kalman filter in parameter space, but with a nonlinear, non-Gaussian observation model arising from the neural network loss. The Laplace approximation replaces this intractable term with a local Gaussian, making the update step analytically tractable within the filtering framework.

(sec:laplace:continuallearning:ewcoslalofi)=
### From EWC to OSLA to Lo-Fi

+++
The state-space formulation provides a unifying Bayesian view of continual learning. Different methods in the literature can be seen as specific choices for (i) how the posterior covariance is represented and updated, and (ii) whether parameter drift between tasks is allowed. The Laplace approximation supplies the Gaussian form needed for tractable updates, but the quality and scalability of the approach depend on the curvature approximation used.

#### Elastic Weight Consolidation (EWC).

+++
{cite:t}`kirkpatrick2017overcoming` proposed *elastic weight consolidation* (EWC), which constrains parameters to remain close to the previous optimum {math}`\theta_{t}^*` using the diagonal of the Fisher information metric {math}`F_{t}`:

```{math}
\begin{align}
	\mathcal{L} = \mathcal{L}(\theta; \mathcal{D}_{t+1}) + \tfrac{\lambda}{2}( \theta - \theta_{t}^*)^\top \operatorname{diag}(F_{t}) ( \theta - \theta_{t}^*).
\end{align}
```

This is equivalent to the proposed filtering framework when choosing {math}`Q = 0` and a diagonal precision. In the sequential setting, a separate quadratic penalty can be added for each of the previous tasks (multi-penalty), or equivalently, the Fisher matrices can be accumulated into a single continuously updated penalty. EWC is computationally efficient but lacks correlation between parameters.

#### Online Structured Laplace Approximation (OSLA).

+++
{cite:t}`ritter2018online` extended EWC by replacing the diagonal Fisher with a *Kronecker-factored* (KFAC) curvature approximation (cf. [Section {name}](#sec:laplace:kfac)), capturing intra-layer parameter correlations while remaining scalable. In the Bayesian filtering view, OSLA maintains a single Gaussian posterior over parameters and updates it *recursively* after each task:

```{math}
\begin{align}
	\Psi_t = \lambda\, \mathrm{KFAC}(F_t) + \Psi_{t-1},
\end{align}
```

where {math}`F_t` is again the Fisher information matrix computed with the current MAP estimate on task {math}`t`. The running update replaces the need to store separate penalties for each task, as in multi-penalty EWC, and naturally implements a sequential Bayesian update with {math}`Q = 0` (no assumed drift). By incorporating inner-layer correlations, OSLA produces a more faithful posterior approximation and better retention over long sequences of tasks.

#### Low-Rank Laplace-Gaussian Filtering (Lo-Fi).

+++
{cite:t}`pmlr-v232-chang23a` and {cite:t}`sliwa2024efficient` generalised the Bayesian framework to arbitrary {math}`Q`, enabling explicit modelling of expected drifts between tasks. They approximate the precision matrix using a low-rank approximation of the GGN (plus diagonal representing the initial prior), which makes both, prediction and update step, scalable while still modelling correlation between different layers.

In this general Bayesian filtering view, the concept of *smoothing* also becomes available. After the forward filtering pass, an optional backward pass refines each task-specific parameter estimate using information from future tasks without revisiting their original data. The process can be compared to passing a message along a chain of messengers, where each messenger (task) has access only to the message (parameters) they just received and their own local observations; over time, the message can degrade (catastrophic forgetting). Smoothing allows later messengers to send clarifications or corrections back upstream, improving earlier versions without re-reading the original letters (data). In a deep learning context, this means earlier task models can be improved based on knowledge from later tasks, while preserving data privacy by never sharing or replaying the original datasets from other tasks.

:::{admonition} Takeaways

+++
The Laplace approximation arising from pre-training a network can be used as a log prior (a regulariser) on subsequent tasks. Doing so encourages the model to use “unused” parts of the parameter space to “store” new training information, and thus mitigates forgetting during continual learning. The underlying algebraic idea is that Bayes’ theorem for Gaussians is a closure: Conditioning a Gaussian prior (an {math}`\ell_2` regulariser) on a (linear) Gaussian likelihood (a Laplace approximated training loss) yields a Gaussian posterior (a Laplace approximated trained network). Hence, updating a Gaussian *posterior* with new data is conceptually the same thing as updating a Gaussian *prior*. The only real challenge is to manage the resulting data-structures well, to keep computational cost manageable. But since Laplace maps this process onto Gaussians, these computational aspects are all mapped to numerical linear algebra, for which very powerful, well-studied algorithms are readily available.
:::

(sec:laplace:data_centric)=
## Data-centric objectives

+++
The Laplace approximation provides a tractable way to understand the relationship between a model and its training data, which, in turn, improves properties such as predictive calibration (cf. [Section {name}](#sec:laplace:calibrated_uncertainty)) and the ability to learn sequentially (cf. [Section {name}](#sec:laplace:continuallearning)). The same model-data relationship is central to the fields studying optimal data selection. Rather than relying on random sampling, methods have been developed to strategically choose the most informative data to label (active learning) and the most efficient points to evaluate (Bayesian optimisation). A fundamentally Bayesian approach underpins these strategies, leveraging a model’s current beliefs to guide data acquisition. Also more nuanced data-centric questions exist, such as determining the most effective training order (curriculum learning), that have been studied from a Bayesian perspective with the help of the Laplace approximation {cite:p}`deng2023towards`. However, here, we focus on active learning and Bayesian optimisation due to their foundational connection to Bayesian principles.

### Active learning: selecting what to label next

+++
The goal of **active learning** is to train an optimal model with a minimal data collection effort. For instance, training neural solvers for partial differential equations often requires a vast amount of training data generated by costly numerical simulations. Instead of random generation, active learning aims to strategically select the most informative data points to label, thereby achieving the same model accuracy with significantly less numerical simulations {cite:p}`musekamp2025active`.

From a Bayesian perspective, this selection process can be formalised as an optimisation problem guided by information theory. {cite:t}`mackay1992information` proposed framing active learning as a sequential design task where the goal is to choose the next data point that is expected to provide the maximal information gain about the model’s parameters {math}`\theta`. This gain is measured by the expected reduction in the entropy of the parameter posterior after observing a new data point {math}`(x_i, y_i)`:

```{math}
:label: eq:laplace:almackay

\mathcal{U}(x_i) = \ShanEntSymb[p(\theta \mid \mathcal{D})] - \mathbb{E}_{y \sim p(y \mid x_i, \mathcal{D})}[\ShanEntSymb[p(\theta \mid \mathcal{D}, x_i, y)]]
```

where {math}`\ShanEntSymb[p] = -\int p(u) \log p(u) \, \mathrm{d}u` denotes the (differential) entropy of a distribution {math}`p`; for a discrete distribution, such as the class-label predictive {math}`p(y \mid \cdot)`, the integral is replaced by the sum {math}`-\sum_u p(u) \log p(u)`.

In general, computing this expected information gain is intractable, though the linearised Laplace approximation provides again a solution. By approximating the posterior {math}`p(\theta \mid \mathcal{D})` with a Gaussian, {cite:t}`mackay1992information` showed that the expected information gain from a potential new data point at input {math}`x` is a monotonic function of the model’s current predictive variance at that location {math}`s^2(x) = \jaclogit(x, \theta_*) \Psi(\theta_*)^{-1} \jaclogit(x, \theta_*)^\top`. This provides a principled justification for the intuitive and widely used uncertainty sampling heuristic: *to learn the most, one should query the points where the model is currently most uncertain* {cite:p}`mackay1992information`.

The objective of maximising parameter information is now widely known as *Bayesian Active Learning by Disagreement* (BALD) {cite:p}`houlsby2011bayesian`. The BALD score is a mathematically equivalent reformulation of the original objective:

```{math}
:label: eq:laplace:bald

\operatorname{BALD}(x) = \ShanEntSymb[p( y \mid x, \mathcal{D})] - \mathbb{E}_{p(\theta \mid \mathcal{D})} \ShanEntSymb[p(y \mid x, \theta)]\,.
```

This version provides the powerful intuition that one should query points where the model is overall uncertain about its prediction (the first term), but where individual parameters {math}`\theta` from the posterior are confident in their potentially disagreeing predictions (the second term is low).

It has been observed, that BALD’s focus on parameter uncertainty can be misaligned with the goal of predictive performance, as it lacks an explicit notion of the relevant input data distribution. This can lead to the selection of obscure or irrelevant data points that offer little benefit to typical test input. To address these shortcomings, the *Expected Predictive Information Gain* (EPIG) {cite:p}`mindermann2022prioritized` was proposed, which measures the expected information a candidate label {math}`y` at input {math}`x` provides about the label {math}`y_*` of a random target input {math}`x_*` drawn from a target distribution {math}`p_*(x_*)`. This objective is formalised as the expected reduction in predictive entropy on a random target point, averaged over all possible outcomes of the candidate query:

```{math}
:label: eq:laplace:epig

\operatorname{EPIG}(x) = \mathbb{E}_{p_*(x_*)}[\ShanEntSymb[p(y_* \mid x_*)] - \ShanEntSymb[p(y_* \mid x_*, x, y)]]\,.
```

Both the BALD and EPIG objectives are agnostic to the method used for posterior approximation. By shifting the entropy computation to the lower-dimensional prediction space (BALD) or considering only the predictive entropy (EPIG), this information-theoretic approach lessens the reliance on a specific form of weight-space uncertainty, opening the door for various uncertainty quantification tools. The Laplace approximation remains a fast and tractable method for obtaining the predictive uncertainty needed to compute either score. Ultimately, however, the performance of these acquisition functions is strongly coupled to the quality of the resulting uncertainty estimates.

(sec:laplace:bayesopt)=
### Bayesian optimisation: finding the optimal point

+++
Many problems can be framed as finding the global optimum of an objective function that is very expensive to evaluate. Bayesian optimisation (BO) is a sequential strategy for finding the global optimum of such expensive black-box functions and works by combining a probabilistic surrogate model of the objective with acquisition functions to select the next point for evaluation {cite:p}`garnett2023bayesian`. Gaussian Processes (GPs) are the standard surrogate model because they are inherently Bayesian, providing a posterior distribution over the function and allowing for many popular acquisition functions to be computed analytically, making the entire process efficient and principled.

The linearised Laplace approximation (LLA) provides a way of using neural networks as surrogates by interpreting them as a Gaussian Process, which allows for reusing existing analytic acquisition functions, such as maximising the *upper confidence bound* given by

```{math}
:label: eq:laplace:ucb

\alpha_{\text{UCB}}(x) = f(x, \theta_*) + \beta\, s(x)
```

where {math}`f(x, \theta_*)` is the network’s mean prediction at the MAP estimate, {math}`s(x)` is the predictive standard deviation from the LLA posterior and {math}`\beta` a hyperparameter balancing exploration and exploitation {cite:p}`kristiadi2023promises`.

The final optimisation process is then iterative. At each step, a neural network is trained on the existing data set and queried using the LLA and an acquisition function for the next data point to evaluate the original black-box function. Then, this new information must be incorporated by retraining or adapting the neural network on the augmented dataset to find a new maximum-a-posteriori (MAP) estimate and then re-computing the Laplace approximation around this new point. This refines the model’s beliefs and its weight-space uncertainty, guiding the next decision in the sequence.

{cite:t}`kristiadi2023promises` demonstrated that the LLA surrogate inherits the strong predictive power and inductive biases of its underlying neural network, allowing it to outperform standard GPs on complex, high-dimensional tasks like searching through image spaces. However, they also note that broken predictive behaviour in extrapolation regions can significantly derail the optimisation process. This becomes clear when considering the Upper Confidence Bound as an example. For a ReLU network, the predictive mean outside the data region becomes a non-constant linear function. This means the predicted value will increase or decrease indefinitely as you move away from the data, while the uncertainty term also tends to grow linearly in these extrapolation regions, following the discussion in [Section {name}](#sec:laplace:bayesianoverconfidence). Since both terms grow linearly, and the UCB score uses their sum, this utility also increases without bound in the same direction, leading the optimiser to choose a point infinitely far away in an unbounded search space. Note that this behaviour is consistent within the context of UCB: A region of extremely large expected reward *and* extremely large uncertainty is a very appealing target for another experiment. So the problem here is arguably not the Bayesian treatment, but a badly chosen model class. Namely one that quite literally believes in pots of gold at the end of the rainbow.

:::{admonition} Takeaways

+++
Laplace approximations endow deep networks with probabilistic *uncertainty*, at tractable computational cost. A probability measure in itself, though, is just an intermediate mathematical notion. Its principal philosophical purpose, perhaps, is to allow meaningful *decisions*. For example, the Laplace posterior can be used to decide where to collect more training data in order to most efficiently reduce uncertainty and thus improve the model. Or, less directly, to find points where the target function (or a function of it) takes extremal values, provides most return. Interestingly, because these decisions have to be taken within a fixed domain, the Bayesian posterior used for this purpose does not have to be “correct” in an absolute sense, only a relative one. This highlights that when picking an approximate inference scheme for Bayesian deep learning, one should consider the downstream use case.
:::