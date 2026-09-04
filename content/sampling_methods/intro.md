(chap:sampling:intro)=
# Introduction to sampling

+++
## Introduction

The fundamental objective of Bayesian statistics is to compute the posterior distribution of a random variable {math}`\theta \in \Theta` given data {math}`\mathcal{D}`. In the context of supervised learning, where the data {math}`\mathcal{D}=\{X,Y\}` comprises features/inputs {math}`X` and responses/labels {math}`Y`, Bayes’ formula for the parameter posterior distribution takes the following form

```{math}
:label: eq:bayes_sampling

p(\theta \mid \mathcal{D}) = \frac{p(Y \mid \theta,X)p(\theta)}
{\int_{\Theta}p(Y \mid \theta,X)p(\theta) \rm{d} \theta} \, ,
```

where {math}`p(Y \mid \theta,X)` denotes the likelihood of {math}`N` observations for the target {math}`Y=\{y_1,\dots,y_N\}` given inputs {math}`X=\{x_1,\dots,x_N\}`, {math}`p(\theta)` is the density of the prior distribution for {math}`\theta`, and

```{math}
p(\mathcal{D}) = \int_{\Theta}p(Y \mid \theta,X)p(\theta) \rm{d} \theta,
```

denotes the *model evidence* of the data. Note that, in general the posterior is often only known up-to a normalizing constant, because, aside from simple models, calculating the integral in the model evidence is not feasible for high-dimensional {math}`\theta` nor is it often available in closed form. Therefore, it is more common to work with the unnormalized posterior {math}`p(\theta \mid \mathcal{D}) \propto p(Y \mid \theta, X) p(\theta)`.

We note that the ultimate goal in solving Equation [](#eq:bayes_sampling) is estimating the *posterior predictive* for new inputs {math}`x^*`, i.e. the pushforward of the posterior through the test likelihood:

```{math}
p(y^* \mid x^*, \theta) \, , \quad {\rm for} \quad  
\theta \sim p(\theta \mid \mathcal{D}) \, .
```

This is a distributed quantity that incorporates both *epistemic and aleatoric uncertainty* (uncertainty caused by a lack of data or knowledge, respectively), but often we content ourselves with its expectation as a point estimator {math}`\hat{p}`:

```{math}
\hat{p}(y^* \mid x^* , \mathcal{D} ) = 
\mathbb{E}_{\theta \mid \mathcal{D}}\left[ p(y^* \mid x^*, \theta) \right]  =
\int_\Theta p(y^* \mid x^*, \theta) 
p(\theta \mid \mathcal{D}) \rm{d}\theta \, .
```

Simulating or sampling from {math}`p(y^* \mid x^*, \theta)`, or from {math}`\hat{p}(y^* \mid x^* )`, or {math}`p(y^* \mid x^* , \hat{\theta})` for some point estimator {math}`\hat{\theta}` is not considered explicitly in this chapter, although it is worth noting that the posterior predictive is often better behaved than the parameter posterior. One primary reason for this may be that it eliminates the non-identifiability arising from exchangeability of the weights on a given layer.

For the remainder of this chapter, we will consider sampling from a generic target distribution {math}`\pi(\theta) = {\kappa}(\theta)/Z`, where {math}`Z=\int_\Theta \kappa(\theta) \rm{d}\theta` may or may not be known. The reader can think of {math}`\pi(\theta):=p(\theta \mid \mathcal{D})` as the posterior already introduced in Equation [](#eq:bayes_sampling), but where we drop the data {math}`\mathcal{D}` for notational convenience.

+++
(sec:mc)=
## Monte Carlo

The most fundamental sampling algorithm is the Monte Carlo algorithm {cite:p}`robert1999monte`, and it is the basis for many other sampling schemes. Monte Carlo works as follows. Suppose we have a random variable {math}`\theta \in \Theta` from a distribution which is continuous on {math}`\mathbb{R}^P`. Let {math}`\pi` be the probability density function for {math}`\theta` and assume that we are able to simulate independent and identically distributed (i.i.d.) samples from {math}`\pi`. Note that we may refer to {math}`\pi` as the distribution itself. For some function of interest {math}`h`, which is integrable with respect to {math}`\pi`, we wish to estimate an expectation

```{math}
\mathbb{E}[h(\theta)] = \int h(\theta)\pi(\theta)\rm{d}\theta,
```

which is often not available in closed form. Quadrature methods {cite:p}`davis1984methods` are suitable in low parameter dimension {math}`P \leq 5`, and with tricks up to {math}`P \approx 20` {cite:p}`genz1980adaptive` or even {math}`P \approx 100` {cite:p}`bungartz2004sparse`. However, for large {math}`P`, Monte Carlo reigns supreme due to its remarkable ability to achieve a mean square error (MSE), defined as {math}`\text{MSE} = \mathbb{E}\left[(\hat{h}_M - \mathbb{E}[h(\theta)])^2\right]`, that decays at the canonical rate {math}`\mathcal{O}(1/M)` in the number of samples {math}`M`, regardless of dimension or smoothness of the quantity of interest {math}`h`.

The Monte Carlo method is extremely simple to use. We wish to generate {math}`M` independent samples {math}`\theta_1,\theta_2,\ldots,\theta_M` as shown in Algorithm [%s](#alg:monte-carlo).

:::{prf:algorithm} Monte Carlo Algorithm
:label: alg:monte-carlo

- **Inputs:** target density $\pi$
- **Output:** $\{\theta_i\}_{i=1}^M$
1. **For** $i = 1$ to $M$:
    1. simulate $\theta_i \sim \pi$ independently
1. **Return** $\{\theta_i\}_{i=1}^M$
:::

Using the Monte Carlo samples {math}`\{\theta_i\}_{i=1}^M` from {math}`\pi`, we can estimate the expectation as follows

```{math}
:label: eq:mc

\mathbb{E}[h(\theta)] 
\approx \hat{h}_M := \frac1M \sum_{i=1}^M h(\theta_i) \, .
```

This beautiful equation has remarkable convergence properties, and it is this gold standard that we aim to achieve with other sampling algorithms. Since the samples {math}`\theta_1,\dots,\theta_M` are i.i.d. from {math}`\pi`, and writing {math}`h_i = h(\theta_i)`, we have

```{math}
:label: eq:unbiased

\mathbb{E}[\hat{h}_M]
= \mathbb{E}\Bigl[ \tfrac1M\sum_{i=1}^M h_i\Bigr]
= \tfrac1M\sum_{i=1}^M \mathbb{E}[h_i]
= \mathbb{E}[h(\theta)].
```

Thus {math}`\hat{h}_M` is an *unbiased* estimator of the target integral. Moreover, we can increase the accuracy of our estimator by increasing the number of samples {math}`M`. By the strong law of large numbers, if {math}`\mathbb{E}|h(\theta)|<\infty`, we can guarantee that the Monte Carlo estimator converges almost surely to the true value as the number of samples grows without bound. Formally,

```{math}
:label: eq:SLLN

\frac{1}{M} \sum_{i=1}^M h(\theta_i) \;\to\; \mathbb{E}[h(\theta)]
\quad \text{almost surely as } M\to\infty.
```

That is, with probability one the estimator will eventually get arbitrarily close to the truth. If we invoke the central limit theorem, then we can also quantify how accurate {math}`\hat{h}_M` is for a given finite {math}`M`. Define the variance

```{math}
V \;=\; \int \{h(\theta)-\mathbb{E}[h(\theta)]\}^2 \,\pi(\theta)\,\mathrm{d}\theta,
```

and assume {math}`V<\infty`. Then as {math}`M\to\infty`,

```{math}
\sqrt{M}\,\left(\frac{\hat{h}_M - \mathbb{E}[h(\theta)]}{\sqrt{V}}\right)
\;\xrightarrow{\mathsf{D}}\; \mathcal{N}(0,1).
```

Equivalently, for large {math}`M`,

```{math}
\hat{h}_M \;\sim\; \mathcal{N}\Bigl(\mathbb{E}[h(\theta)],\,V/M\Bigr),
```

so that the standard error of the Monte Carlo estimator is {math}`\sqrt{V/M}` and its MSE decays at the canonical rate {math}`O(M^{-1})` (i.e. root‐MSE {math}`O(M^{-1/2})`). This {math}`M^{-1/2}` convergence is independent of the dimension {math}`P` or smoothness of {math}`h`, and is the benchmark against which all more advanced sampling methods are compared.

Often, a challenge is that we cannot sample from our target probability distribution {math}`\pi` directly, and can only evaluate or sample from a non-negative unbiased estimator of {math}`\kappa`. Addressing these challenges has led to the development of the algorithms presented in the remainder of this chapter.

+++
(sec:reject)=
## Rejection Sampling

Suppose that {math}`q(\theta)` is a probability density function from which we *can* sample relatively easily and that {math}`\pi(\theta)` is the target density from which we would like to sample. Suppose in addition that there exists a constant, {math}`c > 0`, such that

```{math}
\frac{\pi(\theta)}{q(\theta)} \leq c,
```

whenever {math}`\pi(\theta)>0`. Then, we can formulate an algorithm for simulating {math}`M` samples from {math}`\pi(\theta)` as given in Algorithm [%s](#alg:rejection-sampling).

:::{figure} assets/intro/rs_beta_3plots.png
:label: fig:rejection
:align: center

Rejection sampling from a Beta$(2,2)$ target $\pi(y)$ using a uniform proposal $q(y)$. Left: $q$ alone does not dominate $\pi$, but the envelope $cq$ with $c=3/2$ does. Right: the acceptance probability $k(y)=\pi(y)/(cq(y))$, shown also for a Beta$(3,3)$ target ($c=15/8$), for which acceptance is lower.
:::

This algorithm is referred to as *rejection sampling*. We can see that this recipe will be very computationally efficient if the densities {math}`q(\theta)` and {math}`\pi(\theta)` are approximately equal and the constant {math}`c` can be chosen close to 1. In this case, the acceptance probability will generally be close to 1. On the other hand, if {math}`\pi(\theta)` and {math}`q(\theta)` are very different, then many runs of the algorithm may be required before a value is finally accepted.

:::{prf:example}

We illustrate the use of rejection sampling to sample from the Beta(2, 2) distribution. In this case {math}`\pi(\theta) = 6x(1-x), 0 \leq x \leq 1`. Suppose that we select {math}`q(\theta)` to be the uniform density on {math}`(0, 1)`. Then we can see that {math}`\pi/q \leq 3/2` and the conditions for rejection sampling hold. However, we can see that the acceptance probability will often be small, for example if {math}`\theta < 0.25` or {math}`\theta > 0.75` (something that will happen 1/2 of the time) then acceptance happens less than {math}`75\%` of the time, meaning that we will require significantly more samples than in the ideal i.i.d. case. For the Beta(3, 3) distribution, the rejection rate would be even higher. This can be seen from the right panel of Figure [](#fig:rejection).
:::

Despite its attraction of ease of implementation and simulation, the efficiency of rejection sampling relies on (i) having a good proposal distribution and (ii) a bound {math}`c`. Note that it can easily be modified to work with only {math}`\kappa`, but these difficulties remain.

:::{prf:algorithm} Rejection Sampling
:label: alg:rejection-sampling

- **Inputs:** target density $\pi$, proposal density $q$, constant $c > 0$
- **Output:** $\{\theta_i\}_{i=1}^M$
1. $i \gets 1$
1. **While** $i \leq M$:
    1. simulate $\theta' \sim q$ and $u \sim U(0,1)$
    1. **If** $u < \frac{\pi(\theta')}{c q(\theta')}$:
        1. $\theta_i \gets \theta'$
        1. $i \gets i + 1$
1. **Return** $\{\theta_i\}_{i=1}^M$
:::

+++
## Importance Sampling

The importance sampling estimator of {math}`\mathbb{E}_\pi[h(\theta)]` begins by re-writing the integral as follows

```{math}
\int_\Theta h(\theta)\pi(\theta){\rm d}\theta = 
\int \frac{\pi(\theta)}{q(\theta)}h(\theta)q(\theta){\rm d}\theta \, ,
```

where {math}`q` is a density such that {math}`q(\theta)>0`, whenever {math}`\pi(\theta)h(\theta)\neq 0`. We now produce i.i.d. samples {math}`(\theta_1,\ldots,\theta_M)` from {math}`q`, and estimate

```{math}
\hat{h} = \frac{1}{M}\sum^M_{i=1}\frac{\pi{(\theta_i)}}{q{(\theta_i)}}h(\theta_i) = \frac{1}{M}\sum^M_{i=1}w(\theta_i)h(\theta_i) \, .
```

We call this procedure *importance sampling*. The density {math}`q` is called the proposal or instrumental density and {math}`w(\theta_i)=\frac{\pi{(\theta_i)}}{q{(\theta_i)}}` are the *importance weights*. Note that {math}`\hat{h}` is an unbiased Monte Carlo estimator of {math}`\mathbb{E}_\pi[h(\theta)]`, as shown in Equation [](#eq:unbiased). There are two reasons why we might be interested in performing importance sampling:

*   Sampling from {math}`\pi(\theta)` is not possible or too expensive.

*   The function {math}`h(\theta)` has a large variance, so the conventional unbiased estimator has large Monte Carlo error.

Recalling the results of Section [Monte Carlo](#sec:mc), the effective test function is now {math}`(\pi/q)h`, and so its regularity dictates the convergence behaviour. Hence, if possible, we should choose a proposal density {math}`q` such that the variance of {math}`(\pi/q)h` is small. It can be shown that the optimal proposal distribution is {math}`q \propto \pi |h|`, however that does little good by itself, since we are back to the same problem of approximating {math}`q`. However, it can provide useful guidance.

:::{prf:example}

Let {math}`\pi=\mathcal{N}(0,1)` and {math}`h = \mathbb{1}\left(\theta > 4\right)`, where {math}`\mathbb{1}` is the indicator function taking the value 1 if the argument is true and 0 otherwise. Since we can simulate i.i.d. from {math}`\pi`, it is tempting to do so, i.e. let {math}`\theta_{1},...,\theta_{M} \sim \pi` i.i.d. and set

```{math}
\hat{h} = \frac{1}{M}\sum^M_{i=1} h(\theta_i) \, .
```

The number of samples {math}`M` would have to be very large to get a non-zero estimator. An alternative is to produce an i.i.d. sample from an *exponential* random variable with rate 1 translated to the right by 4. In this case, we obtain samples {math}`\theta > 4` such that

```{math}
w(\theta) = \frac{\pi(\theta)}{q(\theta)} = \frac{1}{\sqrt{2\pi}}\exp\left(-\frac{\theta^2}{2}+(\theta-4)\right) \, .
```
:::

Suppose that we only know how to evaluate {math}`\kappa`. It is easy to see that an unbiased estimator of the normalizing constant can be built using the following identity

```{math}
Z = \int_\Theta \kappa(\theta) {\rm d}\theta =
\int_\Theta \frac{\kappa(\theta)}{q(\theta)} q(\theta) {\rm d}\theta \, .
```

It is left as an exercise to the reader to build a consistent (but biased) self-normalized importance sampling estimator using only {math}`\kappa`. Unlike rejection sampling, importance sampling does not waste samples through the rejection step, which could be a high amount depending on the application at hand.

As alluded to already, a major disadvantage of importance sampling is that it can result in high variance. There are numerous methods beyond what has been introduced here to alleviate such issues, including the *sequential importance sampling and resampling* methods which will be introduced later in Section [Sequential Monte Carlo](#sec:smc).

+++
(sec:mcmc)=
## Markov chain Monte Carlo

The preceding sections have introduced Monte Carlo integration and two classical approaches—rejection sampling and importance sampling—for drawing samples from a target distribution {math}`\pi(\theta)`, typically the posterior in Bayesian inference. While these methods are simple to use and theoretically well-founded, they suffer significant limitations in high-dimensional settings. Rejection sampling becomes increasingly inefficient as the dimension grows, due to the difficulty of finding proposal distributions that tightly bound the target. Importance sampling, although more flexible, tends to suffer from high variance unless the proposal distribution is well matched to the target {cite:p}`agapiou2017importance,chatterjee2018sample`—a condition that is notoriously hard to satisfy in practice, especially in the multimodal, heavy-tailed, and high-dimensional posteriors encountered in Bayesian deep learning {cite:p}`papamarkou2024position`.

Markov Chain Monte Carlo (MCMC) provides a powerful and general framework to overcome these challenges. Rather than attempting to generate independent samples from {math}`\pi(\theta)`, MCMC constructs a *Markov chain* whose stationary distribution is {math}`\pi` itself. By simulating a trajectory through parameter space in such a way that, asymptotically, the empirical distribution of visited states converges to the target, MCMC transforms the task of independent sampling into one of designing a suitable transition mechanism that ensures both *ergodicity* and *invariance* with respect to {math}`\pi` {cite:p}`tierney1994markov`.

In practical terms, MCMC methods produce a correlated sequence of samples {math}`\{\theta_1, \theta_2, \ldots, \theta_M\}`, which, under suitable conditions, can still be used to approximate expectations under {math}`\pi` using ergodic averages. The canonical estimator takes the form:

```{math}
\hat{h}_M = \frac{1}{M} \sum_{i=1}^M h(\theta_i),
```

which, despite the dependence between samples, converges to the true posterior expectation {math}`\mathbb{E}_\pi[h(\theta)]` under mild regularity assumptions {cite:p}`geyer2011introduction`.

In what follows, we introduce several fundamental MCMC algorithms, starting with the *Metropolis–Hastings algorithm*, which provides a generic recipe for constructing {math}`\pi`-invariant Markov chains from arbitrary proposal distributions {cite:p}`metropolis1953equation,hastings1970monte`. We then examine key special cases—random-walk Metropolis, the Metropolis-adjusted Langevin algorithm (MALA), and Hamiltonian Monte Carlo (HMC)—that are particularly relevant in the context of high-dimensional posterior inference in Bayesian deep learning. Throughout, we pay close attention to algorithmic trade-offs, scaling behaviour {cite:p}`roberts1997weak,roberts1998optimal,fearnhead2025scalable`, and diagnostic tools for assessing convergence and sampling efficiency {cite:p}`brooks1998general`.

(sec:mh)=
### Metropolis–Hastings

Markov chain Monte Carlo (MCMC) proceeds by generating a Markov chain {math}`\theta_1 \to \theta_2\to\ldots` with transition kernel {math}`\mathcal{M}`, described by a density {math}`p(\theta,\theta')` on {math}`\Theta^2`, which is invariant with respect to the target posterior {math}`\pi`:

```{math}
:label: eq:invariance

\int_{\Theta}\pi(\theta')\,p(\theta',\theta)\,\mathrm{d}\theta' =\pi(\theta).
```

The Metropolis–Hastings (MH) algorithm achieves this invariance by first choosing an arbitrary proposal density {math}`q(\theta,\theta')` that proposes {math}`\theta'` given the current iterate of the Markov chain {math}`\theta`, and then accepting or rejecting each proposed move so that detailed balance—and hence Equation [](#eq:invariance)—holds.

Concretely, we can define the MH transition kernel as

```{math}
p(\theta,\theta') = q(\theta,\theta')\alpha(\theta,\theta'),
```

where the acceptance probability is

```{math}
\alpha(\theta,\theta') = \min\left\{
1,\,
\frac{\pi(\theta')\,q(\theta',\theta)}
     {\pi(\theta)\,q(\theta,\theta')}
\right\}.
```

Because {math}`\pi(\theta)\propto\kappa(\theta)`, any unknown normalisation cancels and only the unnormalized density {math}`\kappa` is needed. See Algorithm [%s](#alg:mh) for a full implementation.

:::{prf:algorithm} Metropolis-Hastings Algorithm
:label: alg:mh

- **Inputs:** unnormalized target density $\kappa$, proposal distribution $q$
- **Output:** $\{\theta_i\}_{i=1}^M$
1. Generate initial sample $\theta_0 \sim p$, with $\pi \ll p$ (absolute continuity)
1. **For** $i = 0$ to $M - 1$:
    1. propose new state $\theta' \sim q(\theta_i, \cdot)$
    1. compute the acceptance probability
    $$
\alpha(\theta_i,\theta') =
      \min\left\{1,
      \frac{\pi(\theta') q(\theta',\theta_i)}
           {\pi(\theta_i) q(\theta_i,\theta')}
      \right\}
      = \min\left\{1,
      \frac{\kappa(\theta') q(\theta',\theta_i)}
           {\kappa(\theta_i) q(\theta_i,\theta')}
      \right\}
    $$
    1. draw $u \sim \mathcal{U}[0,1]$
    1. **If** $u \leq \alpha(\theta_i,\theta')$:
        1. $\theta_{i+1} \gets \theta'$
    2. **Else:**
        1. $\theta_{i+1} \gets \theta_i$
1. **Return** $\{\theta_i\}_{i=1}^M$
:::

:::{prf:proposition}
:label: prop:mh

The Metropolis-Hastings algorithm satisfies the equation [](#eq:invariance).
:::

:::{prf:proof}
:enumerated: false

Let us first assume the proposal is accepted. Recalling the definition of {math}`p(\theta,\theta')` we have

```{math}
\begin{align}
p(\theta,\theta') = q(\theta,\theta') \times \alpha(\theta,\theta')
= \min\bigg\{q(\theta,\theta'),  \frac{\pi(\theta') q(\theta',\theta)}{\pi(\theta)}\bigg\}.
\end{align}
```

It follows that

```{math}
\begin{align}  
\pi(\theta)p(\theta,\theta') &= \min\{\pi(\theta)q(\theta,\theta'),\pi(\theta')q(\theta',\theta)\} 
= \pi(\theta') p(\theta',\theta) \, ,
\end{align}
```

by symmetry. The symmetry of the rejection case is left as an exercise. This property is called *detailed balance* and is a sufficient condition for invariance Equation [](#eq:invariance).
:::

Invariance with respect to {math}`\pi` is a necessary condition for building convergent estimators from a Markov chain. The chain also needs to have certain other stability properties, collectively known as *ergodicity*, to ensure empirical averages Equation [](#eq:mc) built from its output converge. A desirable and achievable property is called *geometric ergodicity*, which essentially guarantees linear convergence, i.e. the dependence on the initial condition vanishes geometrically.

(sec:rwm)=
#### Random walk Metropolis

A particularly simple—and historically important—special case of the Metropolis–Hastings algorithm is the *random–walk Metropolis* (RWM) proposal, in which one sets

```{math}
q(\theta,\theta') = q(\theta' - \theta),
```

where {math}`q` is any symmetric density, {math}`q(\theta)=q(-\theta)`. The canonical choice is

```{math}
\theta' = \theta + \epsilon,
\qquad
\epsilon\sim \mathcal{N}\bigl(0,\;\lambda^{2}I_{P}\bigr),
```

so that {math}`\theta' \sim \mathcal{N}\left(\theta,\lambda^{2}I_{P}\right)`, i.e. {math}`q(\theta,\cdot) =\mathcal{N}\left(\theta,\lambda^{2}I_{P}\right)`. Due to symmetry, the Metropolis acceptance probability simplifies to

```{math}
\alpha(\theta,\theta')
=
\min\left\{1,\;
\frac{\kappa(\theta')}
     {\kappa(\theta)}\right\}.
```

**Local moves and tuning.** RWM proposals are inherently *local*: each candidate {math}`\theta'` lies in a neighbourhood of {math}`\theta`. If {math}`\lambda` is too small, the chain makes only tiny steps and mixes very slowly; if {math}`\lambda` is too large, most proposals fall in low-density regions and are rejected. Optimal scaling results show that, for a wide class of {math}`P`-dimensional targets, the stationary acceptance rate tends to {math}`\approx0.234` as {math}`P\to\infty` when {math}`\lambda\propto P^{-1/2}` {cite:p}`roberts1997weak`. In practice, one therefore adapts {math}`\lambda` so that the empirical acceptance rate lies between about {math}`20\%` and {math}`30\%`.

**Preconditioning.** When components of {math}`\theta` have very different scales or strong correlations, one can replace {math}`\lambda^2 I_{P}` by {math}`\lambda^2 V`, where {math}`V` is a positive-definite “preconditioning” matrix (often an estimate of the posterior covariance). This *preconditioned RWM* retains symmetry and thus the same simplified acceptance probability, but can yield orders-of-magnitude improvements in computational efficiency, i.e. less samples required to sample the target, because the proposed parameters {math}`\theta'` will be more closely aligned with the posterior distribution, and thus less likely to be rejected.

#### Gibbs Sampler

Given a multi-variate {math}`\theta = (\theta_1, \theta_2 \dots \theta_n)`, it may be intractable to sample all at once. However if the full conditional distributions for each coordinate are known, then it becomes possible to apply *Gibbs sampling*.

The Gibbs sampling algorithm is fairly straightforward. Given the proposals are known, the Gibbs algorithm updates the sample by updating each coordinate one at a time, i.e. {math}`\theta_i' \sim P(\theta_i \mid \theta_{\neg i})`, where {math}`\theta_{\neg i}` denotes all coordinates except {math}`i`. While it may be tempting to update all coordinates at the same time, this would not properly preserve the joint distribution as all coordinates would be updated independently. We include a simple formulation in Algorithm [%s](#alg:intro:gibbs).

:::{prf:algorithm} Gibbs Sampler
:label: alg:intro:gibbs

- **Inputs:** initial state $\theta = (\theta_1, \theta_2 \dots \theta_n)$; full conditional distributions $P(\theta_{i} \mid \theta_{\neg i})$ for $i \in \{1,\dots,n\}$
- **Output:** sample set $\mathcal{S}$
1. $\mathcal{S} \gets \emptyset$
1. **For** each iteration:
    1. **For** $i \in \{1,\dots,n\}$:
        1. $\theta_i' \sim P(\theta_i \mid \theta_{\neg i})$
        1. $\theta \gets (\theta_1, \theta_2, \dots \theta_i', \dots \theta_n)$
        1. $\mathcal{S} \gets \mathcal{S} \cup \{\theta\}$
:::

Given that the Gibbs sampler is drawing from the correct proposal, the typical Metropolis-Hastings acceptance step is not needed in this case. However, this sampler suffers from slow and highly correlated samples due to the nature of the single coordinate update approach.

It is possible to extend the initial Gibbs sampler to a block-style approach in the case where the joint distribution over multiple coordinates is known. Instead of the single-coordinate marginal, if we know the joint distribution over one-set of coordinates conditioned on another set, it is possible to quickly sample by alternative sampling from each joint distribution. This is known as the Block-Gibbs sampler. More formally, if we have two sets of coordinates {math}`I, J`, the block Gibbs sampler proceeds by sampling {math}`\theta'_I \sim P(\theta_I \mid \theta_J)`, and then sampling {math}`\theta'_J \sim P(\theta_J \mid \theta'_I)`. This sampling algorithm avoids the pitfalls of the typical Gibbs sampler. However, it is only possible to use this in cases where the conditional distributions over blocks of coordinates is known.

(sec:mala)=
#### Metropolis-adjusted Langevin Algorithm

The random-walk Metropolis ignores gradient information about {math}`\pi`. The Metropolis–Adjusted Langevin Algorithm (MALA) rectifies this by using an Euler–Maruyama discretisation of the overdamped Langevin diffusion

```{math}
\mathrm{d}\theta_{t} = \tfrac12\nabla\!\log\pi(\theta_{t})\,\mathrm{d}t + \mathrm{d}W_{t},
```

where {math}`\{W_t\}_{t\geq 0}` is standard Brownian motion on {math}`\mathbb{R}^{P}`, and whose stationary law is {math}`\pi`. Concretely, the MALA proposal is

```{math}
\theta' | \theta
\sim
\mathcal{N}\left(
  \theta +\tfrac{\lambda^{2}}{2}\,\nabla\log\kappa(\theta),
  \lambda^{2}I_{P}
\right).
```

This drift term {math}`\nabla\log\kappa(\theta)` biases proposed moves towards regions of higher posterior density, enabling larger step-sizes than RWM and better acceptance rates. The MALA proposal also forms the basis for many other wonderful Monte Carlo algorithms, including the stochastic gradient Langevin dynamics (SGLD) {cite:p}`welling2011bayesian` and all other stochastic gradient MCMC (SG-MCMC) algorithms, which will be the focus of the next chapter.

**Gradient-based tuning.** Under regularity conditions, choosing {math}`\lambda\propto P^{-1/6}` yields an optimal stationary acceptance rate of approximately 0.574 as {math}`P\to\infty` {cite:p}`roberts1998optimal`. In applications, one often adapts {math}`\lambda` to target an acceptance rate of about {math}`50\%` to {math}`60\%`. This can be done by monitoring the acceptance rate and adjusting {math}`\lambda` accordingly, or by using more sophisticated adaptive schemes that adjust the step size based on the empirical gradient information.

**Preconditioned MALA and robustness.** Analogous to RWM, one can introduce a symmetric preconditioning matrix {math}`V`, replacing {math}`I_{P}` by {math}`V` both in the covariance and in the drift:

```{math}
\theta'
\sim
\mathcal{N}\left(
  \theta + \tfrac{\lambda^{2}}{2}\,V\,\nabla\log\kappa(\theta),
  \lambda^{2}V
\right).
```

Preconditioning often dramatically accelerates convergence on anisotropic posteriors. However, because MALA relies on {math}`\nabla\log\kappa`, it can be sensitive to regions where the gradient is large or ill-behaved (e.g. lighter-than-Gaussian tails). Common mitigations include gradient clipping or fallback to RWM outside of a “trust region.”

(sec:mcmc-perf)=
#### Monitoring Performance

Assessing the quality of an MCMC run is essential to ensure that empirical averages {math}`\hat{h}_{M}=\tfrac1M\sum_{i=1}^M h(\theta_i)` reliably approximates {math}`\mathbb{E}_{\pi}[h(\theta)]`. Three widely used diagnostics are:

**1. Autocorrelation and Integrated Autocorrelation Time.** For a scalar summary of {math}`h(\theta)`, define the lag-{math}`k` autocorrelation

```{math}
\rho_k =
\frac{\mathrm{Cov}\bigl[h(\theta_i),\,h(\theta_{i+k})\bigr]}
     {\mathrm{Var}_{\pi}\bigl[h(\theta)\bigr]}.
```

As the Markov chain is stationary, {math}`\rho_0=1`, and {math}`\rho_k\to0` as {math}`k\to\infty`. The slower {math}`\rho_k` decays, the more correlated is the chain. One can show that, via the Markov-chain central limit theorem,

```{math}
\sqrt{M}\bigl(\hat{h}_{M}-\mathbb{E}_{\pi}[h]\bigr)
\xrightarrow{\mathcal{D}}
\mathcal{N}\bigl(0,V_{\mathrm{eff}}\bigr),
\quad \mathrm{where} \quad
V_{\mathrm{eff}} =
\mathrm{Var}_{\pi}[h]\,\Bigl(1 + 2\sum_{k=1}^\infty \rho_k\Bigr).
```

The quantity {math}`\tau_{\mathrm{int}}=1+2\sum_{k=1}^\infty\rho_k` is the *integrated autocorrelation time*, and the *effective sample size* {math}`\mathrm{ESS}=M/\tau_{\mathrm{int}}` measures the number of independent draws represented by {math}`M` correlated samples. Figure [](#fig:acf) illustrates autocorrelation decay. A related metric is the *expected squared jumping distance* (ESJD) {math}`\mathbb{E}[\|\theta_{i+1}-\theta_i\|^2]`, which balances move size against acceptance rate. Larger moves will reduce the correlation between elements of the Markov chain, however, larger moves are also less likely to be accepted, which means that the chain does not evolve. Therefore, the challenge for practitioners, is to balance between small moves (high correlation and high acceptance rate) and large moves (low correlation and low acceptance rate).

**2. Trace Plots.** Plotting the sequence {math}`\{h(\theta_i)\}_{i=1}^M` versus iteration {math}`i` provides a simple-to-interpret visual representation of mixing and stationarity. Well‐mixed chains rapidly traverse the bulk of the posterior, showing no apparent trends or “stickiness” (see Left Panel Figure [](#fig:trace)). Conversely, slow mixing or multimodality is signalled by long periods of drift or entrapment in subregions of the posterior (see Right Panel Figure [](#fig:trace)). Trace plots are a useful visual tool for diagnosing the efficiency of an MCMC algorithm, however, they tend to be most useful for low-dimensional posteriors as it is easier to visualise all of the parameters {math}`\theta`, whereas, for high-dimensional posteriors, visualising all of the parameters in the Markov chain individually is often impractical.

**3. Potential Scale Reduction (Gelman–Rubin) Diagnostic.** When multiple chains {math}`\{\theta_i^{(c)}\}` are run from overdispersed starting points, one can compare the *between-chain variance* {math}`B` to the *within-chain variance* {math}`W`. The *potential scale reduction factor*

```{math}
\widehat{R}
=
\sqrt{\frac{\tfrac{M-1}{M}W + \tfrac{1}{M}B}{W}}
```

tends to 1 as all chains converge to the same target distribution {cite:p}`gelman1992inference`. Values of {math}`\widehat{R}<1.1` are commonly taken to indicate satisfactory convergence, after which samples from all chains may be pooled to produce an approximation to the posterior {math}`\pi`.

In practice, one would usually employ a combination of these diagnostics: trace plots for qualitative assurance, autocorrelation and ESS for quantitative accuracy, and {math}`\widehat{R}` to guard against failure to explore multimodal targets.

:::{prf:example}

We present a Metropolis-Hastings experiment aiming to sample from a target {math}`\pi(\theta)=N(0,1)`. Our proposal is based on a random walk scheme (Section [Random walk Metropolis](#sec:rwm)) {math}`\theta' = \theta + \epsilon`, where {math}`\epsilon \sim \mathcal{N}(0, \lambda^{2} I)` is additive Gaussian noise. We run two different experiments, with different choices of step size parameter {math}`\lambda \in \{0.01,0.9\}`. Our simulations are plotted below in Figures [%s](#fig:trace) and [%s](#fig:acf). For the larger choice of {math}`\lambda`, we notice that the chain explores the space well as the iterations progress, with the ACF decorrelating rapidly. However, for the smaller value of {math}`\lambda`, the chain wanders slowly and mixes poorly, and the ACF decreases much slower, remaining close to one at moderate lags.
:::

:::{figure}
:label: fig:trace

```{image} assets/intro/trace_good.png
:width: 45%
```

```{image} assets/intro/trace_bad.png
:width: 45%
```

Trace plots for $\lambda=0.9$ (left) and $\lambda=0.01$ (right).
:::

:::{figure}
:label: fig:acf

```{image} assets/intro/acf_good.png
:width: 45%
```

```{image} assets/intro/acf_bad.png
:width: 45%
```

Corresponding ACFs for $\lambda=0.9$ (left) and $\lambda=0.01$ (right).
:::

(sec:hmc)=
### Hamiltonian Monte Carlo

Random–walk Metropolis (RWM) requires proposal steps of size {math}`\lambda=O(d^{-1/2})` to maintain a non–degenerate acceptance rate in dimension {math}`d`, whereas MALA can take steps of size {math}`O(d^{-1/6})`. *Hamiltonian Monte Carlo* (HMC) goes a step further: with an integration step size {math}`\epsilon=O(d^{-1/4})` it still enjoys an acceptance rate bounded away from 0 while exploring much larger regions of parameter space between consecutive accept–reject decisions {cite:p}`beskos2013optimal,duane1987hybrid,neal2011mcmc,cobb2021scaling`. The key is to embed the {math}`d`‐dimensional parameter {math}`\theta` in a {math}`2d`-dimensional *phase space* and simulate approximate Hamiltonian dynamics that preserve a surrogate “total energy”.

**Extended target and Hamiltonian.** Introduce an auxiliary *momentum* variable {math}`z\in\mathbb{R}^d` and a positive–definite *mass matrix* {math}`M_0`. Define the Hamiltonian

```{math}
H(\theta,z)=U(\theta)+K(z)
:= -\log\pi(\theta)+\tfrac12 z^{\top}M_0^{-1}z,
```

so that the joint density {math}`\widetilde\pi(\theta,z)\propto\exp[-H(\theta,z)]` has marginal {math}`\pi(\theta)` and independent {math}`z\sim\mathcal{N}(0,M_0)`.

**Hamiltonian dynamics.** The continuous equations of motion are

```{math}
:label: eq:hmc-continuous

\frac{\mathrm{d}\theta}{\mathrm{d}t}=M_0^{-1}z,
\qquad
\frac{\mathrm{d}z}{\mathrm{d}t}=\nabla_{\theta}\log\pi(\theta),
```

which conserve {math}`H` exactly and therefore move along surfaces of constant joint density. Exact simulation is impossible for general targets {math}`\pi`, so HMC uses the second–order, symplectic *leapfrog* integrator {cite:p}`leimkuhler2004simulating` with step size {math}`\Delta t`:

```{math}
:label: eqn:leapfrog_step

\begin{align}
z_{t+\tfrac12\Delta t}
&=z_t+\tfrac{\Delta t}{2}\nabla_{\theta}\!\log\pi(\theta_t),
\notag\\
\theta_{t+\Delta t}
&=\theta_t+\Delta t\,M_0^{-1}z_{t+\tfrac12\Delta t},
\\
z_{t+\Delta t}
&=z_{t+\tfrac12\Delta t}+\tfrac{\Delta t}{2}
  \nabla_{\theta}\!\log\pi(\theta_{t+\Delta t}).
\notag
\end{align}
```

One repeats this map {math}`L` times, with {math}`\epsilon=\Delta t` and {math}`T=L\,\epsilon` is the *integration time*.

**HMC kernel.** Starting from {math}`\theta_{i-1}`:

1.  Draw momentum {math}`z\sim\mathcal{N}(0,M_0)`.

2.  Apply {math}`L` leapfrog steps Equation [](#eqn:leapfrog_step) to obtain {math}`(\theta',z')`.

3.  Accept {math}`(\theta',z')` with probability

    ```{math}
    \alpha
          =
          \min\Bigl\{1,\,
          \exp\bigl[H(\theta_{i-1},z)-H(\theta',z')\bigr]\Bigr\}.
    ```

4.  Set {math}`\theta_{i}=\theta'` if accepted, otherwise {math}`\theta_{i}=\theta_{i-1}`; discard {math}`z'`.

Because the leapfrog map is volume–preserving (i.e. the Jacobian {math}`=1`) and symmetric (its inverse is obtained by {math}`(\theta,z)\mapsto(\theta,-z)` and running the {math}`L` steps backward), the Metropolis correction guarantees {math}`\widetilde\pi` is invariant; the posterior {math}`\pi` is therefore stationary for the marginal chain on {math}`\theta`.

:::{prf:algorithm} Hamiltonian Monte Carlo Kernel
:label: alg:hmc_kernel

- **Inputs:** initial state $\theta_{0}$, target distribution $\pi$, mass matrix $M_0$
- **Output:** $\{\theta_i\}_{i=1}^M$
1. **For** $i = 1$ to $M$:
    1. generate initial momentum $z \sim \mathcal{N}(0, M_0)$
    1. **For** $l = 1$ to $L$:
        1. generate $\theta_{l\Delta t}$ and $z_{l\Delta t}$ from Equation [](#eqn:leapfrog_step)
        1. applied at $(\theta_{(l-1)\Delta t}, z_{(l-1)\Delta t})$, with $\theta_0 = \theta_{i-1}$ and $z_0 = z$
    1. let $(\theta', z') \gets (\theta_{L\Delta t}, z_{L\Delta t})$ and generate $u \sim U[0,1]$
    1. **If** $u \leq \min \left\{ 1, \exp\left[H(\theta, z) - H(\theta', z')\right] \right\}$:
        1. $\theta_i \gets \theta'$
    2. **Else:**
        1. $\theta_i \gets \theta_{i-1}$
1. **Return** $\{\theta_i\}_{i=1}^M$
:::

**High‐dimensional scaling.** For product targets {math}`\pi(\theta)=\prod_{i=1}^d f(\theta^i)` with {math}`M_0=I_d`, one can show {cite:p}`fearnhead2025scalable` that to keep a well-behaved acceptance rate, one must scale {math}`\epsilon = O(d^{-1/4})`, hence {math}`L=O(d^{1/4})`. The computational cost per effectively independent draw is therefore {math}`O(d^{1/4})`, compared with {math}`O(d^{1/3})` for MALA and {math}`O(d)` for RWM.

**Practical tuning.**

*   **Step size {math}`\epsilon`.** During warm–up, adapt {math}`\epsilon` (e.g. with dual‐averaging) to target an acceptance rate in the {math}`60\text{–}80\%` range; theory suggests {math}`65\%` is asymptotically optimal {cite:p}`beskos2013optimal`.

*   **Trajectory length {math}`T=L\,\epsilon`.** Too small {math}`T` gives RWM‐like behaviour; too large wastes computation and may return to near the starting point. Popular heuristics randomise {math}`L` or employ the No‐U‐Turn criterion {cite:p}`hoffman2014no`.

*   **Mass matrix {math}`M_0`.** Setting {math}`M_0` to an estimate of the posterior covariance (or its diagonal) acts like the preconditioning matrices used with RWM and MALA, dramatically improving convergence when parameters are on different scales.

Because HMC combines long, momentum‐driven proposals with high acceptance probabilities, it has become the default sampler in many Bayesian deep‐learning applications, where posteriors are high‐dimensional, strongly correlated and feature narrow valleys that thwart simpler MCMC methods.

+++
(sec:smc)=
## Sequential Monte Carlo

Sequential Monte Carlo (SMC) methods are a set of Monte Carlo algorithms designed to sample from a sequence of target probability densities {math}`\{\pi_n(\theta_{1:n})\}` of increasing dimension, where each one is defined on the product space {math}`\Theta^n = \prod_{i=1}^n \Theta` {cite:p}`doucet2001sequential,del2004feynman`. This is referred to as the *smoothing distribution* in the context of state-space models. Often one may be interested only in the time {math}`n` marginal {math}`\pi_n(\theta_n)`, which is known as the *filtering distribution*. Suppose

```{math}
\begin{align}
\pi_n(\theta_{1:n}) = \frac{\kappa_n(\theta_{1:n})}{Z_n} \, ,
\end{align}
```

where the un-normalized density admits the decomposition

```{math}
:label: eq:gammadecomp

\kappa_n(\theta_{1:n}) = \kappa_n(\theta_n | \theta_{1:n-1}) \kappa_{n-1}(\theta_{1:n-1}) \, ,
```

and {math}`\kappa_n(\theta_n | \theta_{1:n-1})` and {math}`\kappa_1(\theta_1)` can be evaluated, and {math}`Z_n` is the unknown normalizing constant given by

```{math}
\begin{align}
Z_n = \int \kappa_n(\theta_{1:n}) \mathrm{d}\theta_{1:n}.
\end{align}
```

Note that we write

```{math}
\int \mathrm{d}\theta_{1:n} \quad \text{ which is equivalent to }\quad \int_{\Theta} \int_{\Theta} \cdots \int_{\Theta} \mathrm{d}\theta_1 \mathrm{d}\theta_2 \cdots \mathrm{d}\theta_n.
```

SMC provides an approximation of {math}`\pi_1(\theta_1)` and an estimate of {math}`Z_1` at time 1 then an approximation of {math}`\pi_2(\theta_{1:2})` and an estimate of {math}`Z_2` at time 2 and so on. The approximations are done sequentially in time:

*   At time step 1, one samples {math}`N` samples {math}`\{\theta_1^{(i)}\}_{i=1}^N` from some given function {math}`q_1(\theta_1)` and then approximates {math}`\pi_1(\theta_1)` and {math}`Z_1` using these samples;

*   At time step 2, sample {math}`\{\theta_2^{(i)}\}_{i=1}^N` from some function {math}`q_2(\theta_2|\theta_1)`, and use the cumulative samples to approximate {math}`\pi_2(\theta_{1:2})` and {math}`Z_2`.

Consider another probability density {math}`q_n(\theta_{1:n}) = q_n(\theta_n | \theta_{1:n-1})q_{n-1}(\theta_{1:n-1})` that is easy to sample from, and such that the support of {math}`q_n(\theta_{1:n})` includes the support of {math}`\pi_n(\theta_{1:n})`. Then, we have the following importance sampling (IS) identities

```{math}
\begin{align}
\pi_n(\theta_{1:n}) &= \frac{\kappa_n(\theta_{1:n})}{Z_n} = \frac{w_n(\theta_{1:n}) q_n(\theta_{1:n})}{Z_n},\\
Z_n&=\int w_n(\theta_{1:n})q_n(\theta_{1:n}) \mathrm{d}\theta_{1:n},
\end{align}
```

where {math}`w_n(\theta_{1:n})` is the unnormalized weight function and in light of Equation [](#eq:gammadecomp) we have the following recursive definition of the weights

```{math}
\begin{align}
w_n(\theta_{1:n})  = \frac{\kappa_n(\theta_{1:n})}{q_n(\theta_{1:n})} =
\frac{\kappa_n(\theta_n | \theta_{1:n-1})}{q_n(\theta_n | \theta_{1:n-1})} 
w_{n-1}(\theta_{1:n-1}) 
=: \alpha(\theta_n | \theta_{1:n-1}) w_{n-1}(\theta_{1:n-1}) \, .
\end{align}
```

Assume we draw, sequentially in time, {math}`M` independent samples {math}`\theta_{1:n}^{(i)}`, {math}`i=1,\cdots,M`, from {math}`q_n(\theta_{1:n})` (see Algorithm [%s](#alg:SIS)). We have the following unbiased estimator of {math}`Z_n`:

```{math}
:label: eq:SIS_Z

\begin{align}

\widehat{Z}_n &= 
\frac{1}{M} \sum_{i=1}^M w_n(\theta_{1:n}^{(i)}).
\end{align}
```

A consistent self-normalized importance sampling estimator of {math}`\mathbb{E}_n(h_n)`, for an arbitrary function {math}`h_n:\Theta^n\to \mathbb{R}`, is given by

```{math}
:label: eq:SIS_expec

\widehat{\mathbb{E}_n}(h_n) 
=\frac{\sum_{i=1}^N  w_n(\theta_{1:n}^{(i)}) h_n(\theta_{1:n}^{(i)}) }{\sum_{i=1}^N w_n(\theta_{1:n}^{(i)})}  \nonumber 
= \sum_{i=1}^M W_n^{(i)} \,\, h_n(\theta_{1:n}^{(i)}) \, ,
```

where

```{math}
\begin{align}
W_n^{(i)} = \frac{w_n(\theta_{1:n}^{(i)})}{\sum_{i=1}^M w_n(\theta_{1:n}^{(i)})} \, .
\end{align}
```

These are the ingredients of the *sequential importance sampling* (SIS) algorithm, given in Algorithm [%s](#alg:SIS). In Algorithms [%s](#alg:SIS) and [%s](#alg:SMC) we take {math}`h_n(\theta_{1:n}) = \theta_n`.

:::{prf:algorithm} Sequential Importance Sampling (SIS) to estimate $\mathbb{E}_n(\theta_n)$ for $n \geq 1$
:label: alg:SIS

- **Inputs:** number of particles $M$, proposal distributions $q_n$, adjustment multipliers $\alpha_n$
- **Output:** estimates $\widehat{\theta}_n$ and $\widehat{Z}_n$ for $n \geq 1$
1. **For** $i = 1$ to $M$:
    1. initialize weights $w_1^{(i)} \gets 1 / M$
1. $n \gets 1$
1. **Repeat until** termination criterion met:
    1. **For** $i = 1$ to $M$:
        1. sample $\theta_n^{(i)} \sim q_n(\theta_n \mid \theta_{1:n-1}^{(i)})$
        1. update unnormalized weight
        $$
w_{n}(\theta_{1:n}^{(i)}) \gets
        \alpha_n(\theta_{n}^{(i)} \mid \theta_{1:n-1}^{(i)})\,
        w_{n-1}(\theta_{1:n-1}^{(i)})
        $$
        1. normalize weight
        $$
W_n^{(i)} \gets
        \frac{w_n(\theta_{1:n}^{(i)})}
             {\sum_{j=1}^M w_n(\theta_{1:n}^{(j)})}
        $$
    1. compute estimates
    $$
\widehat{\theta}_n \gets
      \sum_{i=1}^M W_n^{(i)} \theta_{n}^{(i)}, \qquad
      \widehat{Z}_n \gets
      \tfrac{1}{M} \sum_{i=1}^M w_n(\theta_{1:n}^{(i)})
    $$
    1. $n \gets n + 1$
:::

As {math}`n` grows the weights will degenerate, in the sense that one will dominate the rest. This can be monitored by the effective sample size, which is related to the variance of the weights:

```{math}
{\sf ESS} = \frac{\left(\sum_{i=1}^M w_n^{(i)}\right)^2}{\sum_{i=1}^M (w_n^{(i)})^2} \, .
```

One way to mitigate this somewhat is by resampling, i.e. for {math}`i=1,\dots,M`, let

```{math}
\theta_{1:n}^{(i)} \leftarrow \theta_{1:n}^{(j)} \, , 
\quad {\rm for}\quad j \sim (W_n^{(1)},\dots,W_n^{(M)}) \, , \quad {\rm and}\quad  w_n^{(i)} \leftarrow 1/M \, .
```

Resampling slightly increases the variance of the estimator, but the particles are rejuvenated at the filtering end, and under suitable conditions the filter can even remain stable online for infinite time {cite:p}`del2004feynman`. Note that this branching of the particles forward in time means coalescence backward in time, so the particles still degenerate for small times, which is problematic for the smoother. SIS along with resampling is called sequential importance resampling (SIR) or sequential Monte Carlo (SMC). See Algorithm [%s](#alg:SMC). In the context of a state space model, {math}`q_n` is often chosen as the forward evolution kernel of the hidden process, and in the filtering context this specific choice is often called the *bootstrap particle filter*. In practice, one resamples when the ESS is less than a threshold (e.g. {math}`M/2` or {math}`M/4`). See {cite:t}`chopin2020introduction` for a recent comprehensive introduction.

:::{prf:algorithm} Sequential Monte Carlo (SMC) to estimate $\mathbb{E}(\theta_n)$ for $n \ge 1$
:label: alg:SMC

- **Inputs:** number of particles $M$, proposals $q_n$, adjustment multipliers $\alpha_n$
- **Output:** estimates $\widehat{\theta}_n$ and $\widehat{Z}_n$ for $n \ge 1$
*Note:* Initialization at $n=1$
1. **For** $i=1$ to $M$:
    1. sample $\theta_1^{(i)} \sim q_1(\theta_1)$
    1. compute $w_1(\theta_1^{(i)}) \gets \kappa_1(\theta_1^{(i)})/q_1(\theta_1^{(i)})$
$$
W_1^{(i)} \gets \frac{w_1(\theta_1^{(i)})}{\sum_{j=1}^{M} w_1(\theta_1^{(j)})}
    \quad\text{for } i=1,\dots,M,
    \qquad
    \widehat{Z} \gets 1
$$
*Note:* Iterate for subsequent times
1. $n \gets 1$
1. **Repeat until** termination criterion met:
    *Note:* Effective sample size at time $n$
    $$
ESS_n \gets \left(\sum_{i=1}^{M} \left(W_n^{(i)}\right)^2\right)^{-1}
    $$
    *Note:* Optional resampling at time $n$
    1. **If** $ESS_n \le M/2$:
        *Note:* Multinomial resampling with probabilities $W_n^{(1:M)}$
        1. **For** $i=1$ to $M$:
            1. draw $j \sim \mathrm{Categorical}\left(W_n^{(1)},\dots,W_n^{(M)}\right)$
            1. $\theta_{1:n}^{(i)} \gets \theta_{1:n}^{(j)}$
            1. $w_n^{(i)} \gets 1/M$
        $$
\widehat{Z} \gets \left(\frac{1}{M}\sum_{i=1}^{M} w_n(\theta_{1:n}^{(i)})\right)\widehat{Z}
        $$
    *Note:* Advance to time $n+1$
    1. $n \gets n+1$
    1. **For** $i=1$ to $M$:
        1. sample $\theta_{n}^{(i)} \sim q_{n}(\theta_{n}\mid \theta_{1:n-1}^{(i)})$
        1. update unnormalized weight
        $$
w_{n}(\theta_{1:n}^{(i)}) \gets
        \alpha_{n}(\theta_{n}^{(i)}\mid \theta_{1:n-1}^{(i)})
        w_{n-1}(\theta_{1:n-1}^{(i)})
        $$
    $$
W_{n}^{(i)} \gets
      \frac{w_{n}(\theta_{1:n}^{(i)})}{\sum_{j=1}^{M} w_{n}(\theta_{1:n}^{(j)})}
      \quad\text{for } i=1,\dots,M
    $$
    *Note:* Estimates at time $n$
    $$
\widehat{\theta}_{n} \gets
      \sum_{i=1}^{M} W_{n}^{(i)}\,\theta_{n}^{(i)},
      \qquad
      \widehat{Z}_{n} \gets
      \frac{1}{M}\sum_{i=1}^{M} w_{n}(\theta_{1:n}^{(i)})\,\widehat{Z}
    $$
:::

+++
(sec:intro:summary)=
## Summary

This chapter introduced the Monte Carlo algorithm and a number of methods that develop it. Each addresses the same underlying difficulty: simulating from, or estimating expectations under, a target distribution {math}`\pi(\theta)` that in most applications can only be evaluated up to an unknown normalising constant.

When independent samples from {math}`\pi` are available, Monte Carlo integration (Section [Monte Carlo](#sec:mc)) applies directly. When they are not, rejection sampling (Section [Rejection Sampling](#sec:reject)) and importance sampling construct estimators from an accessible proposal distribution. Both are exact in principle, but become inefficient in high dimension unless the proposal closely matches the target.

Markov chain Monte Carlo (Section [Markov chain Monte Carlo](#sec:mcmc)) removes the need for independent draws by constructing a Markov chain whose stationary distribution is the target. The Metropolis–Hastings algorithm (Section [Metropolis–Hastings](#sec:mh)) provides the general recipe, with the random-walk Metropolis algorithm (Section [Random walk Metropolis](#sec:rwm)), the Gibbs sampler, the Metropolis-adjusted Langevin algorithm (Section [Metropolis-adjusted Langevin Algorithm](#sec:mala)), and Hamiltonian Monte Carlo (Section [Hamiltonian Monte Carlo](#sec:hmc)) as the special cases treated here. Since the samples are correlated, step-size tuning and the convergence diagnostics of Section [Monitoring Performance](#sec:mcmc-perf) are essential in practice.

Sequential Monte Carlo (Section [Sequential Monte Carlo](#sec:smc)) extends importance sampling to a sequence of targets on nested product spaces, as in filtering. Sequential importance sampling with resampling propagates weighted particles forward in time, with resampling when the effective sample size drops.

The methods play complementary roles: rejection and importance sampling suit low-dimensional or well-matched problems, MCMC is the standard tool for high-dimensional Bayesian computation, and sequential Monte Carlo is suited to inference along a sequence. These ideas are taken further in the chapters that follow. For example, stochastic gradient MCMC is covered in Chapter [](#chap:sampling_methods_sg_mcmc), and sequential Monte Carlo *samplers*, which applies SMC to an artificial sequence of intermediate targets with MCMC mutations, with tempering and multilevel extensions for neural network posteriors, in Chapter [](#chap:smcs).