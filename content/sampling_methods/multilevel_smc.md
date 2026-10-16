(chap:smcs)=
# Sequential Monte Carlo samplers

+++
This chapter develops sequential Monte Carlo (SMC) samplers as a population-based approach to inference in Bayesian deep learning.[^footnote-1] Building on the sequential Monte Carlo methods of Chapter [](#chap:sampling:intro), we focus on the two extensions that make them practical for neural network posteriors: multilevel SMC and scalable parallel SMC.

(sec:smcs:motivation)=
## Motivation

+++
Chapter [](#chap:sampling:intro) introduced SMC as a particle filter {cite:p}`doucet2001sequential,doucet2011tutorial`: a latent state evolving in time, tracked by a population of particles over a sequence of growing spaces. This chapter applies the same population-based approximation to a static target—a Bayesian posterior on a fixed parameter space. Without any dynamics to generate them, the intermediate distributions are now constructed by design, and the resulting methods are known as SMC samplers {cite:p}`del2006sequential`.

It is useful to place the neighbouring methods relative to this sampler. Tempering is the mechanism inside the sampler that builds the intermediate distributions, annealing the likelihood from the prior to the posterior {cite:p}`geyer1991markov,gelman1998simulating,neal2001annealed` (data-tempering {cite:p}`chopin2002sequential` is a common alternative). Annealed importance sampling (AIS) is the reduced case in which resampling is omitted {cite:p}`neal2001annealed`, so it is best viewed as a simplified SMC sampler rather than a distinct method. The chapter’s two main contributions are developments of the SMC sampler, along two different axes: multilevel SMC (MLSMC) improves statistical efficiency by combining a hierarchy of coarse-to-fine approximations in a telescoping estimator {cite:p}`beskos2017multilevel,beskos2018multilevel`, while parallel SMC improves scalability by running many samplers concurrently and weighting them by their normalizing-constant estimates, avoiding the heavy communication of island particle methods {cite:p}`verge2015parallel,whiteley2016role,liang2025scalable`.

These methods matter for Bayesian deep learning because neural network posteriors are high-dimensional, multi-modal with many symmetric modes, and costly to evaluate—the regime where importance sampling degenerates and single-chain MCMC mixes slowly. An SMC sampler anneals an entire population towards the posterior, provides an unbiased estimate of the model evidence, and parallelises naturally. The remainder of the chapter develops the two extensions that make this practical at scale: MLSMC with trace-class neural network priors {cite:p}`sell2023trace,chada2025bayesian`, and scalable parallel SMC {cite:p}`liang2025scalable`.

(sec:smcs:intro)=
## Introduction

+++
Monte Carlo based Bayesian algorithms of the type described in Chapter [](#chap:sampling:intro) typically require generating many samples from a target distribution which at best can be evaluated up to a normalizing constant. This can be expensive and cumbersome in high-dimensions, and so methods to improve complexity are paramount. As discussed in Chapter [](#chap:sampling:intro), the best computational complexity achievable by a Monte Carlo algorithm in general is cost {math}`\propto1/`MSE, i.e. achieving a mean square error (MSE) of {math}`\mathcal{O}(\varepsilon^{2})` for {math}`\varepsilon>0` will asymptotically require {math}`\mathcal{O}(\varepsilon^{-2})` samples. This is hence referred to as the *canonical rate*, as it holds for any intractable integral for which we can generate i.i.d. unbiased samples of the integrand.

The sequential Monte Carlo (SMC) sampler {cite:p}`del2006sequential,dai2022invitation,chopin2020introduction` was developed at the turn of the millennium {cite:p}`jarzynski1997equilibrium,berzuini2001resample,gilks2001following,neal2001annealed,chopin2002sequential`, and is well suited to this high-dimensional regime. Forcing the (successive) importance distributions to be close provides a *generic mechanism* to overcome the “curse-of-dimensionality”, i.e. complexity *constant* scaling like {math}`e^P` for dimension {math}`P` {cite:p}`chatterjee2018sample,agapiou2017importance`. Under suitable high-dimensional stability assumptions, SMC samplers can avoid the exponential degradation of direct importance sampling and may achieve polynomial cost scaling in dimension {cite:p}`beskos2014stability`. The SMC sampler also handles bad initialization or multi-modality on par with other population methods {cite:p}`lee2010utility`, and facilitates adaptive tuning of the Markov kernel {cite:p}`buchholz2021adaptive`, with comparable efficiency to MCMC approaches like the No-U-Turn Sampler (NUTS) {cite:p}`hoffman2014no`. These benefits are not shared by the particle filter, which does not utilize MCMC methodology. It also delivers an *unbiased* estimator of the normalizing constant, or model evidence, which can be useful in practice {cite:p}`chopin2020introduction`.

The simplest method for improving complexity is parallelization. Standard i.i.d. Monte Carlo simulation is “embarrassingly parallel”, as the samples can all be simulated *independently* on different machines. If the cost of each simulation dominates the {math}`\mathcal{O}(\varepsilon^{-2})` communication and summation operations, which should both have extremely small scalar constants which are often invisible to the algorithm ({math}`<10^{-9}` smaller than likelihood computations), then the constant and hence final time complexity can be reduced significantly in practice, even down to effectively {math}`\mathcal{O}(1)` when the merge is invisible. Monte Carlo methods for Bayesian simulation do not typically deliver i.i.d. samples, and so they are not a priori parallel. SMC samplers admit a certain degree of parallelism in their raw form, and we will show that they can also be provably “embarrassingly” parallelized beyond this {cite:p}`liang2025scalable`.

If the underlying problem we are simulating from additionally requires a discrete approximation of a function then the complexity compounds. For example, suppose that achieving a bias{math}`^2` of {math}`\mathcal{O}(\varepsilon^2)` demands a cost of {math}`\mathcal{O}(\varepsilon^{-\xi})`. Then the MSE complexity compounds as {math}`\mathcal{O}(\varepsilon^{-2-\xi})`. To overcome this issue, one remedy to reduce the computational cost is through a modified Monte Carlo estimator which is known as multilevel Monte Carlo (MLMC) {cite:p}`heinrich2001multilevel,giles2008multilevel,giles2015multilevel`. MLMC introduces a hierarchy of intermediate convergent discretization levels and constructs an estimator as a telescopic sum of a coarse and cheap estimator with many samples and successive increment estimators with increasing cost and decreasing sample size, ultimately improving the complexity to {math}`\mathcal{O}(\varepsilon^{-(2\wedge \xi)})` instead of the product form, which can translate to huge scaling gains for small target MSE.

MLMC methodology in the Bayesian context was first developed for Markov chain Monte Carlo (MCMC) {cite:p}`hoang2013complexity,dodwell2015hierarchical` and sequential Monte Carlo (SMC) samplers {cite:p}`beskos2017multilevel`. We consider a multilevel sequential Monte Carlo (MLSMC) sampler based on the application of a function-space version of the latter algorithm {cite:p}`beskos2018multilevel` to a strongly convergent class of neural network models known as *trace-class neural networks* (TNN) {cite:p}`sell2023trace,chada2025bayesian`. These are Bayesian feedforward neural networks (BNN) which impose a *width decay* on the weights instead of a standard {math}`L^2` width decay prior. This simple and natural idea eliminates exchangeability and non-identifiability of the weights and delivers strong convergence with a tunable smoothness parameter. The limiting Gaussian measures are defined in function spaces with similar properties to Sobolev spaces {cite:p}`bogachev2007measure`. The resulting method was originally developed in {cite:t}`chada2025bayesian`, where the authors were able to demonstrate the canonical complexity is achievable. We will use function evaluations as the fundamental unit of computational cost, which is also a convenient proxy for wall-clock time which is agnostic to particulars of the implementation or hardware.

The organization of this chapter is as follows. First, we introduce the SMC sampler in [Section {name}](#sec:smcs). In [Section {name}](#sec:mlsmc) we present the MLSMC sampler method for TNN. In [Section {name}](#sec:psmc) we present the parallel SMC method.

(sec:smcs)=
## SMC sampler

+++
Suppose that we aim to simulate from a target distribution

```{math}
\pi(\theta) := p(\theta \mid \mathcal{D}) \propto
p(Y \mid X, \theta)\,p(\theta) \, .
```

One option is to use an MCMC kernel {math}`\mathcal{M}` such that {math}`p \mathcal{M} = p` to simulate {math}`\theta^{(k)} \sim \mathcal{M}(\theta^{(k-1)}, \cdot)`, but this method produces correlated local samples and is prone to getting stuck in a given basin of attraction. If that basin of attraction is in the set of indistinguishable basins that we want then it may not particularly matter in practice. Gradient-based MCMC methods such as HMC can reduce random-walk behaviour, but they can still mix poorly across isolated posterior modes or permutation-symmetric basins.

Population methods such as importance sampling (IS) are able to mitigate this issue, but they suffer from their own difficulties. In particular, in its raw form, IS suffers from a curse of dimensionality in both parameter and data {cite:p}`chatterjee2018sample`, and as such it is not suitable for high dimensional problems and big data. The SMC sampler bypasses this problem by

*   interpolating between an easy-to-sample-from initial distribution {math}`p_0` and the target distribution of interest {math}`p` with several intermediate distributions {math}`p_t` such that {math}`p_T = p`, and

*   interleaving sequential importance sampling (SIS) with resampling and mutation by a sequence of appropriate MCMC kernels {math}`\mathcal{M}_t` such that {math}`p_t \mathcal{M}_t = p_t`.

:::{prf:algorithm} SMC sampler (c = communication here)
:label: alg:smc_main

- **Inputs:** number of particles {math}`M`, sequence of distributions {math}`\pi_t`, MCMC kernels {math}`\mathcal{M}_t`
- **Output:** particle system {math}`\{\theta_t^i\}_{i=1}^M` and normalizing constant estimate {math}`Z_T^{M}`
1. initialize {math}`\theta_0^i \sim \pi_0` for {math}`i=1,\dots,M`; set {math}`Z_0^{M} \gets 1`
1. **For** {math}`t = 1` to {math}`T` (in serial):
    *Note:* (Optional) adapt annealing parameter
    1. select {math}`\lambda_t` such that {math}`{\sf ESS} = \alpha M`
    *Note:* (c) Update estimate of normalizing constant
    1. {math}`Z_t^{M} \gets Z_{t-1}^{M} \cdot \frac{1}{M}\sum_{k=1}^M p(Y \mid \theta_{t-1}^k, X)^{\lambda_t - \lambda_{t-1}}`
    1. **For** {math}`i = 1` to {math}`M` (in parallel):
        1. define {math}`w_t^i \propto p(Y \mid \theta_{t-1}^i, X)^{\lambda_t - \lambda_{t-1}}`
        *Note:* (c) Selection step
        1. sample ancestor index {math}`I_t^i \sim \{w_t^1, \dots, w_t^M\}`
        *Note:* Mutation step
        1. sample {math}`\theta_t^i \sim \mathcal{M}_t(\theta_{t-1}^{I_t^i}, \cdot)`
:::

+++
In the Bayesian context, typically the prior {math}`p` is a suitable initial distribution. The sequence of intermediate targets can be built to gradually introduce the data, for example either from growing subsets of the data {cite:p}`chopin2002sequential` or with a tempering schedule {math}`0=\lambda_0<\lambda_1 < \dots < \lambda_T=1` {cite:p}`neal2001annealed`:

```{math}
p_t(\theta) \propto
p(Y \mid X, \theta)^{\lambda_t}\,p(\theta) \, .
```

Three quantities are tracked along this path. The normalizing constant at temperature {math}`\lambda_t` is

```{math}
Z_t = \int p(Y \mid X, \theta)^{\lambda_t}\,p(\theta)\,d\theta,
```

which at {math}`\lambda_T=1` equals the model evidence {math}`p(Y \mid X)`. [](#alg:smc_main) maintains an unbiased estimate {math}`Z_t^{M}` at each step, with {math}`Z_T^{M}` returned as the estimate of the evidence. Successive targets are linked by the incremental importance weight that reweights particle {math}`i` from {math}`\pi_{t-1}` towards {math}`\pi_t`,

```{math}
w_t^i \propto p(Y \mid X, \theta_{t-1}^i)^{\lambda_t-\lambda_{t-1}},
\qquad
\tilde w_t^i = \frac{w_t^i}{\sum_{k=1}^{M} w_t^k}.
```

How evenly these weights are spread is measured by the effective sample size (ESS),

```{math}
\mathrm{ESS} = \frac{1}{\sum_{i=1}^{M} (\tilde w_t^i)^2} \in [1,M].
```

Values near {math}`M` indicate balanced weights, whereas small values mean a few particles dominate. The ESS drives the algorithm in two ways: a low value motivates resampling, and in the adaptive variant the next temperature is chosen to preserve a target {math}`\mathrm{ESS}=\alpha M`. Data-tempering and likelihood-tempering construction schemes for the intermediate targets can also be mixed and matched. We will consider another option in [Section {name}](#sec:mlsmc). See [](#alg:smc_main) for the version with tempering for Bayesian inference. The steps which require communication during execution are denoted with (c). It is noteworthy that if we skip the resampling step then the algorithm is called annealed importance sampling (AIS) and was introduced in {cite:t}`neal2001annealed`. The normalizing constant then only needs to be computed once at the end. This version of the algorithm is convenient because it is embarrassingly parallel – communication is only required at resampling times. However, the purpose of resampling is to prevent degeneracy and improve stability, and this benefit often outweighs the added variance {cite:p}`chopin2002sequential,chopin2020introduction`.

(sec:mlsmc)=
## MLSMC for TNN

+++
In this section we will build up the MLSMC for TNN method. First we introduce the MLMC method, followed by the TNN. Then we present the MLSMC sampler and provide theoretical justification, and numerical results validating and extending the theory.

### MLMC method

+++
We shall begin with a short review of MLMC. MLMC is useful when accurate simulations are expensive but coarse approximations are cheap. Rather than estimating the finest-level expectation directly, one estimates a coarse expectation plus a sequence of corrections. If consecutive levels are strongly coupled, the correction variances decay rapidly, so fewer samples are needed at expensive fine levels. Let us assume that we are given a probability density {math}`p`, on a state-space {math}`\mathsf{U}`. It is of interest to compute expectations of {math}`p-`integrable functions, {math}`\varphi:\mathsf{U}\rightarrow\mathbb{R}`; {math}`p(\varphi):=\int_{\mathsf{U}}\varphi(u)p(u)du`. Now, we assume that we need to approximate {math}`p` by a density {math}`p_l` on a state-space {math}`\mathsf{U}_l\subseteq\mathsf{U}` such that:

1.  {math}`\lim_{l\rightarrow\infty}p_l(\varphi)=p(\varphi)`, for any integrable {math}`\varphi:\mathsf{U}\rightarrow\mathbb{R}`.

2.  Computing with {math}`p_l` grows progressively more expensive as {math}`l` increases.

Fix a finest level {math}`L\in\{1,2,\dots\}`. The finest-level expectation decomposes into a cheap coarse term plus a sum of level-wise corrections,

```{math}
:label: eq:tele

p_L(\varphi) = p_0(\varphi) + \sum^L_{l=1}[p_l-p_{l-1}](\varphi),
```

where {math}`[p_l-p_{l-1}](\varphi)` abbreviates {math}`p_l(\varphi)-p_{l-1}(\varphi)`. MLMC is built on exactly this decomposition: estimating the right-hand side term by term turns out to be cheaper than attacking the left-hand side directly. The vehicle for this is a *coupling* of each consecutive pair: a joint density {math}`\check{p}_l` on {math}`\mathsf{U}_l\times\mathsf{U}_{l-1}`, for {math}`l\in\{1,\dots,L\}`, whose marginals recover the pair, i.e. {math}`\int_{\mathsf{U}_{l-1}}\check{p}_l(u_l,u_{l-1})du_{l-1}=p_{l}(u_{l})` and {math}`\int_{\mathsf{U}_l}\check{p}_l(u_l,u_{l-1})du_l=p_{l-1}({u}_{l-1})`.

Equipped with these couplings, the estimator is assembled in two steps.

1.  Draw {math}`M_0\in\mathbb{N}` i.i.d. samples {math}`U_0^1,\dots,U_0^{M_0}` from the coarsest density {math}`p_0`.

2.  For each increment level {math}`l\in\{1,\dots,L\}`, and mutually independently of everything else, draw {math}`M_l\in\mathbb{N}` i.i.d. pairs \
    &#x20;{math}`(U_l^1,\tilde{U}_{l-1}^1),\dots,(U_l^{M_l},\tilde{U}_{l-1}^{M_{l}})` from the coupling {math}`\check{p}_l`.

The resulting estimator reads

```{math}
:label: eq:psi_ml

p_L^{ML}(\varphi) :=
\frac{1}{M_0}\sum_{i=1}^{M_0}\varphi(U_0^i) +
\sum_{l=1}^L \frac{1}{M_l}\sum_{i=1}^{M_l}\{\varphi(U_l^i)-\varphi(\tilde{U}_{l-1}^i)\}.
```

Using i.i.d. samples {math}`U_L^1,\dots,U_L^M` from {math}`p_L` to approximate {math}`p_L(\varphi)`, we have

```{math}
:label: eq:psi_iid

p^{IID}_L(\varphi) :=
\frac{1}{M}\sum_{i=1}^{M}\varphi(U_L^i).
```

Neither estimator carries sampling bias, and the mean square error of any unbiased estimator decomposes into its variance plus the squared discretization bias; for [](#eq:psi_ml),

```{math}
:label: eq:MSE

\mathbb{E}[(p_L^{\sf ML}(\varphi)-p(\varphi))^2] =
\underbrace{\operatorname{Var}[p_L^{ML}(\varphi)]}_{\sf variance} +
\underbrace{[p_L-p](\varphi)^2}_{{\sf bias}^2} \, ,
```

where {math}`\operatorname{Var}` denotes the variance operator. The same decomposition holds for [](#eq:psi_iid) with an identical bias term (which an optimized allocation balances against the variance), so any multilevel advantage must enter through the variance. For the multilevel estimator this is

```{math}
:label: eq:varr

\operatorname{Var}[p_L^{\sf ML}(\varphi)] = \frac{\operatorname{Var}[\varphi(U_0^1)]}{M_0}+\sum_{l=1}^L\frac{
\operatorname{Var}[\varphi(U_l^1)-\varphi(U_{l-1}^1)]
}{M_l},
```

while the single-level estimator [](#eq:psi_iid) has

```{math}
\operatorname{Var}[p^{\sf IID}_L(\varphi)] = \frac{\operatorname{Var}[\varphi(U_L^1)]}{M}.
```

Whenever the couplings force the increment variances {math}`\operatorname{Var}[\varphi(U_l^1)-\varphi(U_{l-1}^1)]` to decay rapidly in {math}`l`, few samples are required at the expensive fine levels, and the multilevel estimator [](#eq:psi_ml) attains the same order of MSE (in the sense of [](#eq:MSE)) as [](#eq:psi_iid) at strictly lower cost. The following theorem quantifies when this occurs.

:::{prf:theorem} Giles [@giles2008multilevel]
:label: thm:VMLMC

+++
Assume rate constants {math}`(\alpha,\beta,\gamma) \in \mathbb{R}_+^3`, satisfying {math}`\alpha \geq
\frac{\min(\beta,\gamma)}{2}`, which govern respectively the weak error, the increment variance, and the per-sample cost:

*   {math}`|p_l(\varphi)  - p(\varphi)| =\mathcal{O}(2^{-\alpha l})`.

*   {math}`\operatorname{Var}[\varphi(U_l^1)-\varphi(U_{l-1}^1)]=\mathcal{O}(2^{-\beta l})`.

*   {math}`C_l =\mathcal{O}(2^{\gamma l})`.

Let {math}`\varepsilon < 1` and set {math}`L := \lceil \log(1/\varepsilon) \rceil`. Then the sample allocation {math}`(M_{1},\dots,M_L) \in \mathbb{N}^{L}` can be chosen so that the estimator achieves

```{math}
\mathrm{MSE} =\mathbb{E}[(p_L^{ML}(\varphi)-p(\varphi))^2]=\mathcal{O}(\varepsilon^2),
```

at a total cost obeying

```{math}
:label: eq:mlmcCost

\mathrm{Cost(MLMC)} := \sum_{l=0}^L M_l C_l =
\begin{cases} \mathcal{O}(\varepsilon^{-2}), \quad &\mathrm{if} \ \beta
>\gamma,\\ \mathcal{O}(\varepsilon^{-2}( \log \varepsilon)^2), \quad
&\mathrm{if} \ \beta =\gamma,\\
\mathcal{O}(\varepsilon^{-2-\frac{(\gamma-\beta)}{\alpha}}), \quad
&\mathrm{if} \ \beta <\gamma,
\end{cases}
```

where {math}`C_l` denotes the cost of producing one coupled difference {math}`\varphi(U_l^i)-\varphi({U}_{l-1}^i)`.
:::

+++
The cost [](#eq:mlmcCost) is asymptotically below that of the single-level estimator [](#eq:psi_iid). To see this, note that [](#eq:psi_iid) requires {math}`\mathcal{O}(\varepsilon^{-2})` samples, each at per sample cost {math}`\mathcal{O}(\varepsilon^{-\gamma/\alpha})`, for a total of {math}`\mathcal{O}(\varepsilon^{-2-\gamma/\alpha})`, which exceeds the MLMC complexity whenever {math}`\beta>0`. When {math}`\beta > \gamma` the multilevel cost reaches {math}`\mathcal{O}(\varepsilon^{-2})`, known as the canonical rate: the cost of i.i.d. sampling of a tractable integrand, and hence unimprovable in general.

(sec:tnn)=
### Trace class neural networks

+++
We first fix the setting. The data consist of {math}`N\in\mathbb{N}` pairs {math}`\mathcal{D} = \left((x_1,y_1),\dots,(x_N,y_N)\right)`, with {math}`x_i \in \mathsf{X}` and {math}`y_i\in \mathsf{Y}` for {math}`i\in\{1,\dots,N\}`. The goal is to learn from these a predictive model {math}`f: \mathsf{X} \rightarrow \mathsf{Y}`, and a standard route is a parametric family {math}`f: \mathsf{X}\times\Theta \rightarrow \mathsf{Y}` with {math}`\Theta\subseteq\mathbb{R}^{P}`. The inputs {math}`x_{1:N}` are treated as deterministic, i.e. the discriminative rather than generative view of supervised learning.

**Regression.**  If {math}`\mathsf{Y}=\mathbb{R}^C`, the observations are modelled, for {math}`i\in\{1,\dots,N\}`, as

```{math}
:label: eq:modelreg

y_i = f(x_i,\theta)+ \epsilon_i \, , \qquad
\epsilon_i \stackrel{\textrm{ind}}{\sim} \mathcal{N}_C(0,\Sigma_i),
```

where the noise terms are independent across {math}`i\in\{1,\dots,N\}` (denoted {math}`\textrm{ind}`) and {math}`\mathcal{N}_C(\mu,\Sigma)` is the Gaussian distribution on {math}`\mathbb{R}^C` with mean {math}`\mu` and covariance {math}`\Sigma`. The corresponding likelihood is

```{math}
:label: eq:reglikelihood

p(Y|\theta, X) = \prod_{i=1}^N \phi_C(y_i;f(x_i,\theta),\Sigma_i),
```

with {math}`\phi_C(y;\mu,\Sigma)` the density of {math}`\mathcal{N}_C(\mu,\Sigma)` evaluated at {math}`y`.

**Classification.**  If {math}`\mathsf{Y} = \{1,\dots,C\}` for some {math}`C\in\mathbb{N}`, for convenience of the present exposition we will define {math}`f:\mathsf{X}\times\Theta\rightarrow\mathbb{R}^C`, with {math}`f(x,\theta)=(f_1(x,\theta),\dots,f_C(x,\theta))`, and then separately the *softmax* function as

```{math}
:label: eq:class

S_k(x_i,\theta) :=
\frac{\exp\{f_k(x_i,\theta)\}}{\sum_{j=1}^C \exp\{f_j(x_i,\theta)\}}\, , \qquad k\in\mathsf{Y} \, .
```

The labels are modelled as {math}`y_i \sim S(x_i,\theta)`, independently over {math}`i\in\{1,\dots,N\}`; here {math}`S(x,\theta)=(S_1(x,\theta),\dots,S_C(x,\theta))` is read as a categorical distribution over the {math}`C` classes associated with input {math}`x`. In this case, the likelihood is

```{math}
:label: eq:classlikelihood

p(Y|\theta, X) = \prod_{i=1}^N \prod_{k=1}^C S_{k}(x_i,\theta)^{\mathbb{I}_{[y_i=k]}} \, .
```

The predictive {math}`S` is Lipschitz, so conclusions about {math}`f` translate immediately. We conclude this part by stating the key advantages of TNNs: one is able to have a stable infinite limit related to the width, as well as high-dimensional scalability, due to the fact that one does not need to define the covariance operator structure.

(sec:bnn)=
#### Bayesian neural networks

+++
Define the element-wise activation function as {math}`\sigma`. Let {math}`\mathsf{X} = \mathbb{R}^D` and {math}`\mathsf{Y}=\mathbb{R}^C`. A DNN is specified by layer dimensions {math}`(D_0,\dots,D_L)\in\mathbb{N}^{L+1}`, where necessarily {math}`D_0=D` (input layer) and {math}`D_L=C` (output layer), together with weight matrices {math}`A_d\in \mathbb{R}^{D_{d} \times D_{d-1}}` and bias vectors {math}`b_d\in \mathbb{R}^{D_d}` for {math}`d\in\{1,\dots,L\}`. Collecting the parameters as {math}`\theta := \left((A_1,b_1),\dots,(A_L,b_L)\right)`, so that {math}`\theta\in\Theta=\bigotimes_{d=1}^L\{ \mathbb{R}^{D_d\times D_{d-1}}\times\mathbb{R}^{D_d}\}`, the network output is built by the recursion

```{math}
:label: eq:DNN

\begin{align}\nonumber
g_0(x,\theta) & := & A_1x + b_1,\\ \nonumber
g_{L'}(x,\theta) & := & A_{L'}\sigma
(g_{L'-1}(x)) + b_{L'}\, ,
\qquad L' \in\{1,\dots L-1\}, \\

f(x,\theta) &:=&  A_L\sigma
(g_{L-1}(x)) + b_L,
\end{align}
```

with the final-layer output {math}`f(x,\theta)` defining the DNN. The BNN is given by placing a prior {math}`p` on {math}`\Theta`.

(ssec:tnn)=
#### TNN

+++
We now introduce the trace class neural network (TNN) priors, which were first proposed in {cite:t}`sell2023trace`, and differ from standard BNN priors in two fundamental ways. The first is that the prior on the weights and biases depends on the rows and columns, as opposed to standard isotropic weight decay. This is referred to as *width-decay* and it eliminates exchangeability and the associated non-identifiability. The second thing is that nested approximations of the well-defined TNN function-space converge strongly to a non-Gaussian process in the limit of infinite width. This stands in contrast to standard isotropic priors, under which the appropriately rescaled network converges weakly to a Gaussian process {cite:p}`neal1995bayesian,matthews2018gaussian`, so that the infinite-width limit is a kernel method and the compositional structure of the prior is lost. This will be made precise below.

These priors were introduced to mimic Gaussian measure priors {math}`p \sim \mathcal{N}(0,\mathcal{C})` for inverse problems over function-space, for which Gaussian random fields are commonly simulated through the Karhunen-Loève expansion

```{math}
:label: eq:kle

f_{\sf KL} = \sum_{j \in \mathbb{Z}^+} \sqrt{\lambda_j} \iota_j \Phi_j, \qquad \iota_j \sim \mathcal{N}(0,1),
```

with {math}`(\lambda_j,\Phi_j)_{j \in \mathbb{Z}^+}` the eigenpairs of the covariance operator {math}`\mathcal{C}` and {math}`\{\iota_j\}_{j \in \mathbb{Z}^+}` Gaussian white noise; see {cite:t}`lord2014introduction` for the derivation of [](#eq:kle) and its use in stochastic numerics. Priors built from [](#eq:kle) scale poorly with input dimension, however, and this shortcoming was the original motivation for TNN priors, in which the weight and bias variances are collected in a trace-class diagonal covariance operator {math}`\mathcal{C}`. See {cite:p}`sell2023trace{Theorem 1}`. More precisely, the TNN prior is given by

```{math}
:label: eq:tnn

A_{ij,d} \sim \mathcal{N}(0, (ij)^{-s}),
\quad b_{i,d}^{} \sim \mathcal{N}(0, i^{-s}) \, .
```

The tuning parameter {math}`s` controls how much information one believes concentrates on the first nodes, and as such also controls the smoothness, expressiveness, and flexibility of the prior in terms of the functions it can represent. In the case of {math}`s>1`, we refer to the prior as trace-class, which is the motivation for the name trace-class neural network prior.

A convenient property of these priors is strong convergence, which is made precise with the following proposition. The proof is given in {cite:t}`chada2025bayesian`. This is the key result that enables application of the MLSMC algorithm to TNN. The advantage of TNN is that it allows one not to prespecify the covariance operator structure in the prior, and it is also capable of having an infinite-width limit.

:::{prf:proposition}
:label: prop:tnnconv

+++
Assume that for all {math}`z\in \mathbb{R}`, {math}`|\sigma(z)| \leq |z|`. Then for {math}`s > 1/2`, the prior predictive output is square summable for the prior defined as in [](#eq:tnn). Furthermore, let {math}`x \in \mathsf{X}`, and consider the TNN {math}`f_l(x,\theta_l)` truncated at {math}`D_l=2^l` width, for {math}`l \in \mathbb{N}`, with the limit denoted {math}`f(x,\theta)=\lim_{l \rightarrow \infty}f_l(x,\theta_l)`, as described above. Then there is a {math}`K(x)>0` such that

```{math}
:label: eq:tnnconv

\mathbb{E}\left[| f_l(x, \theta_l)  - f(x,\theta) |^2\right]
\leq K 2^{-(2s-1) l} \, .
```
:::

(sec:mlbnn)=
#### Multilevel TNN

+++
The remaining architectural freedom is the hidden-layer widths {math}`D_{L'}`, {math}`L'\in\{1,\dots,L-1\}`; the input and output widths are dictated by the problem, {math}`D_0=D` and {math}`D_L=C`. We take the depth {math}`L` as fixed and give every hidden layer a common width governed by a resolution parameter {math}`l\in\mathbb{N}`, namely {math}`D_l=2^l`, after which the per-layer width variables {math}`D_{L'}` are no longer needed. Denote the corresponding vector of parameters by {math}`\theta_l := \left((A_{1}^l,b_{1}^l),\dots,(A_{D}^l,b_{D}^l)\right)
\in \Theta_l \subset \Theta`. The limiting infinite width NN output function {math}`f(x,\theta)` [](#eq:DNN) approximated at finite resolution {math}`l` is denoted by {math}`f_l(x,\theta_l)`, the likelihood by {math}`p_l(Y|\theta_l,X)`, and the posterior distribution by

```{math}
:label: eq:bnn_approx

\pi_l(\theta_l)
:= p_l (\theta_l | \mathcal{D}) \propto
p_l(Y|\theta_l,X)p_{l}(\theta_l)
=: \kappa_l(\theta_l) \, .
```

Both {math}`\pi_l` and the network {math}`f_l(x,\theta_l)` should be read as finite-width surrogates for their counterparts under the non-parametric limiting DNN as {math}`l \rightarrow \infty`, whenever that limit is well defined.

(sec:algo_math)=
### MLSMC TNN method

+++
This section presents the algorithm for estimating expectations under the posterior {math}`\pi`, in particular the posterior predictive {math}`\mathbb{E}_{\pi}[f(x,\theta)],` together with the mathematical results that justify it. The guarantees transfer unchanged to any objective {math}`\varphi \circ f` with {math}`\varphi` Lipschitz, a class covering the usual performance metrics, negative log likelihood and accuracy among them. Everything is stated for the posterior predictive first, then extended as a corollary. Splitting off the discretization error via {math}`\mathbb{E}_{\pi}[f(x,\theta)] =
\mathbb{E}_{\pi_L}[f_L(x,\theta_L)] +
(\mathbb{E}_{\pi}[f(x,\theta)]-\mathbb{E}_{\pi_L}[f_L(x,\theta_L)])`, the plan is to bring the MLMC machinery to bear on the telescoping identity

```{math}
:label: eq:ml_id_desc

\mathbb{E}_{\pi_L}[f_L(x,\theta_L)] = \sum_{l=1}^L\left\{\mathbb{E}_{\pi_l}[f_l(x,\theta_l)] - \mathbb{E}_{\pi_{l-1}}[f_{l-1}(x,\theta_{l-1})]\right\}  + \mathbb{E}_{\pi_0}[f_0(x,\theta_0)] \, .
```

It will be shown that estimating the summands on the R.H.S. independently achieves a MSE matching {math}`(\mathbb{E}_{\pi}[f(x,\theta)]-\mathbb{E}_{\pi_L}[f_L(x,\theta_L)])^2` at lower optimal cost than directly approximating {math}`\mathbb{E}_{\pi_L}[f_L(x,\theta_L)]`.

(sec:algorithm)=
#### Algorithm

+++
Our construction is that of {cite:t}`beskos2017multilevel,beskos2018multilevel`, and we adopt notation compatible with those works so that their results can be invoked directly. The input {math}`x\in\mathsf{X}` and the finest level {math}`L\in\mathbb{N}` are held fixed throughout. The parameter spaces are built recursively, {math}`\Theta_l = \Theta_{l-1} \times \Delta_l` for {math}`l\in\{1,\dots,L\}`, with {math}`\Delta_0=\Theta_0`; the role of these spaces will emerge shortly. Write {math}`\theta_0=\delta_{0}\in\Theta_0` and, for {math}`l\in\{1,\dots,L\}`,

```{math}
\theta_l = (\theta_{l-1},\delta_l) = ({\delta}_0,\dots,\delta_{l}) \in\Theta_l.
```

The coordinates {math}`(\delta_1,\dots,\delta_{l})` record the new parameters introduced as the width grows from level {math}`l-1` to level {math}`l`; the algorithm to be presented is organized around these objects. Next introduce proposals: a strictly positive density {math}`q_0(\theta_0)` on {math}`\Theta_0` and, for each level, a strictly positive conditional density {math}`q_l(\cdot|\theta_{l-1})` on {math}`\Delta_l`. The incremental weights are then {math}`G_0(\theta_0) = \kappa_0(\theta_0)/q_0(\theta_0)` together with

```{math}
G_l(\theta_l) = \frac{\kappa_{l}(\theta_l)}{\kappa_{l-1}(\theta_{l-1})q_{l}(\delta_{l}|\theta_{l-1})} \, .
```

For each {math}`l\in\{1,\dots,L-1\}`, take a Markov kernel {math}`\mathcal{K}_l` leaving {math}`\pi_l` invariant, and compose it with the proposal for the next increment, giving

```{math}
:label: eq:kernel

\mathcal{M}_l(\theta_{l},d\theta_{l+1}') = \mathcal{K}_l(\theta_{l},d\theta_{l}')q_{l+1}(\delta_{l+1}|\theta_{l}')d\delta_{l+1},
```

in which {math}`\theta_{l+1}' = (\theta_{l}',\delta_{l+1})` and {math}`d\delta_{l+1}` denotes Lebesgue measure of the matching dimension. Step 4 of [](#alg:mlsmc_dnn) applies exactly this kernel.

Define {math}`\eta_0(\theta_0)=q_0(\theta_0)` and, for each {math}`l\in\{1,\dots,L\}`,

```{math}
\eta_l(\theta_l) = \pi_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1}).
```

The sampler below produces particle approximations of these densities, and thereby of expectations taken under them. In the subsequent exposition, given {math}`(\theta_l^1,\dots,\theta_l^M)\in\Theta_l^M`, the so-called {math}`M-`empirical measure will be denoted {math}`\eta_l^M`. In other words we will have access to {math}`M\in\mathbb{N}` samples {math}`(\theta_l^1,\dots,\theta_l^M)\in\Theta_l^M` so that

```{math}
:label: eq:eta_empirical

\eta_l^{M}(\varphi_l) := \frac{1}{M}\sum_{i=1}^{M}\varphi_l(\theta_l^i) \,
{\rightarrow} \, \eta_l(\varphi_l) \, \quad {\sf a.s.}
```

:::{prf:algorithm} Multilevel Sequential Monte Carlo Sampler for TNN
:label: alg:mlsmc_dnn

- **Inputs:** highest resolution {math}`L \in \mathbb{N}`; number of samples per level {math}`(M_0, \dots, M_{L}) \in \mathbb{N}^{L+1}`, with {math}`+\infty > M_0 \geq M_1 \geq \cdots \geq M_{L} \geq 1`
- **Output:** particles {math}`(\theta_1^1,\dots,\theta_1^{M_1},\dots,\theta_L^1,\dots,\theta_L^{M_{L}})`, from which Equation [](#eq:ml_est) is constructed
*Note:* Initialization
1. **For** {math}`i = 1` to {math}`M_0`:
    1. independently sample {math}`\theta_0^i \sim q_0(\theta_0) d\theta_0`
1. {math}`l \gets 0`
*Note:* Iteration across levels
1. **While** {math}`l < L`:
    1. **For** {math}`i = 1` to {math}`M_{l+1}`:
        1. sample {math}`\theta_{l+1}^i \mid \theta_{l}^1,\dots,\theta_{l}^{M_{l}}` independently using
        $$
\sum_{j=1}^{M_{l}}
        \frac{G_{l}(\theta_{l}^j)}
             {\sum_{s=1}^{M_{l}} G_{l}(\theta_{l}^s)}
        \mathcal{M}_l(\theta_{l}^j, d\theta_{l+1})
        $$
    1. {math}`l \gets l + 1`
:::

+++
Recalling [](#eq:ml_id_desc), we will approximate the increments as follows

```{math}
\frac{\eta_{l}^{M_{l}}(G_{l}f_l)}{\eta_{l}^{M_{l}}(G_{l})}-\eta_{l}^{M_{l}}(f_{l-1}).
```

Note that {math}`\eta_{l}^{M_{l}}(f_{l-1})` will converge in probability (as {math}`M_{l}\rightarrow\infty`) to {math}`\pi_{l-1}(f_{l-1})` {cite:p}`del2004feynman`. Then {math}`\eta_{l}^{M_{l}}(G_{l})` will converge to

```{math}
\begin{align}
\int_{\Theta_l}\pi_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1})\frac{\kappa_{l}(\theta_l)}{\kappa_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1})}d\theta_l  =
\frac{1}{Z_{l-1}}\int_{\Theta_l}\kappa_{l}(\theta_l)d\theta_l
 =  \frac{Z_l}{Z_{l-1}} \, ,
\end{align}
```

and {math}`\eta_{l}^{M_{l}}(G_{l}f_l)` converges to

```{math}
\int_{\Theta_l}\pi_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1})\frac{\kappa_{l}(\theta_l)}{\kappa_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1})}f_l(x,\theta_l)d\theta_l
=
\frac{1}{Z_{l-1}}\int_{\Theta_l}\kappa_{l}(\theta_l)f_l(x,\theta_l)d\theta_l \, .
```

Consequently, one can use the following approximation of {math}`\pi_L(f_L)`:

```{math}
:label: eq:ml_est

\widehat{\pi}_L(f_L) = \sum_{l=1}^L\left\{
\frac{\eta_{l}^{M_{l}}(G_{l}f_l)}{\eta_{l}^{M_{l}}(G_{l})}-\eta_{l}^{M_{l}}(f_{l-1})
\right\} + \frac{\eta_{0}^{M_{0}}(G_{0}f_0)}{\eta_{0}^{M_{0}}(G_{0})}.
```

Ordinarily we choose {math}`q_l` so that for each {math}`\theta_l\in\Theta_l`

```{math}
:label: eq:cond_disc

p_l(\theta_l) = p_{l-1}(\theta_{l-1})q_l(\delta_l|\theta_{l-1}).
```

This means that

```{math}
G_{l}(\theta_{l}) =
\frac{p_l(Y|\theta_l,X)}
{p_{l-1}(Y|\theta_{l-1},X)}.
```

The main result is given in {cite:t}`chada2025bayesian`.

:::{prf:proposition}
:label: prop:main_res

+++
Under suitable assumptions, an analog of [](#thm:VMLMC) applies to MLSMC with rates determined by [](#eq:tnnconv) for costs {math}`C_l \propto 2^{2l}`, i.e. our effective {math}`2\alpha = \beta = 2s-1` and {math}`\gamma=2`. Hence we are able to choose {math}`L, \{M_1,\dots,M_L\}` appropriately to achieve MSE of {math}`\mathcal{O}(\varepsilon^2)` for the canonical complexity {math}`\mathcal{O}(\varepsilon^{-2})`.
:::

(sec:num_tnn)=
### Numerical experiments

#### Regression Problem

+++
The first numerical experiment will be based on a well-specified Bayesian regression problem, as given in [](#eq:modelreg), i.e. the data is generated from the model. We let {math}`\Sigma_i^2 =0.01^2 I`, {math}`\sigma(z) = \tanh(z)`, {math}`D=10`, {math}`D_l=2^l`, and {math}`N=200`. The inputs are simulated as {math}`x_i \sim \mathcal{N}(2,0.5)`. We use a high-resolution parameter {math}`l=9` for the ground truth. We compare SMC and MLSMC samplers, using 100 replications to compute MSE. We report the complexity rate {math}`\xi` such that cost {math}`\propto` MSE{math}`^{-\xi}` for different values of {math}`\beta=2s-1`. Recall that {math}`C_l = \mathcal{O}(2^{2 l})` so one expects to attain the canonical rate of convergence {math}`\xi=1` when {math}`s>1.5 \rightarrow \beta>2`. We conduct our numerical experiment with levels {math}`L \in \{3,4,\ldots,7\}` and smoothness parameters {math}`s\in\{3,1.1\}`, delivering canonical and sub-canonical convergence behaviour. The results are presented in [](#fig:reg_results2), including credible sets around the MSE values, given by the thin blue and red curves. Further details on implementation of the MLSMC sampler in general can be found in {cite:t}`beskos2018multilevel`.

The MLSMC sampler reaches any given accuracy more cheaply than its single-level counterpart. In particular: (i) the error-versus-cost slopes differ, with MLSMC approximately attaining the canonical {math}`1/`MSE rate for {math}`s=3`; (ii) at the smallest MSE the cost gap is roughly a factor of 10 for {math}`s=3`, so the asymptotic advantage is already realized at finite resolution; and (iii) the advantage shrinks in the sub-canonical case {math}`s=1.1`.

:::{figure}
:label: fig:reg_results2

```{image} assets/multilevel_smc/plot1-eps-converted-to.png
:width: 45%
```

```{image} assets/multilevel_smc/plot6-eps-converted-to.png
:width: 45%
```

Regression problem: error vs cost plots for SMC and MLSMC using TNN priors. Left: $s=3$. Right: $s = 1.1$. Credible intervals are provided by thin curves.
:::

(sec:mnist)=
#### Binary MNIST Classification

+++
As a second proof of concept, we consider a simplified binary MNIST {cite:p}`lecun2010mnist` classification task using {math}`D=100` principal components (PCs) of the original {math}`28\times28` image dataset, with {math}`N=400`. The results are similar to the regression case, and are presented in [](#fig:mnist2).

:::{figure}
:label: fig:mnist2

```{image} assets/multilevel_smc/plot1_mnist.png
:width: 45%
```

```{image} assets/multilevel_smc/plot6_mnist.png
:width: 45%
```

MNIST Classification problem: error vs cost plots for SMC and MLSMC, using TNN priors. Left: $s=3$. Right: $s=1.1$. Credible sets are provided in the thin curves.
:::

(sec:psmc)=
## Parallel SMC

+++
For this section, we will assume a fixed and finite network architecture {math}`f` of the standard form, as introduced in [Section {name}](#sec:bnn). This gives rise to a computable target distribution (up to a normalizing constant) {math}`\pi = \kappa/Z`.

A first source of parallelism is internal to a single SMC run: the MCMC mutation of each particle can proceed concurrently {cite:p}`lee2010utility,paige2014asynchronous,syed2024optimised`, and it is in the mutations that the likelihood evaluations, the dominant computational cost, take place. Two hardware caveats apply. Each core must have sufficient memory available, or else the “memory wall” is hit {cite:p}`ivanov2021data`. Resampling forces communication among all particles, a potential bandwidth bottleneck, although on single-instruction-multiple-data (SIMD) hardware such as a single multi-core CPU or GPU this is usually benign. Being simple to implement, and sometimes exhibiting strong parallel scaling {cite:p}`lee2010utility`, this intra-SMC parallelism is the most widely used.

:::{prf:algorithm} SMC$_\parallel$ sampler
:label: alg:psmc

- **Inputs:** number of parallel runs {math}`R`; number of particles {math}`M`
- **Output:** particle systems {math}`\{\theta^{i,r}\}_{i=1}^{M}` and normalizing constants {math}`Z^{M,r}` for {math}`r = 1,\dots,R`
1. **For** {math}`r = 1` to {math}`R`:
    1. run Algorithm [](#alg:smc_main)
    1. output {math}`\{\theta^{i,r}\}_{i=1}^{M}` and {math}`Z^{M,r}`
:::

+++
Beyond this, one can run several SMC samplers at once and combine them, as studied in {cite:t}`verge2015parallel,whiteley2016role`. The schemes recommended there typically make *all samples* communicate, which aids stability but works against scalability once a large model must be spread over many SIMD nodes whose inter-connect is slower, or absent altogether. In the island particle model {cite:p}`verge2015parallel`, the total sample budget {math}`M` is divided into {math}`R \leq R_{\rm max}` SMC islands of {math}`M/R` samples apiece. If the SMCs never interact and are combined by naive (equal-weight) averaging, an asymptotic bias penalty of order {math}`(R/M)^2` appears {cite:p}`crisan2018performance,verge2015parallel`. However, *weighting each SMC appropriately removes this penalty* {cite:p}`whiteley2016role,dai2022invitation`. The scalable parallel sampler that results is denoted SMC{math}`_\parallel`.

[](#alg:psmc) displays the SMC{math}`_\parallel` method, and we define the consistent (in {math}`R`, for finite {math}`M` suitably large) SMC{math}`_\parallel` ratio estimator as

```{math}
:label: eq:est_psmc

\hat{\varphi}_{\text{SMC$_\parallel$}}
        = \sum_{r=1}^{R} \omega_r \pi^{M,r}(\varphi) \, , \quad \omega_r = \frac{Z^{M,r}}{\sum_{s=1}^{R} Z^{M,s}} \, .
```

*This estimator is consistent, which is not the case for a naive unweighted average.* See [](#prop:psmc_converge) and the discussion above.

(sec:pmcmc)=
### Parallel MCMC

+++
MCMC can also exploit parallel resources: one can retain a single sample from each of {math}`M` parallel chains, each run for {math}`b` warm-up epochs, achieving {math}`\text{MSE}=O(e^{-b}+1/M)` {cite:p}`margossian2024nested`, which is indistinguishable from the single-chain result for {math}`b \gg \log M`, at a parallel cost equivalent to SMC when the integrated autocorrelation time matches the total number of mutation epochs. However, unlike SMC, the resulting estimator is not consistent for finite {math}`b`, which can potentially spoil convergence. A systematic empirical comparison of parallel SMC and parallel MCMC for Bayesian deep learning is given in {cite:t}`liang2024comparison`.

### Theoretical Result

+++
The convergence result is made rigorous in the following proposition. Suitable assumptions and proof are provided in {cite:t}`liang2025scalable`.

:::{prf:proposition}
:label: prop:psmc_converge

+++
For suitable {math}`\varphi, m, M, T`, there exists a {math}`K         >0`, which depends on {math}`\varphi,m,T`, such that for any {math}`R \in \mathbb{N}`,

```{math}
:label: eq:mse_PSMC

\mathbb{E}[(\hat{\varphi}_{\text{SMC$_\parallel$}} - \pi(\varphi))^2] \leq \frac{K}
        {M R} \, .
```
:::

### Numerical experiments

+++
As above, we will measure wall-clock time complexity in terms of function evaluations, as an implementation and hardware agnostic proxy. We will refer to a likelihood evaluation as an *epoch*, in analogy with stochastic gradient descent (SGD) {cite:p}`robbins1951stochastic,bottou2010large` based optimization methods like Adam {cite:p}`kingma2014adam`. However, note that the time to compute an epoch of mini-batches depends heavily on the size of the problem and the details of the hardware and implementation. If all data fits on a single computational unit (either a node or a core), then a serial epoch of mini-batch likelihood computations will typically require much longer wall-clock time in comparison to a single likelihood evaluation. In contrast, if only a single mini-batch fits on a node, and interconnect is slow, then even data parallel may be slower than serial mini-batches. It is also worth noting that mini-batch gradients can deliver comparable gain to full gradient steps far from convergence, which can significantly accelerate progress as measured by epochs {cite:p}`bottou2018optimization`.

#### Proof of concept

+++
First, we will consider logistic regression on the Australian Credit data {cite:p}`statlog_australian_credit_approval_143`, illustrating comparable results to state-of-the-art (SOTA) parallel MC algorithms and the {math}`1/R` convergence guaranteed by [](#prop:psmc_converge).

:::{figure} assets/multilevel_smc/comparison_pSMC_parallelMCMC_AIS.png
:label: fig:parallelMCMC_AIS
:align: center
:width: 80%

Empirical variance of posterior mean estimates for Bayesian logistic regression on Australian Credit ($m=690$, $d=15$), for SMC$_\parallel$-pCN, various MCMC$_\parallel$ methods, and AIS. SMC$_\parallel$-pCN and the MCMC$_\parallel$ baselines are compared at matched total sample size $M R$, with $M=511$ particles per process; AIS is equivalent to $M=1$ particle per process. SMC$_\parallel$-pCN uses pCN mutations with $30$ mutation steps per tempering stage and adaptive tempering that keeps the effective sample size above $M/2$. Curves are averaged over $25$ repetitions; vertical bars show $\pm 3$ standard errors. Both axes are log-scaled. Some MCMC$_\parallel$ curves are reproduced directly from @schwedes2021rao, with the authors' permission.
:::

+++
The dataset has {math}`d=15` covariates and {math}`m=690` data and Bayesian logistic regression is used. We compare SMC{math}`_\parallel`-pCN (preconditioned Crank–Nicolson, pCN) with annealed importance sampling {cite:p}`neal2001annealed`, various MCMC{math}`_\parallel` methods, and recent synchronous-parallel MCMC {cite:p}`schwedes2021rao`. [](#fig:parallelMCMC_AIS) reports how the empirical variance of the posterior mean estimates decays with the number of samples and of parallel processes; the decay matches the {math}`1/R` rate of [](#prop:psmc_converge). Among the baselines in the figure that avoid communication scaling with {math}`R`, SMC{math}`_\parallel`-pCN achieves the lowest empirical variance. SMC and AIS incur a per-sample overhead in cost ({math}`\sum_t m_t`, where {math}`m_t` are the mutations at tempering step {math}`t`), but this is offset by intra-parallelism ({math}`\sum_t m_t/M \ll 1`).

#### IMDb dataset

+++
We now consider a large problem in semantic classification of natural language, using the IMDb dataset {cite:p}`imdb` of {math}`50{,}000` movie reviews, split evenly into training and test sets.

**Architecture.**  Each review is mapped to a fixed feature vector by SBERT embeddings {cite:p}`reimers-2019-sentence-bert` from the model `all-mpnet-base-v2` {cite:p}`song2020mpnet`[^footnote-2], whose weights are *frozen* throughout: they are used only to produce the 768 dimensional \[CLS] embedding and are never sampled. On top of these embeddings we place (i) one hidden layer with 128 neurons, (ii) ReLU activation, (iii) a final linear layer, and (iv) a softmax output. Inference is over *all* weights and biases of this network, of dimension {math}`P = 98690`, and not merely the final classifier head. The whole train (25000 data) and test dataset (25000 data) are considered.

**Sampling setup.**  The likelihood is the softmax (categorical cross-entropy) likelihood over the whole training set of 25000 reviews, and inference uses an adaptive likelihood-tempering SMC sampler with HMC mutations of length {math}`T`. We compare SMC{math}`_\parallel` against two baselines that share the same prior, likelihood, and frozen embeddings: HMC{math}`_\parallel`, which averages over {math}`M R` independent HMC chains of length {math}`T`, and a single serial HMC run of length {math}`M R T`; in every method the HMC step size is adapted to a common acceptance-rate target, so the comparison is made at matched cost.

:::{figure}
:label: fig:punchline-s1

```{image} assets/multilevel_smc/imdb_tirr.png
:width: 45%
```

```{image} assets/multilevel_smc/imdb_ffc.png
:width: 45%
```

IMDb sentiment classification, with the full posterior over all $P=98690$ network weights, using SMC$_\parallel$ with $M=32$ particles for each of $R$ SMC runs, HMC mutations, and adaptive tempering that keeps the effective sample size above $M/2$. Left: test accuracy and negative log-likelihood (NLL) against the number of parallel runs $R$ at a budget of about $\tau_{\sf irr}=10^4$ epochs, alongside a serial HMC baseline given $M R\tau_{\sf irr}$ epochs and HMC$_\parallel$ with $M R$ chains with $\tau_{\sf irr}$ epochs each. Right: converged values ($R=8$) showing the catastrophic failure (collapsed accuracy and inflated NLL) once the budget drops below $\tau_{\sf irr}$. The $R$- and budget-axes are log-scaled; all curves are averaged over $5$ realizations.
:::

+++
**Experiment.**  Denote by {math}`m` the number of evaluations of {math}`\kappa` per mutation step, which we refer to as *epochs*, and by {math}`T` the number of tempering-and-mutation steps. The product {math}`mT` (or {math}`\sum_{t=1}^T m_t` for an adaptive schedule) is the total simulation cost, which we call the *irreducible serial time* {math}`\tau_{\sf irr}`: the serial simulation the sampler needs in order to mix, which cannot be removed by adding more parallel runs. In practice {math}`\tau_{\sf irr}` is read off empirically as the budget at which accuracy and NLL plateau, here {math}`\tau_{\sf irr}\approx 10^4` epochs. When {math}`m` and/or {math}`T` are too small, so that the budget falls well short of {math}`\tau_{\sf irr}`, the sampler has not mixed and the method *catastrophically fails*: test accuracy collapses and NLL inflates. [](#fig:punchline-s1) summarises this behaviour. The left panel shows performance as a function of {math}`R` at {math}`\tau_{\sf irr}`, where SMC{math}`_\parallel` improves steadily with {math}`R` and matches the far more expensive serial HMC baseline; the right panel fixes {math}`R=8` and reduces the budget below {math}`\tau_{\sf irr}`, exposing the catastrophic failure of SMC{math}`_\parallel` and HMC{math}`_\parallel` once the budget falls short of the irreducible time.

[^footnote-1]: Parts of this chapter are adapted from @chada2025bayesian, @liang2025scalable, and @liang2024comparison.

[^footnote-2]: https://huggingface.co/sentence-transformers/all-mpnet-base-v2
