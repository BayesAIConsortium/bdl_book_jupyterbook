(chap:sampling_methods_sg_mcmc)=
# Stochastic Gradient MCMC

+++
Deep learning models extensively use stochastic gradients (i.e., mini-batches) to reduce the memory requirements of model training {cite:p}`bottou1991stochastic,amari1993backpropagation`. Under suitable assumptions, stochastic gradients computed from individual samples or minibatches can be used in place of full-batch gradients while retaining almost-sure convergence guarantees. {cite:p}`sebbouh2021almost`. In this chapter, we introduce stochastic gradient MCMC (SG-MCMC) methods, in which stochastic gradients make MCMC suitable for modern deep learning models.

+++
## Introduction

Standard MCMC algorithms are asymptotically exact samplers under suitable conditions {cite:p}`hastings1970monte`; Chapter [](#chap:sampling:intro) provides an introduction to MCMC and HMC. For Langevin-based methods, an appropriate decaying step-size schedule yields asymptotic convergence to the target distribution {cite:p}`chen2015convergence`. However, using the entire dataset to compute each update makes full-batch MCMC impractical for deep learning models, which commonly have tens of millions of parameters. Inspired by stochastic gradient descent (SGD), stochastic gradient MCMC (SG-MCMC) uses stochastic gradients to approximate full-batch MCMC updates. Similar ideas have been transferred to mini-batch Metropolis–Hastings (MH) algorithms {cite:p}`korattikara2014austerity,quiroz2019speeding,zhang2020asymptotically`. However, mini-batch MH often relies on strong assumptions, such as bounded log-likelihoods or bounded gradients, which are not guaranteed for deep learning models. SG-MCMC {cite:p}`welling2011bayesian,chen2014stochastic,ding2014bayesian` addresses these practical concerns under fewer assumptions and with substantially lower memory requirements, thereby bridging the gap between MCMC and Bayesian neural networks (BNNs).

+++
## Classic SG-MCMC Algorithms

In this section, we introduce two concrete examples of SG-MCMC algorithms that are widely used for deep learning models.

### Notation

Let {math}`\theta` be the parameters of a deep learning model, and let {math}`\mathcal{S}=\{\theta_s\}_{s=1}^{M}` be a sample set collected during the MCMC process. Due to memory constraints, we can collect only a finite number of samples to characterise the posterior distribution:

```{math}
p(\theta\mid\mathcal{D}) \approx \frac{1}{M}\sum_{s=1}^{M}\delta(\theta-\theta_s),
```

where {math}`\delta(\cdot)` is the Dirac delta function (also known as the unit impulse) {cite:p}`jeffrey1990linear`. At inference time, the sample set is used for Bayesian model averaging (BMA) {cite:p}`chen2025bayesian`, in which the predictive probabilities of all collected samples contribute to the final prediction:

```{math}
\hat{y} = \argmax_{y\in\mathcal{Y}}\mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}\left[p(y\mid x,\theta)\right] \approx \argmax_{y\in\mathcal{Y}}\frac{1}{M}\sum_{s=1}^{M}p(y\mid x,\theta_s).
```

The goal of each SG-MCMC algorithm is to sample from the posterior distribution and collect such a sample set to reconstruct an approximate posterior distribution.

### Stochastic Gradient Langevin Dynamics

Stochastic Gradient Langevin Dynamics (SGLD) {cite:p}`welling2011bayesian` is the first well-known SG-MCMC algorithm. The full-batch energy function is {math}`U(\theta)=-\sum_{(\bm{x}, \bm{y})\in\mathcal{D}}\log p(\bm{y}\mid\bm{x},\theta)-\log p(\theta)`. Its mini-batch estimate is

```{math}
\widetilde{U}(\theta)=-\frac{|\mathcal{D}|}{|\bm{\Xi}|}\sum_{(\bm{x}, \bm{y})\in\bm{\Xi}}\log p(\bm{y}\mid\bm{x},\theta)-\log p(\theta),
```

where {math}`\bm{\Xi}\subset\mathcal{D}` is the sampled batch. The sampling process for SGLD follows the Langevin dynamics:

```{math}
\theta_{t+1} \gets \theta_t-\alpha_t\nabla_{\theta_t}\widetilde{U}(\theta_t)+\sqrt{2\alpha_t}\,\bm{\epsilon}_t,~~~~\bm{\epsilon}_t \sim \mathcal{N}(\bm{0},\bm{I}),
```

where {math}`\alpha_t` is the step size at iteration {math}`t`. The complete SGLD framework is outlined in Algorithm [%s](#sgmcmc_algo:sgld). Compared with standard SGD, SGLD introduces an additional random-noise term {math}`\sqrt{2\alpha_t}\,\bm{\epsilon}_t`. As noted by {cite:t}`welling2011bayesian`, SGLD approaches standard unadjusted Langevin dynamics {cite:p}`roberts1996exponential` when the step size {math}`\alpha_t` is sufficiently small, as the injected noise dominates the stochastic-gradient noise. Under standard regularity conditions, convergence is guaranteed by a decaying step-size schedule {cite:p}`eon1998online,welling2011bayesian,ma2015complete`. Any schedule satisfying i) {math}`\sum_{t=1}^{\infty}\alpha_t=\infty` and ii) {math}`\sum_{t=1}^{\infty}\alpha_t^2<\infty` allows the sampler to explore the target distribution while controlling its asymptotic error {cite:p}`chen2025bayesian`. Notably, mini-batch noise in SGD without externally injected noise can produce behaviour qualitatively similar to SGLD, but converges to a biased approximate posterior {cite:p}`mandt2017stochastic`.

:::{prf:algorithm} Stochastic Gradient Langevin Dynamics (SGLD)
:label: sgmcmc_algo:sgld

- **Inputs:** dataset $\mathcal{D}$, initial sample $\theta_0 \in \Theta$, step-size schedule $\{\alpha_t\}_{t\geq 0}$
- **Output:** collected samples $\mathcal{S} \subset \Theta$
1. $\theta \gets \theta_0$; $\mathcal{S} \gets \emptyset$
1. **For** each iteration:
    1. $\bm{\Xi} \gets$ a mini-batch sampled from $\mathcal{D}$
    1. $\widetilde{U} \gets -\frac{|\mathcal{D}|}{|\bm{\Xi}|}\log p(\bm{\Xi}\mid \theta) - \log p(\theta)$ — *Note:* Compute energy
    1. $\theta \gets \theta - \alpha_t \nabla_{\theta} \widetilde{U} + \sqrt{2\alpha_t}\bm{\epsilon}$ — *Note:* $\bm{\epsilon} \sim \mathcal{N}(\bm{0}, \bm{I})$
    1. $\mathcal{S} \gets \mathcal{S} \cup \{\theta\}$
:::

### Stochastic Gradient Hamiltonian Monte Carlo

Full-batch Hamiltonian Monte Carlo (HMC) has achieved gold-standard performance in Bayesian inference {cite:p}`izmailov2021what`. HMC incorporates a kinetic-energy term characterised by a set of auxiliary “momentum” variables. To adapt HMC to deep learning models and reduce its memory requirements, stochastic gradient HMC (SG-HMC) {cite:p}`chen2014stochastic` eliminates the need for full-batch gradients and removes the MH correction used by full-batch HMC. Specifically, naive SG-HMC replaces the full-batch energy in the Hamiltonian with its mini-batch estimate:

```{math}
\widetilde{H}(\theta,\bm{r})=\widetilde{U}(\theta)+\frac{1}{2}\bm{r}^{\top}\bm{M}^{-1}\bm{r}=-\frac{|\mathcal{D}|}{|\bm{\Xi}|}\sum_{(\bm{x}, \bm{y}) \in\bm{\Xi}}\log p(\bm{y}\mid\bm{x},\theta)-\log p(\theta)+\frac{1}{2}\bm{r}^{\top}\bm{M}^{-1}\bm{r},
```

where {math}`\bm{r}` is the momentum and {math}`\bm{M}` is a positive-definite mass matrix. Despite its intuitive appeal and simplicity, naive SG-HMC does not preserve the target posterior {cite:p}`chen2014stochastic`: stochastic-gradient noise makes the desired joint distribution {math}`\pi(\theta,\bm{r})\propto\exp(-H(\theta,\bm{r}))`, where {math}`H(\theta,\bm{r})=U(\theta)+\frac{1}{2}\bm{r}^{\top}\bm{M}^{-1}\bm{r}`, non-invariant. To mitigate this problem, {cite:t}`chen2014stochastic` introduces a “friction” term {math}`C\bm{M}^{-1}\bm{r}`. If {math}`\widehat{B}` estimates the stochastic-gradient noise coefficient and {math}`C\geq\widehat{B}`, the practical dynamics are

```{math}
\mathrm{d}\theta=\bm{M}^{-1}\bm{r}\,\mathrm{d}t~~~~\text{and}~~~~\mathrm{d}\bm{r}=-\nabla_{\theta}\widetilde{U}(\theta)\,\mathrm{d}t-C\bm{M}^{-1}\bm{r}\,\mathrm{d}t+\sqrt{2(C-\widehat{B})}\,\mathrm{d}\bm{W}_t.
```

Here, {math}`\bm{W}_t` is standard Brownian motion. The friction and injected-noise terms counteract the stochastic-gradient noise. These modified dynamics are commonly known as second-order Langevin dynamics {cite:p}`wang1945theory`. The complete SG-HMC framework is outlined in Algorithm [%s](#sgmcmc_algo:sghmc).

:::{prf:algorithm} Stochastic Gradient Hamiltonian Monte Carlo (SG-HMC)
:label: sgmcmc_algo:sghmc

- **Inputs:** dataset $\mathcal{D}$, initial sample $\theta_0 \in \Theta$, initial momentum $\bm{r}_0$, step-size schedule $\{\alpha_t\}_{t\geq 0}$, inner steps $m$, friction coefficient $C$, noise estimate $\widehat{B}$
- **Output:** collected samples $\mathcal{S} \subset \Theta$
1. $\theta \gets \theta_0$; $\bm{r} \gets \bm{r}_0$; $\mathcal{S} \gets \emptyset$
1. **For** each iteration:
    1. $\bm{r} \sim \mathcal{N}(\bm{0}, \bm{M})$ — *Note:* Optionally resample momentum
    1. $(\theta^{(0)}, \bm{r}^{(0)}) \gets (\theta, \bm{r})$
    1. **For** $i = 1$ to $m$:
        1. $\bm{\Xi} \gets$ a mini-batch sampled from $\mathcal{D}$
        1. $\widetilde{U} \gets -\frac{|\mathcal{D}|}{|\bm{\Xi}|}\log p(\bm{\Xi}\mid \theta^{(i-1)}) - \log p(\theta^{(i-1)})$ — *Note:* Compute energy
        1. $\theta^{(i)} \gets \theta^{(i-1)} + \alpha_t \bm{M}^{-1}\bm{r}^{(i-1)}$
        1. $\bm{r}^{(i)} \gets \bm{r}^{(i-1)} - \alpha_t \nabla_{\theta} \widetilde{U} - \alpha_t C \bm{M}^{-1} \bm{r}^{(i-1)} + \sqrt{2\alpha_t(C-\widehat{B})}\bm{\epsilon}_t$ — *Note:* $\bm{\epsilon}_t \sim \mathcal{N}(\bm{0}, \bm{I})$
    1. $(\theta, \bm{r}) \gets (\theta^{(m)}, \bm{r}^{(m)})$ — *Note:* No Metropolis-Hastings step
    1. $\mathcal{S} \gets \mathcal{S} \cup \{\theta\}$
:::

+++
## Practical Concerns and Recent Improvements

SG-MCMC methods have been shown to be effective on many small-scale deep learning models {cite:p}`welling2011bayesian,chen2014stochastic,ding2014bayesian`. However, their application to large-scale deep learning is hindered by several practical concerns. In this section, we discuss these concerns and selected attempts to address them. Chapter [](#chap:sampling_methods_low_precision_sampling) presents a complementary perspective on scalable, low-precision sampling.

### Sampling from Diverse Local Minima

The loss surface (or energy landscape) of deep learning models is highly non-convex and multimodal {cite:p}`li2018visualizing,zhang2020cyclical`, which makes it difficult for sampling algorithms to move between modes. Moreover, a loss surface typically contains multiple local minima that may all represent good solutions. However, traditional SG-MCMC samplers commonly use a decaying step size and converge to only one local minimum, reducing their ability to explore the entire loss surface.

To mitigate this problem, {cite:t}`zhang2020cyclical` proposed cyclical stochastic gradient MCMC (cSG-MCMC), which uses the cyclical step-size schedule shown in Figure [%s](#sgmcmc_fig:cyclical). The schedule is defined as

```{math}
\alpha_t = \frac{\alpha_0}{2}
\left[\cos\left(\frac{\pi~\text{mod}(t-1,\lceil T/K\rceil)}{\lceil T/K\rceil}\right)+1 \right],
```

where {math}`\alpha_0` is the initial step size, {math}`K` is the number of cycles, and {math}`T` is the total number of iterations. The cyclical step-size schedule enables the sampler to leave the current mode when the step size increases. It can therefore explore and characterise multiple modes while retaining asymptotic convergence to the target distribution.

:::{figure} assets/sg_mcmc/csgmcmc_reproduced.png
:label: sgmcmc_fig:cyclical
:align: center
:width: 80%

Comparison of cyclical and decaying step-size schedules. Adapted from [@zhang2020cyclical] with the permission of Ruqi Zhang.
:::

### Sampling from Wide and Robust Local Minima

Distribution shift between training and test data creates a generalisation challenge for SG-MCMC algorithms {cite:p}`bansak2024learning`. Local minima identified using the training data may not remain local minima under the test data, especially when they are sharp {cite:p}`baldassi2016unreasonable,chaudhari2019entropy`. To mitigate this problem, {cite:t}`lientropy` introduces Entropy-MCMC (EMCMC), which incorporates flatness-aware optimisation methods {cite:p}`chaudhari2019entropy,foret2021sharpnessaware,bisla2022low` into the SG-MCMC framework. Specifically, EMCMC introduces local entropy into the posterior distribution of model parameters:

```{math}
p(\widetilde{\theta}\mid\mathcal{D})=p(\theta,\theta_a\mid\mathcal{D})\propto\exp\left\{-f(\theta)-\frac{1}{2\eta}\|\theta-\theta_a\|^2\right\},
```

where {math}`f(\theta)=U(\theta)` is the negative log-posterior energy, {math}`\eta>0` is a coupling constant, and {math}`\theta_a` is an auxiliary parameter vector that locally explores the neighbourhood of {math}`\theta`. This posterior induces an extended energy whose gradient is

```{math}
:label: eq:grad

\nabla_{\widetilde{\theta}}U(\widetilde{\theta})=\left[
        \begin{array}{c}
        \nabla_{\theta}U(\widetilde{\theta}) \\
        \nabla_{\theta_a}U(\widetilde{\theta})
        \end{array}
    \right]=\left[
        \begin{array}{c}
        \nabla_{\theta}f(\theta)+\frac{1}{\eta}(\theta-\theta_a) \\
        \frac{1}{\eta}(\theta_a-\theta)
        \end{array}
    \right].
```

This form provides a clear interpretation of the EMCMC sampling process: i) {math}`\theta` seeks low-energy regions of the energy landscape, and ii) {math}`\theta_a` acts as a “direction-correction force” that pulls {math}`\theta` away from sharp minima. Figure [%s](#sgmcmc_fig:emcmc) illustrates this process. EMCMC frames flatness-aware sampling as a standard MCMC process on an extended energy function without introducing an additional correction step. It also has convergence guarantees and can converge faster than previous flatness-aware methods {cite:p}`chaudhari2019entropy,dziugaite2018entropy`.

:::{figure} assets/sg_mcmc/emcmc_reproduced.png
:label: sgmcmc_fig:emcmc
:align: center
:width: 70%

The sampling dynamics of Entropy-MCMC, showing how the guiding variable $\theta_a$ pulls $\theta$ toward flat regions in the energy landscape. Adapted from @lientropy with the permission of Bolian Li.
:::