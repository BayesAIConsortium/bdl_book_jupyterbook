(chap:vi_intro)=
# Introduction to variational inference

+++
In probabilistic machine learning, our goal is often not just to find a single “best” set of parameters for a model, but to quantify our uncertainty over all possible parameters. We accomplish this using Bayesian inference.

Let {math}`\Theta \subseteq \mathbb{R}^{P}` denote the parameter space, and let {math}`\theta \in \Theta` be the vector of all parameters in our model (e.g., the weights and biases of a neural network). Let {math}`\mathcal{D} = \{(x_n, y_n)\}_{n=1}^{N}` be our observed training data, consisting of {math}`N` input–target pairs. Bayesian inference updates our prior beliefs about the parameters, {math}`p(\theta)`, after observing the data, yielding the *posterior* distribution {math}`p(\theta \mid \mathcal{D})`.

This update is governed by Bayes’ theorem:

```{math}
:label: eq:vi_intro_bayes_rule

p(\theta \mid \mathcal{D})
    = \frac{p(\mathcal{D} \mid \theta)p(\theta)}{p(\mathcal{D})}.
```

Here,

*   {math}`p(\theta \mid \mathcal{D})` is the **posterior distribution**, summarising our updated beliefs about {math}`\theta` given the data;

*   {math}`p(\mathcal{D} \mid \theta)` is the **likelihood**, viewed as a function of {math}`\theta`: for a given choice of parameters, it quantifies how likely the observed data are under the model;

*   {math}`p(\theta)` is the **prior distribution**, which encodes our beliefs about {math}`\theta` before observing any data; and

*   {math}`p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)p(\theta)\,\mathrm{d}\theta` is the **marginal likelihood**, also called the **evidence**.

The central computational challenge in most modern Bayesian inference problems is evaluating {math}`p(\mathcal{D})`, which requires integrating the joint distribution {math}`p(\mathcal{D},\theta)` over the parameter space {math}`\Theta`. For models such as Bayesian neural networks, this integral is high-dimensional and generally lacks an analytic closed-form solution. Consequently, the posterior density {math}`p(\theta \mid \mathcal{D})` usually cannot be evaluated in normalised form, motivating approximate inference methods. This chapter focuses on variational inference, also known as variational Bayes, a widely used family of scalable approximate inference methods {cite:p}`Jordan1999variational,wainwright2008graphical,blei2017variational`.

(sec:variational_approach)=
## A Variational Approach: Bayesian Inference as Optimisation

+++
The core idea of variational inference (VI) is to reframe Bayesian inference as an optimisation problem. Instead of representing the generally intractable posterior {math}`p(\theta \mid \mathcal{D})` exactly, we choose a tractable distribution {math}`q(\theta;\varparams)` from a variational family {math}`\mathcal{Q}`, where {math}`\varparams` denotes its variational parameters. Concretely, {math}`\mathcal{Q}` could be a family of Gaussian distributions or a family of factorised distributions, as developed in Chapter [](#chap:vi_mean_field_vi). The goal is to find the member {math}`q(\theta;\varparams^\ast)` that best approximates the posterior.

### The Kullback–Leibler Divergence

+++
Following {cite:t}`kullback1951information`, the **Kullback–Leibler (KL) divergence** from {math}`q(\theta;\varparams)` to the posterior {math}`p(\theta \mid \mathcal{D})` is

```{math}
:label: eq:vi_intro_kl_definition

\begin{aligned}
    D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right)
    &= \int q(\theta;\varparams)
       \log\frac{q(\theta;\varparams)}{p(\theta \mid \mathcal{D})}\,\mathrm{d}\theta \\
    &= \mathbb{E}_{q(\theta;\varparams)}\left[\log q(\theta;\varparams)-\log p(\theta \mid \mathcal{D})\right].
\end{aligned}
```

Because the KL divergence is asymmetric, its direction must be stated explicitly. VI seeks variational parameters satisfying

```{math}
:label: eq:vi_intro_kl_objective

\varparams^\ast \in \operatorname*{arg\,min}_{\varparams}
    D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right).
```

(sec:kl_rewrite)=
### Rewriting the KL Objective

+++
Expanding the posterior in Equation [](#eq:vi_intro_kl_definition) using Bayes’ theorem gives

```{math}
:label: eq:vi_intro_kl_expansion

\begin{aligned}
    D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right)
    &= \mathbb{E}_{q(\theta;\varparams)}\left[\log q(\theta;\varparams)-\log p(\mathcal{D},\theta)\right]
       + \log p(\mathcal{D}).
\end{aligned}
```

Although the KL divergence itself cannot generally be evaluated exactly because {math}`\log p(\mathcal{D})` is unavailable, the log-evidence term is constant with respect to {math}`\varparams` and therefore does not affect the optimiser in Equation [](#eq:vi_intro_kl_objective). The remaining expectation does not require evaluating {math}`p(\mathcal{D})` and can be estimated using samples from {math}`q(\theta;\varparams)` whenever the joint density {math}`p(\mathcal{D},\theta)` and variational density {math}`q(\theta;\varparams)` can be evaluated.

(sec:elbo)=
### The Evidence Lower Bound (ELBO)

+++
Rearranging Equation [](#eq:vi_intro_kl_expansion) yields

```{math}
:label: eq:vi_intro_elbo_decomposition

\log p(\mathcal{D})
    = \underbrace{\mathbb{E}_{q(\theta;\varparams)}\left[\log p(\mathcal{D},\theta)-\log q(\theta;\varparams)\right]}_{\mathcal{L}(\varparams)}
    + \underbrace{D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right)}_{\geq 0}.
```

The first term defines the **evidence lower bound (ELBO)** {math}`\mathcal{L}(\varparams)`. Because the KL divergence is non-negative,

```{math}
:label: eq:vi_intro_elbo_bound

\mathcal{L}(\varparams) \leq \log p(\mathcal{D}).
```

The gap between the ELBO and the log evidence is exactly {math}`D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right)`; the closer the variational distribution is to the posterior in this divergence, the smaller the gap. Since {math}`\log p(\mathcal{D})` is constant with respect to {math}`\varparams`, Equation [](#eq:vi_intro_elbo_decomposition) also gives

```{math}
:label: eq:vi_intro_elbo_kl_equivalence

\operatorname*{arg\,max}_{\varparams}\mathcal{L}(\varparams)
    = \operatorname*{arg\,min}_{\varparams}
      D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta \mid \mathcal{D})\right).
```

Thus, VI replaces evaluation of the posterior in normalised form with optimisation of an objective that can be estimated under suitable choices of likelihood, prior, and variational family.

(sec:interpreting_elbo)=
### Interpreting the ELBO

+++
Using {math}`p(\mathcal{D},\theta)=p(\mathcal{D} \mid \theta)p(\theta)`, the ELBO can be written as

```{math}
:label: eq:vi_intro_elbo_interpretation

\begin{aligned}
    \mathcal{L}(\varparams)
    &= \mathbb{E}_{q(\theta;\varparams)}\left[\log p(\mathcal{D} \mid \theta)\right]
       - D_{KL}\left(q(\theta;\varparams)\middle\|p(\theta)\right).
\end{aligned}
```

The alternative form in Equation [](#eq:vi_intro_elbo_interpretation) reveals a trade-off between two terms:

*   **Expected log-likelihood (data fit):** This term rewards variational distributions that place more probability in regions of parameter space where the observed data have high likelihood.

*   **KL penalty (regularisation):** This term penalises departures of {math}`q(\theta;\varparams)` from the prior {math}`p(\theta)` and can therefore be viewed as a complexity penalty.

VI selects the member of {math}`\mathcal{Q}` that balances fit to the observed data against departure from the prior.

(sec:otherlowerbounds)=
### Other Lower Bounds on the Log-Evidence

+++
The ELBO is one lower bound on {math}`\log p(\mathcal{D})`. More generally, strictly positive unbiased estimators of {math}`p(\mathcal{D})` can induce lower bounds through Jensen’s inequality; this optional subsection states the construction underlying importance-weighted and sequential Monte Carlo objectives.

Assume that {math}`q(\theta;\varparams)>0` whenever {math}`p(\mathcal{D},\theta)>0`. Then

```{math}
:label: eq:vi_intro_importance_identity

\mathbb{E}_{q(\theta;\varparams)}\left[\frac{p(\mathcal{D},\theta)}{q(\theta;\varparams)}\right]
    = \int p(\mathcal{D},\theta)\,\mathrm{d}\theta
    = p(\mathcal{D}).
```

For a concave function {math}`\varphi` and random variable {math}`X`, Jensen’s inequality states that {math}`\varphi(\mathbb{E}\left[X\right])\geq\mathbb{E}\left[\varphi(X)\right]`. Taking {math}`\varphi=\log` in Equation [](#eq:vi_intro_importance_identity) gives

```{math}
:label: eq:vi_intro_jensen_elbo

\begin{aligned}
    \mathcal{L}(\varparams)
    &= \mathbb{E}_{q(\theta;\varparams)}\left[\log\frac{p(\mathcal{D},\theta)}{q(\theta;\varparams)}\right] \\
    &\leq \log \mathbb{E}_{q(\theta;\varparams)}\left[\frac{p(\mathcal{D},\theta)}{q(\theta;\varparams)}\right]
     = \log p(\mathcal{D}).
\end{aligned}
```

More generally, let {math}`r(\mathbf{u};\varparams)` denote the sampling density of all random variables {math}`\mathbf{u}` used to construct a strictly positive unbiased estimator {math}`g(\mathcal{D};\mathbf{u},\varparams)` of {math}`p(\mathcal{D})`, and suppose that {math}`\mathbb{E}_{r(\mathbf{u};\varparams)}\left[\lvert\log g(\mathcal{D};\mathbf{u},\varparams)\rvert\right]<\infty`. Then

```{math}
:label: eq:vi_intro_unbiased_estimator

\mathbb{E}_{r(\mathbf{u};\varparams)}\left[g(\mathcal{D};\mathbf{u},\varparams)\right]
    = p(\mathcal{D}).
```

Jensen’s inequality gives the lower bound

```{math}
:label: eq:vi_intro_general_lower_bound

\mathcal{L}_g(\varparams)
    := \mathbb{E}_{r(\mathbf{u};\varparams)}\left[\log g(\mathcal{D};\mathbf{u},\varparams)\right]
    \leq \log p(\mathcal{D}).
```

The standard ELBO is recovered by taking {math}`\mathbf{u}=\theta`, {math}`r(\theta;\varparams)=q(\theta;\varparams)`, and {math}`g(\mathcal{D};\theta,\varparams)=p(\mathcal{D},\theta)/q(\theta;\varparams)`. For example, {cite:t}`burda2016importance` introduce the importance-weighted autoencoder (IWAE) objective, which uses {math}`s` independent samples {math}`\theta_j\sim q(\cdot;\varparams)` and the estimator

```{math}
:label: eq:vi_intro_iwae_estimator

g(\mathcal{D};\mathbf{u},\varparams)
    = \frac{1}{s}\sum_{j=1}^{s}
      \frac{p(\mathcal{D},\theta_j)}{q(\theta_j;\varparams)},
    \qquad \mathbf{u}=\{\theta_j\}_{j=1}^{s}.
```

Related lower bounds can be constructed using sequential Monte Carlo {cite:p}`maddison2017filtering`. Although {math}`\mathcal{L}_g(\varparams)` may not be available in closed form, it can often be optimised using Monte Carlo gradient estimators. However, a tighter bound alone does not guarantee better optimisation or predictive performance, because the signal-to-noise ratio and other properties of the associated gradient estimator also matter {cite:p}`rainforth2018tighter`.

In summary, VI approximates {math}`p(\theta \mid \mathcal{D})` by optimising a distribution from a tractable family, with the ELBO providing an objective that balances data fit against departure from the prior. Chapter [](#chap:vi_mean_field_vi) develops the mean-field construction in detail, and subsequent chapters consider further variational families and optimisation methods.