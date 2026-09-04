(chap:sbi:intro)=
# Simulation-based inference

+++
Simulation-based inference (SBI) {cite:p}`sisson_overview_2018,cranmer2020frontier,deistler2025simulation` has the same goal as Bayesian inference: Given observed data and a set of beliefs, apply Bayes’ rule to update those beliefs accordingly. Unlike other Bayesian inference methods such as Markov chain Monte Carlo (MCMC) or variational inference, SBI does not rely on evaluations of the likelihood, but only on samples thereof. This makes SBI applicable to any kind of black-box function that takes in parameters and simulates observational data, a so-called forward simulator.

As an illustrative example, imagine that you are an astronomer interested in the study of planets that orbit distant stars, i.e., exoplanets. Exoplanets are usually obscured by the bright light emanating from their host stars. Astronomers observe the brightness and spectrum of these stars in a time series called a light curve, and aim to infer from these light curves properties of the exoplanets using physical models. Suppose we want to estimate the mass of an exoplanet {math}`\theta_o` from the light curve observations {math}`x_o` of a distant star. Using relevant physical theory, we design a computer program that maps a hypothetical mass {math}`\theta` to a simulated light curve {math}`x` (the “forward model”). By running such a simulator, we can obtain samples {math}`x` from the likelihood {math}`p(x \mid \theta)`, but evaluating the likelihood can be computationally expensive or impossible (e.g., if the simulator takes a long time to run, or if it contains many latent variables). *Simulation-based inference* computes the posterior distribution {math}`p(\theta \mid x_o)` while only relying on samples (or simulations) from the likelihood. To achieve this, recent SBI methods use neural networks to parameterise an “inference model” {math}`q_{\psi}(\theta \mid x_o)`, where {math}`\psi` denotes the neural-network weights.

+++
## Introduction

:::{figure} assets/sbi/fig1.png
:label: fig:overview
:align: center

*a:* Simulators generate synthetic data $x$ (right) given parameters $\theta$ (left). Bayesian inference computes the posterior distribution over parameters (left) given observations $x_o$ (right). *b:* Simulation-based inference uses simulations (i.e., samples from the likelihood, but no likelihood evaluations) to perform Bayesian inference. A recent SBI method, Neural Posterior Estimation (NPE) uses neural networks to improve inference. NPE draws samples from the prior and runs the simulator to generate pairs of ($\theta, x$). These data are then used to train an inference network: A conditional generative model that estimates the posterior distribution given any data $x$. After training, the inference network is evaluated at any observation in order to draw samples from the posterior, without requiring further simulations or training. Figure modified from @deistler2025simulation.
:::

Classical approaches for Bayesian inference (e.g., MCMC; Chapter [](#chap:sampling:intro), or variational inference; Chapter [](#chap:vi_intro)) rely on repeated evaluations of the likelihood and prior in order to characterise the posterior distribution and to generate samples from it. While powerful, these approaches face limitations for some forward models: Many models in science and engineering are defined as stochastic simulators, which can efficiently be run in the forward direction (i.e., their likelihood can be *sampled* from), but whose likelihood (as well as its gradient with respect to parameters) cannot be evaluated, or only with great computational cost {cite:p}`sisson_overview_2018,papamakarios2019neural,luckmann2021simulation`. For example, the simulator might sample a long series of latent variables {math}`z_{1:N}` before generating observations. Evaluating the likelihood of an observation {math}`p(x_o \mid \theta)` would require integrating out these latent variables via

```{math}
p(x_o \mid \theta) = \int p(x_o, z_{1:N} \mid \theta) \, \mathrm{d}z_{1:N},
```

which can be computationally prohibitive even for a moderate number of latent variables. In addition, many classical Bayesian inference methods are expensive at test-time: Every time we want to perform inference on new data, we have to re-run the inference procedure, which might involve computationally expensive iterations (e.g., in MCMC sampling). As a result, it has been challenging to tackle high-throughput or real-time applications with Bayesian inference.

*Simulation-based inference* (SBI) methods overcome these limitations. The basic goal of SBI is to perform inference given only access to simulations (i.e., samples from the likelihood), without having to explicitly evaluate the likelihood, or any access to the internal workings of the forward model or its gradients. In addition, many SBI methods also *amortize* the cost of Bayesian inference: After an initial phase of simulation and training, they can perform inference for any new dataset within milliseconds, typically requiring only a single forward pass through a deep neural network at test time.

Early SBI methods were developed for cases in which the likelihood of the model cannot be evaluated, and are often referred to as *Approximate Bayesian Computation* methods {cite:p}`beaumont2002approximate,marin2012approximate,sisson_overview_2018`. Wasserstein ABC provides an example of comparing observed and simulated empirical distributions directly, avoiding the need to choose summary statistics {cite:p}`bernton2019approximate`. More recently, the ability of SBI methods to perform inference on any black-box model, to massively parallelise model runs, and to amortize inference has expanded their scope. For example, even when the likelihood or model gradients can—in principle—be evaluated, this might require interfacing with code describing the forward model written across decades in outdated programming languages. In contrast, SBI methods entirely decouple the inference process from simulation runs, which makes it easy to apply SBI methods to simulators that are written in other programming languages or that have to be run on dedicated hardware (or, in principle, which might even correspond to a physical or living system generating the simulations from input). In other scenarios, the likelihood *can* be evaluated, but one needs fast inference for real-time or high-throughput analysis of observed phenomena, which is often out of reach for, e.g., MCMC sampling, but is possible with amortized SBI methods {cite:p}`dax2021real`. Finally, in some cases, large-scale training can also yield highly accurate and robust inference results {cite:p}`lueckmann2021benchmarking,dax2021real,deistler2022energy`. Likely due to its simplicity, the core idea of SBI—to use forward simulations to train a supervised machine learning model—has been used in modified ways across many domains, for example, for tabular foundation models {cite:p}`hollmann2025accurate`.

The core insight of SBI is that performing Bayesian inference can be seen as a form of performing *conditional density estimation*. In recent years, a flurry of powerful conditional density estimation methods—such as normalizing flows—have been developed, often under the moniker of “Generative AI”. This has enabled an explosion of powerful new SBI methods which build on these advances, and which have been successfully applied in a wide range of scientific disciplines, e.g., in neuroscience {cite:p}`gonccalves2020training,groschner2022biophysical,deistler2022energy,confavreux2023meta,rossler2023skewed,hashemi2023amortized`, cognitive science {cite:p}`boelts2022flexible,von2022mental`, biology {cite:p}`korfmann2023deep,avecilla2022neural`, and physics {cite:p}`mishra2022neural,dax2021real,dax2025real,hahn2022accelerated,lemos2024field,barrado202315nh3,Gebhard_2025_Flow,cole2022fast,bhardwaj2023sequential`. Here, we introduce the basic ideas and concepts of SBI. We focus on *Neural Posterior Estimation*, a conceptually simple yet powerful algorithm, but also review other approaches, diagnostic tools, and current developments. Two notable methods are *Neural Likelihood Estimation* {cite:p}`papamakarios2019sequential` and *Neural Ratio Estimation* {cite:p}`thomas2022likelihood,hermans2020likelihood,durkan2020contrastive,miller2022contrastive`, which we explain later in the chapter.

+++
## Simulation-based inference with Neural Posterior Estimation

### Neural Posterior Estimation

The goal of Bayesian inference is to calculate—at least approximately—the posterior distribution of the form {math}`p(\theta \mid x)`. Via Bayes’ rule, the posterior is proportional to the product of the prior and the likelihood {math}`p(\theta \mid x)= \frac{1}{Z(x)} p(x \mid \theta) p(\theta)`, where {math}`Z(x) = \int p(x, \theta) \, \mathrm{d}\theta = p(x)` is the normalizing constant, also called the marginal likelihood or evidence. Calculating {math}`Z(x)` is often challenging, and many classical Bayesian inference algorithms such as MCMC sampling therefore aim to obtain samples from the posterior without having to evaluate it. However, even these algorithms require (typically many) evaluations of the likelihood {math}`p(x \mid \theta)`. However, many models in scientific applications are defined through numerical simulators: One can easily generate *samples* from the likelihood {math}`p(x \mid \theta)` by running the simulator, but *evaluating* the likelihood can be expensive or even impossible. How can one perform Bayesian inference in such setting?

A key insight is that one does not need Bayes rule for Bayesian inference: The posterior distribution {math}`p(\theta \mid x_o)` at some particular {math}`x=x_o` is, by definition, a *conditional distribution*, i.e., a normalised slice through the joint distribution {math}`p(\theta, x)` at {math}`x=x_o` (Figure [](#fig:SBI_as_supervised_learning), left). Thus, an alternative way to get the posterior is to generate a simulated dataset with—possibly many—paired samples {math}`D=\{(\theta_i,x_i)\}_{i=1}^N`, and then to directly estimate this posterior distribution via conditional density estimation. We can generate such a simulated dataset by first sampling the {math}`\theta_i` from the prior, and then, for each {math}`\theta_i`, to run the simulator once to get the associated {math}`x_i`, as {math}`p(\theta, x)=p(\theta)p(x \mid \theta)`.

We want to learn a mapping from {math}`x` to the distribution {math}`p(\theta \mid x)`. One way to tackle this is to parameterise the posterior distribution by a functional form, for example, a Gaussian with mean {math}`\mu(x)` and covariance {math}`\Sigma(x)`, and to use a neural network (with parameters {math}`\psi`) to capture the mapping from {math}`x` to associated posterior parameters {math}`(\mu_{\psi}(x),\Sigma_{\psi}(x))`. One can learn {math}`\psi` by maximising the log-likelihood of the parameters, i.e., by minimising the loss

```{math}
\begin{align}
\mathcal{L}(\psi)= - \frac{1}{N}\sum_{i=1}^N \log q_{\psi}(\theta_i \mid x_i),
\end{align}
```

where in this case, the predicted posterior probabilities {math}`q_{\psi}(\theta \mid x)` would be given by a Gaussian, {math}`q_\psi(\theta \mid x)= \mathcal{N}(\theta \mid \mu_\psi(x), \Sigma_\psi(x))` {cite:p}`BlumFrancois_10,papamakarios2016fast`. Thus—and provided that one can generate a large set of simulations {math}`D`—we have turned Bayesian inference into a supervised learning problem.

:::{figure} assets/sbi/fig4.png
:label: fig:SBI_as_supervised_learning
:align: center

Framing Bayesian inference as a supervised learning problem. Left: A database of simulations ($\theta, x$), sampled from the joint distribution $p(\theta, x)$ by drawing samples from the prior and running the simulator. The posterior distribution (left, bottom) is a horizontal cut through the joint distribution. Right: NPE trains an inference network on these simulations. After training, the network can be evaluated at any observation $x_o$ and directly predicts the posterior distribution. Figure modified from @deistler2025simulation.
:::

Clearly, a Gaussian might be a poor approximation to the posterior, but this approach can readily be generalised to more flexible models. For example, one can use a Mixture of Gaussians {cite:p}`Bishop_94,papamakarios2016fast`, which requires the network to learn the mapping to multiple means {math}`\mu_j(x)`, covariances {math}`\Sigma_j(x)` and associated mixture weights {math}`\pi_j(x)`. In recent years, more flexible and easy-to-train neural conditional density estimators have been proposed, which can be used for estimating non-Gaussian posteriors. In particular, *Normalizing Flows* {cite:p}`papamakarios2021normalizing` constitute a class of neural conditional densities estimators which have many favourable properties for SBI, as we will describe in the next section.

Once the conditional density estimator (sometimes referred to as *inference network*) has been trained on simulated data {math}`D`, it can readily be *evaluated* on any new observation {math}`x_o` to approximate the posterior {math}`p(\theta \mid x_o) \approx q_{\psi}(\theta \mid x_o)`. Indeed, it can be shown that—provided that the density estimator is sufficiently flexible, that the model is well specified, and the optimisation identifies the minimum—it will yield the true posterior {cite:p}`papamakarios2016fast`. To demonstrate this, we rewrite the loss function (in the limit of infinite training data, i.e., in expectation of samples from {math}`p(\theta, x) = p(\theta)p(x \mid \theta)`) as

```{math}
\begin{split}
    \mathcal{L} &= \mathbb{E}_{p(\theta)p(x \mid \theta)}[-\log q_{\psi}(\theta \mid x)] \\
    &= \iint -p(\theta)p(x \mid \theta) \log q_{\psi}(\theta \mid x) \; \mathrm{d}\theta  \; \mathrm{d}x \\
    &= \iint -p(\theta) \frac{p(\theta \mid x)p(x)}{p(\theta)} \log q_{\psi}(\theta \mid x) \; \mathrm{d}\theta  \; \mathrm{d}x \\
    &= \int p(x) \int -p(\theta \mid x) \log q_{\psi}(\theta \mid x) \; \mathrm{d}\theta  \; \mathrm{d}x. \\
\end{split}
```

The inner integral differs from the Kullback–Leibler divergence {math}`D_{KL}\left(p(\theta \mid x)\middle\|q_{\psi}(\theta \mid x)\right)` only by the conditional entropy of {math}`p(\theta \mid x)`, which does not depend on the neural-network parameters {math}`\psi`. Therefore, if the conditional density estimator is well specified, the population loss is minimised when {math}`q_{\psi}(\theta \mid x) = p(\theta \mid x)` for {math}`p(x)`-almost every {math}`x`.

One big advantage of these approaches is that—once the inference network has been trained—the posterior distribution for new observations {math}`x_o` can be evaluated by just plugging it into the inference network to get {math}`q_{\psi}(\theta \mid x)`. Thus, inference can be performed with a single forward pass through a neural network, i.e., it is *amortized* (Figure [](#fig:SBI_as_supervised_learning), right). Conversely, a drawback of this approach is that our simulations need to cover the entire prior, which might be wasteful as we might generate many simulations which look very different to an observation {math}`x_o` we are interested in, and which therefore do not help in constraining the relevant posterior. To address this issue (but which requires giving up on amortization), so-called *sequential* approaches have been developed, in which simulations are generated adaptively to target a posterior for a specific observation. We discuss these approaches in Section [Amortized and sequential inference](#sec:sequential).

The approach described above is now commonly known as *Neural Posterior Estimation* (NPE). Given its conceptual simplicity and the many variations that it allows, it is challenging to track down its precise origins, and variants of it have likely been discovered independently in different application domains. {cite:t}`beaumont2002approximate` and {cite:t}`BlumFrancois_10` introduced the idea of fitting a regression model from data to parameters to refine rejection-ABC approaches, which can be interpreted as NPE with a Gaussian inference network. {cite:t}`papamakarios2016fast` showed that flexible density estimators can directly learn the posterior, when optimised with log-loss minimisation (also see {cite:t}`le2017inference`), and introduced a sequential version. {cite:t}`lueckmann2017flexible` (which, to our knowledge, introduced the term (S)NPE) introduced an alternative sequential formulation, and also embedding networks for learning summary statistics from time series data. {cite:t}`greenberg2019automatic` provided the first method that could use normalizing flows and is compatible with sequential learning, and used convolutional embedding nets. {cite:t}`ardizzone2018analyzing` proposed using invertible neural networks for solving inverse problems, resulting in an algorithm very similar to flow-based NPE, and {cite:t}`radev2020bayesflow` popularised the amortization property of NPE.

### Conditional density estimation with normalizing flows

:::{figure} assets/sbi/fig6.png
:label: fig:flows
:align: center

Normalizing flows. Left: A target distribution which we aim to estimate based on samples. Right: Normalizing flows define a base distribution (e.g., a multivariate Gaussian) and transform this distribution through multiple transformations $T_i$. After training, the transformed distribution $q_{\psi}(\theta)$ approximates the target distribution.
:::

As described above, a core task in SBI is to estimate the conditional distribution of parameters given some observations. Normalizing flows {cite:p}`papamakarios2021normalizing` have emerged as a popular conditional density estimator for SBI. They are simple and efficient to train, can be used to quickly generate samples, or to evaluate exact log-probabilities of the learned density. The latter can be useful for many downstream analyses such as maximum-a-posteriori estimation or visualisation of the posterior distribution. Finally, normalizing flows are (typically) trained by minimising the negative log-likelihood as loss function. This loss function is mass-covering, which, as we will discuss below, can be beneficial for SBI.

Normalizing flows model a target density {math}`q(\theta)` by learning a transformation {math}`T: Z \rightarrow \Theta` such that a random variable {math}`Z`, which is defined to follow a simple “base” distribution {math}`p(z)` (typically a multivariate standard Normal distribution) is transformed into a random variable {math}`\Theta`, such that the probability density of {math}`\Theta` is {math}`q(\theta)`. The probability density modelled by such a transformation can be computed with the change of variables formula

```{math}
:label: sbi:eq:change_of_variables

q(\theta) = \Big| \text{det}\Big(\frac{\partial T^{-1}}{\partial\theta}\Big) \Big|  p_Z(T^{-1}(\theta)),
```

where {math}`T^{-1}` is the inverse of the transformation {math}`T` and {math}`\text{det}\big(\frac{\partial T^{-1}}{\partial\theta}\big)` is the determinant of the Jacobian of the inverse transformation {math}`T^{-1}`. Normalizing flows stack many transformations {math}`T` in order to model complicated densities.

To evaluate Equation [](#sbi:eq:change_of_variables) and to efficiently train normalizing flows, the transformation {math}`T` must be a diffeomorphism (i.e., it must be differentiable and invertible). In addition, the Jacobian of {math}`T^{-1}` is of size {math}`N \times N` (for densities of dimensionality {math}`N`), and computing the determinant of such a matrix is typically of computational complexity {math}`\mathcal{O}(N^3)`, which would be prohibitive for large {math}`N`. As such, normalizing flows require that the transformation is constructed such that the determinant of the Jacobian can be evaluated efficiently. Normalizing flows differ in how they define the transformation {math}`T` such that it fulfills these criteria. A popular class of transformations are autoregressive flows, and within SBI, masked autoregressive flows {cite:p}`papamakarios2017masked` and neural spline flows {cite:p}`durkan2019neural` are particularly popular. These normalizing flows define bijective transformations for every individual dimension and then couple dimensions with neural networks.

:::{figure} assets/sbi/fig2.png
:label: fig:density_estimators
:align: center

Expressiveness of different generative models for approximating the posterior distribution with NPE. The ground truth posterior (left) consists of two moons and has low posterior density in between. Using a Gaussian (second from left) as generative model produces a poor—albeit mass-covering—posterior distribution. A mixture of multiple Gaussians (second from right) improves accuracy, but fails to produce the moon shapes. Normalizing flows (right) enable NPE to accurately capture complex posterior distributions.
:::

Normalizing flows train the transformations {math}`T` by minimising the negative log-likelihood of the data

```{math}
\psi = \operatorname*{arg\,min}_{\psi} - \log \prod_{i=1}^{M} q_{\psi}(\theta_i).
```

In neural posterior estimation (NPE), normalizing flows are used to estimate the posterior probability density {math}`p(\theta \mid x)`. To estimate a *conditional* probability density, NPE conditions that transformation {math}`T` on data {math}`x`, or an embedding {math}`s(x)` thereof (Section [Learning from high-dimensional data with embedding networks](#sec:sbi:embedding_nets)). This enables NPE to learn different posterior distributions for different data.

The normalizing flow is then trained over pairs of parameters and simulation outputs, where the parameters are sampled from the prior. NPE minimises the loss

```{math}
:label: eq:npe_loss

\mathcal{L} = \mathbb{E}_{p(\theta)p(x \mid \theta)}[-\log q_{\psi}(\theta \mid x)].
```

If the normalizing flow is well-specified, i.e., the true conditional posterior {math}`p(\theta \mid x)` belongs to the family of normalizing-flow models, then there exists some {math}`\psi^\ast \in \Psi` such that {math}`q_{\psi^\ast}(\theta \mid x) = p(\theta \mid x)` for {math}`p(x)`-almost every {math}`x`, where {math}`\Psi` denotes the parameter space. In this case, the population loss is minimised at the true conditional posterior, assuming successful optimisation. After training, one can then sample from the posterior for any observation {math}`x_o` by drawing a sample from the base distribution and transforming it with the series of transformations {math}`T` (which are conditioned on {math}`x_o`). Notably, samples from normalizing flows are i.i.d. and follow the estimated density exactly (i.e., normalizing flows do not require any approximations to draw samples from the modelled density). In addition, normalizing flows can also evaluate the density of parameters under the posterior via Equation [](#sbi:eq:change_of_variables).

An important property of this loss is that it tends to be mass-covering: it will strongly penalise the normalizing flow being too narrow (as {math}`\log q(\theta \mid x)` becomes a large negative number, leading to very high loss). As such, for limited training data or for imperfect convergence of the normalizing flow, {math}`q(\theta \mid x)` will tend towards being broader than the true posterior distribution {math}`p(\theta \mid x)`. In many applications of SBI, this is a desirable feature: too narrow posteriors would indicate overly confident parameter estimates, which could lead to wrong claims of scientific discovery {cite:p}`hermans2022crisis`. This mass-covering behaviour is in contrast to variational inference, which typically uses divergence objectives whose covering or seeking behaviour depends on the divergence direction, with common formulations tending to be mode-seeking (i.e., towards a too narrow posterior approximation) {cite:p}`li2016renyi`. Since we operate with finite samples, none of these are guarantees.

(sec:sbi:embedding_nets)=
### Learning from high-dimensional data with embedding networks

Many models in science and engineering produce high-dimensional simulation outputs such as images or long time series. In order to estimate parameters underlying such data, NPE can be combined with *embedding networks* (Figure [](#fig:embedding_nets)) {cite:p}`lueckmann2017flexible`. Embedding networks are neural networks which take as input high-dimensional data {math}`x` and return a lower-dimensional summary statistic {math}`s(x)` of them. These summary statistics are then passed to the inference network, which uses the summary statistics to approximate the posterior. Ideally, these summary statistics will be (approximately) sufficient, i.e., still preserve all relevant information for the posterior, so that {math}`p(\theta \mid x) \approx p(\theta \mid s(x))` {cite:p}`fearnhead2011constructing,chen2021neural`.

:::{figure} assets/sbi/fig3.png
:label: fig:embedding_nets
:align: center

Neural Posterior Estimation (NPE) with embedding networks. Left: For many models, simulation outputs can be high-dimensional. For example, simulators might produce long time series, images, or i.i.d. data given a particular parameter set. Middle: In order to perform inference based on such observations, NPE can be combined with suitable embedding networks that efficiently reduce these data to lower-dimensional summary statistics. These summary statistics are then processed by the inference network, and both neural networks can be trained end-to-end. Right: Illustration of a posterior distribution underlying such observations.
:::

In NPE, the embedding network can be trained end-to-end with the inference network, that is, with the log-likelihood loss specified in Equation [](#eq:npe_loss). Provided that the embedding network is sufficiently expressive, it will automatically learn appropriate summary statistics, without requiring separate data-compression methods {cite:p}`chen2021neural` (although those might still be empirically useful in some cases {cite:p}`chen2023learning`). The choice of embedding network depends on the type of data modelled by the simulator. For example, for images, a popular choice for the embedding net are convolutional neural networks (CNNs), and recurrent neural networks (RNNs) might be useful for time-series data {cite:p}`lueckmann2017flexible`.

An important case is the setting in which the observation {math}`x_o` does not constitute a single observation, but rather stands for a *set* of observations {math}`x=\{x_1, \ldots x_N\}` which are thought to be sampled i.i.d. from the same parameter. In this case, the posterior distribution should be invariant to the ordering of the observations, which can be achieved by using a permutation invariant embedding network such as a set transformer {cite:p}`lee_set_2019`, as proposed by {cite:t}`chan2018exchangeable` for applications in population genetics. {cite:t}`radev2020bayesflow` proposed and evaluated using such networks for datasets with varying size.

Instead of learning summary statistics end-to-end with an embedding network, one can also manually extract summary statistics from the data and perform inference based on these statistics. This requires domain knowledge and, if the choice of summary statistics is poor, might lead to a significant loss of information about parameters, but it can be beneficial if one is interested in inferring parameters given particular features of data. Indeed, in many cases, the simulator is not able to reproduce *all* features of the (experimentally) observed data. Manually defined summary statistics can avoid such misspecification by focusing only on properties of the data which can be modelled by the simulator.

+++
(sec:sequential)=
## Amortized and sequential inference

As described above, NPE *amortizes* the cost of inference: After training on a simulated dataset generated from prior predictives (i.e., simulation results based on parameters drawn from the prior distribution), they can perform inference for *any* new observed data {math}`x_o`. In cases where one is interested in performing inference for many different observations, or where inference is time-critical, this can be highly beneficial. In cases where one only aims to perform inference for a single observation (or for few observations), NPE can be wasteful: By training a neural network on *prior* predictives, the neural network has to learn from a broad range of (simulated) data, even though it is eventually only evaluated at few observations.

*Sequential* methods have been developed to improve the simulation-efficiency of SBI in these cases. These methods draw parameters (which are then used to generate the training dataset) from a “proposal” distribution. This proposal distribution is chosen such that simulation outputs are expected to be closer to the observation than prior predictives. A popular choice for the proposal distribution is the posterior distribution obtained by running NPE with a limited number of simulations (but explicit active learning schemes have also been proposed {cite:p}`griesemer2024active`).

These methods are often called *sequential*, as the inference network is trained across multiple, sequentially simulated datasets, where the approximate posterior after each round guides generation of the next dataset. It has been demonstrated that sequential methods can improve simulation efficiency, often by an order of magnitude or more {cite:p}`lueckmann2021benchmarking,gloeckler2022variational,deistler2022truncated`. While potentially reducing the number of required simulations, using sequential methods with NPE can have a drawback: Drawing parameters from a proposal distribution (instead of from the prior) biases the posterior distribution towards regions that were oversampled in the proposal (compared to the prior). Several methods have been developed to overcome this. {cite:t}`papamakarios2016fast` proposed to train the neural network with the standard log-likelihood loss, and to then correct the posterior post-hoc. {cite:t}`lueckmann2017flexible` suggested to importance-weight the loss, such that the density estimator directly approximates the posterior after training. {cite:t}`greenberg2019automatic` introduced a contrastive loss, which implicitly frames conditional density estimation as a classification problem {cite:p}`durkan2020contrastive`, and which can be combined with arbitrary acquisition functions {cite:p}`griesemer2024active`. Finally, {cite:t}`BlumFrancois_10` and {cite:t}`deistler2022truncated` proposed to draw parameter sets from a restricted region of the prior, such that no modifications of the loss function or post-hoc corrections are required. Notably, for many other SBI methods, such as Neural Likelihood Estimation, NLE {cite:p}`papamakarios2019sequential,lueckmann2019likelihood`, or Neural Ratio Estimation, NRE {cite:p}`hermans2020likelihood,thomas2022likelihood,durkan2020contrastive,miller2022contrastive`, parameters for the training dataset can be drawn from *any* distribution and the neural networks can nonetheless be trained with standard loss functions, as the prior is explicitly taken into account during inference with MCMC or variational inference.

+++
## Evaluating the correctness of SBI methods

### The problem of evaluating SBI methods

After having obtained an approximate posterior with SBI methods, a central challenge is to evaluate the quality of this posterior. This is difficult because, for any real-world task, the ground-truth posterior is unavailable.

Several diagnostic tools have been developed to overcome this limitation. Note, however, that many of these tools do not provide *sufficient* conditions for posterior correctness. Instead, they provide *necessary* conditions for posterior correctness, and thus enable detection of inaccuracies in the posterior estimate. Other methods can, in principle, provide sufficient conditions for correctness, but rely on additional hyperparameters or the accuracy of other trained neural networks, which can compromise the accuracy of the diagnostic tool. Below, we focus on a set of diagnostic tools called coverage diagnostics, and we then briefly describe other methods.

### Coverage diagnostics

:::{figure} assets/sbi/fig5.png
:label: fig:diagnostics
:align: center

Coverage diagnostics for identifying inaccuracies in the approximate posterior. Left: Expected coverage based on the highest-probability density (HPD) region aims to detect whether the joint posterior distribution is over- or under-confident. Marginal simulation-based calibration (SBC) aims to detect inaccuracies in the marginals of the posterior distribution. Both of these methods produce rank distributions, which are optimal if the posterior lies on the diagonal, and indicate issues if it lies above or below the diagonal. Figure modified from @deistler2025simulation.
:::

A set of diagnostic tools that aim to detect inconsistencies in the posterior distribution are *coverage diagnostics* {cite:p}`cook2006validation,talts2018validating`. These methods begin by generating a calibration dataset: They sample parameters from the prior and run the simulator to generate samples {math}`\theta, x \sim p(\theta, x)`. For every {math}`x` in the calibration dataset, these methods then draw samples from the posterior {math}`\theta_{\text{post}} \sim q_{\psi}(\theta \mid x)` obtained with an SBI method. For amortized SBI methods, this process is fast, as inference can be performed for any {math}`x` without retraining or resimulating.

Coverage diagnostics then reduce posterior samples {math}`\theta_{\text{post}}` and the ground truth parameters {math}`\theta` to 1-dimensional quantities, via a reducing function {math}`f(\cdot): \mathbb{R}^N \rightarrow \mathbb{R}`, where {math}`N` is the parameter dimensionality. The posterior approximation is *calibrated* if, for any reducing function {math}`f(\cdot)`, the rank of {math}`f(\theta)` is distributed uniformly within {math}`f(\theta_{\text{post}})` (Figure [](#fig:diagnostics)). Different diagnostic tools use different mappings {math}`f(\cdot)` to detect posterior inconsistencies. For example, {math}`f(\theta)` can compute the posterior log-probability {math}`q_{\psi}(\theta \mid x)` (often referred to as *expected coverage* based on the highest-probability density (HPD) region) {cite:p}`miller2021truncated,hermans2022crisis,deistler2022truncated`, or it can pick the 1D marginals of {math}`\theta` {cite:p}`cook2006validation,talts2018validating`. In the former case, coverage diagnostics can detect inconsistencies in higher-order moments of the posterior (Figure [](#fig:diagnostics), left). In the latter case, coverage diagnostics cannot detect inconsistencies in higher-order moments of the posterior, but they can provide intuition about which parameter is inconsistent (Figure [](#fig:diagnostics), right).

A core advantage of coverage diagnostics is that they can be run with relatively small calibration sets (typically, {math}`\sim 200` simulations), and that they do not require further tuning of hyperparameters or training of a “diagnostic neural network”. This makes them robust and easy to deploy. However, coverage diagnostics typically provide a necessary, but not a sufficient condition for posterior correctness. In particular, if the approximate posterior {math}`q_{\psi}(\theta \mid x)` always returns the prior distribution {math}`p(\theta)` (instead of an approximation to the posterior), then it passes the above described coverage diagnostics. This highlights the need for additional diagnostic tools.

### Other methods

A simple tool for diagnosing the posterior approximation {math}`q_{\psi}(\theta \mid x)` is to perform *Posterior Predictive Checks* (PPCs). Given an observation, these methods draw samples from {math}`q_{\psi}(\theta \mid x_o)`, simulate them, and then compare the simulation outputs (the posterior predictives) to the observation. For every posterior sample {math}`\theta`, it should be possible to obtain a simulation result that closely matches the observation. If posterior predictive checks fail (e.g., because all posterior predictives are far away from the observation, or if some parameter sets can *never* match the observation), then this indicates issues in the accuracy of {math}`q_{\psi}(\theta \mid x)`. In particular it might be trained on too few simulations, or it might hint towards *model misspecification* (a case where the observation {math}`x_o` cannot be matched by *any* parameter set, as discussed in Section [Current developments](#sec:sbi:current_developments)).

Beyond coverage checks and PPCs, many other diagnostic tools for SBI have been developed over the past years. For example, some local calibration diagnostics aim to provide sufficient and necessary conditions for posterior correctness {cite:p}`linhart2024c2st,sailynoja2025posterior`. However, these methods often require to train an additional neural network and hinge on the accuracy of that additional trained neural network. Because of this, these diagnostic tools require larger calibration datasets and might themselves have issues (e.g., due to poor convergence of the additional trained neural network).

+++
## Alternative methods

Neural Posterior Estimation (NPE) directly estimates the posterior distribution. In addition to NPE, researchers have developed other methods for SBI. These approaches train neural networks to emulate either the likelihood {math}`p(x \mid \theta)` (Neural Likelihood Estimation, NLE) or the likelihood-to-evidence ratio {math}`p(x \mid \theta)/p(x)` (Neural Ratio Estimation, NRE) and then use traditional Bayesian inference methods (e.g., MCMC) to draw samples from the posterior. Finally, some recent methods also go beyond estimating only the posterior or the likelihood(-ratio). Below, we describe these alternative methods for SBI.

### Likelihood estimation

Neural Likelihood Estimation (NLE) uses neural density estimators (such as conditional normalizing flows) to estimate the likelihood {math}`p(x \mid \theta)` {cite:p}`papamakarios2019sequential`. This estimate can be learned from samples of the joint distribution {math}`\{(\theta_i, x_i)\}_{i=1}^{N} \sim p(\theta, x)` by optimising the negative log likelihood of simulation outputs given parameters. NLE trains a conditional density estimator {math}`\ell_{\psi}(x \mid \theta)`, such as a normalizing flow, by optimising {math}`\psi` towards the objective

```{math}
\psi^{\ast} \in \operatorname*{arg\,min}_{\psi} \left[ -\frac{1}{N} \sum_{i=1}^N \log \ell_{\psi}(x_i \mid \theta_i) \right].
```

After training, the trained model can be used to *emulate* the simulation process by drawing samples from the likelihood approximation {math}`\ell_{\psi}(x \mid \theta)`.

For posterior inference, the trained model is multiplied by the prior to obtain the (unnormalised) posterior estimate

```{math}
\begin{align}
    \hat{p}(\theta \mid x) \coloneqq \frac{1}{Z_{\psi}(x)}\ell_{\psi}(x \mid \theta) p(\theta),
    \qquad
    Z_{\psi}(x) \coloneqq \int \ell_{\psi}(x \mid \theta)p(\theta) \, \mathrm{d}\theta.
\end{align}
```

In order to draw samples from the (approximate) posterior distribution {math}`\hat{p}(\theta \mid x_o)`, a method to draw samples from an unnormalised distribution is necessary, for example, rejection sampling or Markov chain Monte Carlo (Section [Sampling from the approximate posterior using unnormalised inference models](#sec:sbi:sampling_unnormalized_models)). Normalizing flows enable (differentiable) evaluation of the likelihood approximation {math}`\ell_{\psi}(x_o \mid \theta)`, making this possible, albeit being potentially computationally expensive. Variational inference has been shown to speed up high-dimensional sampling from NLE {cite:p}`wiqvist2021sequential,gloeckler2022variational`. However, any of these sampling methods are typically slow and can introduce additional errors into the inference procedure. An additional disadvantage of NLE (compared to NPE) is that it cannot be combined with an embedding network that is trained end-to-end. One can embed {math}`x_o` *before* training with a separate neural network, e.g., an autoencoder, then perform NLE on the embedded simulation outputs {cite:p}`chen2021neural`.

NLE can also be used in a sequential scheme to reduce the number of required simulations for inference given a particular observation. These sequential NLE methods focus on a specific observation {math}`x_o`, generating data nearby {math}`x_o` iteratively {cite:p}`papamakarios2019sequential`. A major advantage of NLE in that scenario is that it does not require modifications of the loss function: The neural density estimator will converge to the likelihood for any distribution {math}`\tilde{p}(\theta)` from which parameters are drawn.

(sec:likelihood_to_evidence_ratio_estimation)=
### Likelihood-to-evidence ratio estimation

The likelihood-to-evidence ratio {math}`r(x \mid \theta) \coloneqq \frac{p(x \mid \theta)}{p(x)} = \frac{p(\theta, x)}{p(\theta) p(x)}` is a natural estimation target for simulation-based inference because Bayes’ rule gives {math}`p(\theta \mid x) = r(x \mid \theta)p(\theta)`. Ratio estimators allow more flexible architectures than diffeomorphic normalizing flow layers with tractable determinants. However, density ratios can span many orders of magnitude and therefore be numerically difficult to approximate directly. For this reason, one typically estimates {math}`\log r(x \mid \theta)`. A stable approach to estimating {math}`\log r(x \mid \theta)` converts the problem into a classification task. Several variants have been developed {cite:p}`cranmer2015approximating,durkan2020contrastive,miller2022contrastive,thomas2022likelihood`; here, we focus on the method of {cite:t}`hermans2020likelihood`.

NRE introduces a binary class label {math}`y` and defines the joint distribution {math}`\pi(\theta, x, y) \coloneqq \pi(\theta, x \mid y)\pi(y)`. We assign equal class probabilities, {math}`\pi(y=0) \coloneqq \pi(y=1) \coloneqq \frac{1}{2}`, and define the class-conditional distributions as

```{math}
\begin{align}
    \pi(\theta, x \mid y) \coloneqq
    \begin{cases}
        p(\theta) p(x), & y=0, \\
        p(\theta, x), & y=1.
    \end{cases}
\end{align}
```

{math}`p(\theta) p(x)` is called the product of marginals. It is best understood algorithmically: One takes two independent samples {math}`\theta_{a}, \theta_{b} \sim p(\theta)` and simulates one of them {math}`x_{b} \sim p(x \mid \theta_{b})`. Then the samples are distributed {math}`(\theta_{a}, x_{b}) \sim p(\theta)p(x)`. Sampling from the joint {math}`p(\theta, x)` is achieved by first sampling the prior {math}`\theta_{c} \sim p(\theta)` and simulating {math}`x_{c} \sim p(x \mid \theta_{c})`. We then have a sample {math}`(\theta_{c}, x_{c}) \sim p(\theta, x)`.

Bayes’ rule shows why this classification problem identifies the desired ratio:

```{math}
\begin{align*}
    \frac{\pi(y=1 \mid \theta,x)}{\pi(y=0 \mid \theta,x)}
    &= \frac{\pi(\theta,x \mid y=1)\pi(y=1)}{\pi(\theta,x \mid y=0)\pi(y=0)}
    &= \frac{p(\theta,x)}{p(\theta)p(x)}
     = r(x \mid \theta).
\end{align*}
```

Consequently, {math}`\pi(y=1 \mid \theta,x) = \frac{r(x \mid \theta)}{1+r(x \mid \theta)} = \sigma(\log r(x \mid \theta))`, where {math}`\sigma` denotes the sigmoid function. We therefore construct

```{math}
\begin{align*}
    & \hat{\pi}(y=1 \mid \theta, x) \coloneqq \sigma(f_{\psi}(\theta, x))
    & \text{ and } &
    &\hat{r}(x \mid \theta) \coloneqq \exp(f_{\psi}(\theta, x)),
\end{align*}
```

where {math}`\hat{\pi}(y=1 \mid \theta, x)` is a classifier, with {math}`\hat{\pi}(y=0 \mid \theta, x)=1-\hat{\pi}(y=1 \mid \theta, x)`, and {math}`\hat{r}(x \mid \theta)` is a likelihood-to-evidence ratio estimate. Both are parameterised by the neural network {math}`f_{\psi}(\theta, x)` with weights {math}`\psi`. We optimise {math}`\psi` by minimising the conditional KL divergence,

```{math}
:label: eqn:likelihood_to_evidence_ratio_loss

\begin{align}
    \psi^{\ast}
    &\in \arg\min_\psi D_{KL}\left(\pi(y \mid \theta, x)\middle\|\hat{\pi}(y \mid \theta, x)\right),
     \\
    D_{KL}\left(\pi(y \mid \theta, x)\middle\|\hat{\pi}(y \mid \theta, x)\right)
    &= \iint \pi(\theta, x) \sum_{y \in \{0, 1\}} \pi(y \mid \theta, x)
    \log \frac{\pi(y \mid \theta, x)}{\hat{\pi}(y \mid \theta, x)} \, \mathrm{d}x \, \mathrm{d}\theta.
    
\end{align}
```

Here, {math}`\pi(\theta, x)` is the marginal mixture induced by the constructed classification problem:

```{math}
\pi(\theta, x)
    \coloneqq \pi(y=1)\pi(\theta, x \mid y=1) + \pi(y=0)\pi(\theta, x \mid y=0)
    = \frac{p(\theta, x) + p(\theta)p(x)}{2}.
```

The conditional KL divergence differs from the binary cross-entropy used to train the classifier only by the conditional entropy of {math}`\pi(y \mid \theta,x)`, which is constant with respect to {math}`\psi`. The two objectives therefore have the same minimiser. Assuming sufficient model capacity and successful optimisation, the conditional KL divergence is zero at the optimum, so {math}`\pi(y=1 \mid \theta,x) = \sigma(f_{\psi^{\ast}}(\theta,x))` and {math}`f_{\psi^{\ast}}(\theta,x) = \log r(x \mid \theta)` almost everywhere under {math}`\pi(\theta,x)`.

We can now form the unnormalised posterior approximation {math}`\hat{p}'(\theta \mid x) \coloneqq \hat{r}(x \mid \theta)p(\theta) = p(\theta)\exp(f_{\psi}(\theta,x))`. Unless the classifier recovers the exact ratio, {math}`\hat{p}'` need not integrate to one. Its normalised counterpart is

```{math}
\hat{p}(\theta \mid x) \coloneqq \frac{\hat{p}'(\theta \mid x)}{Z_{\psi}(x)} = \frac{\hat{r}(x \mid \theta)}{Z_{\psi}(x)} p(\theta) = \frac{\exp(f_{\psi}(\theta,x))}{Z_{\psi}(x)}p(\theta),
```

with {math}`Z_{\psi}(x) \coloneqq \int \exp(f_{\psi}(\theta,x))p(\theta)\,\mathrm{d}\theta`. At the exact optimum, {math}`\exp(f_{\psi^{\ast}}(\theta,x))=r(x \mid \theta)` and hence {math}`Z_{\psi^{\ast}}(x)=1`. In practice, posterior behaviour is usually characterised using samples, and samplers for unnormalised distributions do not require {math}`Z_{\psi}(x)` to be computed.

One can draw samples from {math}`\hat{p}(\theta \mid x)` using any sampling method for unnormalised distributions, just like with NLE (Section [Sampling from the approximate posterior using unnormalised inference models](#sec:sbi:sampling_unnormalized_models)). The ratio-based neural network {math}`f_{\psi}(\theta, x)` can be both more expressive and cheaper than the normalizing flow-based likelihood approximation from NLE. Flows require parameterising an efficient diffeomorphism with a tractable log determinant, whereas NRE permits arbitrary neural-network architectures. One limitation is that the standard NRE objective in Equation [](#eqn:likelihood_to_evidence_ratio_loss) does not generally attain the same asymptotic efficiency as maximum-likelihood density estimation {cite:p}`rhodes2020telescoping,choi2022density,yu2025density`. Alternative objectives have therefore been explored for ratio estimation {cite:p}`glaser2022maximum,miller2023simulationbased`.

(sec:sbi:sampling_unnormalized_models)=
### Sampling from the approximate posterior using unnormalised inference models

While NPE approximates the posterior directly, NLE and NRE require an additional inference step to draw samples from the posterior. To this end, one can use general-purpose sampling methods (see Chapter [](#chap:sampling:intro)) and adapt them to sample from an unnormalised posterior estimate. Rejection sampling or reweighted sampling from the prior are natural choices for simple problems with low dimensionality due to straightforward embarrassingly parallel implementations {cite:p}`murphy2012machine`. Both of these methods start by drawing {math}`N` samples from the prior

```{math}
:label: eqn:prior_samples

\begin{aligned}

    \theta_i \sim p(\theta), & & i \in \{1, 2, \ldots, N \}.
\end{aligned}
```

In *rejection sampling*, we choose a constant {math}`M \geq 1` such that {math}`M \cdot p(\theta) \geq \hat{p}'(\theta \mid x_o)` for all {math}`\theta` in the support of {math}`\hat{p}'(\theta \mid x_o)`, where {math}`\hat{p}'(\theta \mid x_o)` denotes the unnormalised posterior density. For each of the {math}`N` samples, we draw

```{math}
\begin{aligned}
    u_i \sim \mathcal{U}(0, 1), & & i \in \{1, 2, \ldots, N \},
\end{aligned}
```

where we let {math}`\mathcal{U}(a, b)` denote the one-dimensional uniform distribution between {math}`a` and {math}`b`. This algorithm yields samples from the approximate posterior {math}`\hat{p}(\theta \mid x_o)` in the following way:

```{math}
u_i < \frac{\hat{p}'(\theta_i \mid x_o)}{M \cdot p(\theta_i)} \implies \theta_{i} \sim \hat{p}(\theta \mid x_o).
```

Rejection sampling generates {math}`N/M` samples, on average. It is efficient when {math}`M \approx 1`, *i.e.,* in low dimensions and when the prior and approximate posterior have similar support. Rejection sampling is inefficient when the approximate posterior is narrow compared to the prior.

In *reweighted sampling*, we use the samples from the prior in Equation [](#eqn:prior_samples), but assign them a self-normalised importance weight. The weights and self-normalised weights are defined as

```{math}
\begin{align}
    \widetilde{w}_i &\coloneqq \hat{p}'(\theta_i \mid x_o),
    &
    w_i &\coloneqq \frac{\widetilde{w}_i}{\sum_{j=1}^{N} \widetilde{w}_j} = \frac{\hat{p}'(\theta_i \mid x_o)}{\sum_{j=1}^{N} \hat{p}'(\theta_j \mid x_o)}.
\end{align}
```

Histograms and moments of the distribution can be computed using weighted samples {math}`w_i \theta_i`. The variance of the weights can be large, implying one sample takes most of the probability mass. This occurs in the same situations when rejection sampling is inefficient.

One sampling method that can be more robust to high dimensionality and approximate posteriors that make up only a small volume (compared to the prior) is *Markov chain Monte Carlo*. Rather than explaining further, we simply define the Metropolis-Hastings acceptance probability and refer the reader to Chapter [](#chap:sampling:intro). Given a fixed {math}`x` and proposal distribution {math}`\mathcal{T}` with transition density {math}`\mathcal{T}(\theta' \mid \theta)` from {math}`\theta` to {math}`\theta'`, the acceptance probability {math}`\alpha(\theta' \mid \theta, x)` in a Metropolis-Hastings step is

```{math}
\begin{aligned}
    \alpha(\theta' \mid \theta, x_o) 
    &\coloneqq \min\left(1, \frac{\hat{p}(\theta' \mid x_o)}{\hat{p}(\theta \mid x_o)} \frac{\mathcal{T}(\theta \mid \theta')}{\mathcal{T}(\theta' \mid \theta)} \right) 
    = \min\left(1, \frac{\hat{p}'(\theta' \mid x_o)/\cancel{Z_{\psi}(x_o)}}{\hat{p}'(\theta \mid x_o)/\cancel{Z_{\psi}(x_o)}} \frac{\mathcal{T}(\theta \mid \theta')}{\mathcal{T}(\theta' \mid \theta)} \right)
\end{aligned}
```

This acceptance probability does not depend on {math}`Z_{\psi}(x_o)` and can therefore draw samples from the approximate posterior even when {math}`\hat{p}'(\theta \mid x_o)` is not normalised. Designing an efficient transition kernel and identifying convergence is an open area of research.

### Estimating multiple quantities at once

Finally, some recent methods go beyond estimating the posterior, the likelihood, or the likelihood-ratio and estimate several of these quantities, typically with separate models {cite:p}`wiqvist2021sequential,gloeckler2022variational,radev2023jana`. Recently, the Simformer {cite:p}`gloeckler2024allinone` has been proposed to estimate the full joint distribution {math}`p(\theta, x)` as well as all of its conditionals and marginals in a single model, based on transformers and a diffusion model. Thereby, the Simformer improves the flexibility of SBI and it combines desirable features of NPE (e.g., to directly draw samples from the posterior) and NLE (e.g., to emulate the simulator).

+++
(sec:sbi:current_developments)=
## Current developments

Neural Posterior Estimation (NPE), as well as alternative methods such as Neural Likelihood Estimation (NLE) or Neural Ratio Estimation (NRE), have been demonstrated as powerful tools for SBI. Over the past years, researchers have extended these methods in many ways, and have developed new SBI methods that go beyond estimating the posterior or likelihood(-ratio). We outline recent trends in using neural networks for SBI.

### Better generative models

Normalizing flows have been a popular choice for NPE for several years, likely due to their robustness in training, their ability to quickly (and exactly) evaluate the log-probability of samples, and their mass-covering property. However, the fact that the transformation {math}`T` trained by normalizing flows has to be invertible can constrain the flexibility of the normalizing flow. This may reduce the accuracy of posterior estimates of normalizing flows, especially for high-dimensional and structured parameter spaces (e.g., if the parameters are a time series or an image). To overcome this, recent methods have explored other conditional generative models for NPE. In recent years, diffusion models have been particularly popular due to their ability to estimate high-dimensional parameter spaces, and to maintain the ability to evaluate the log-probability via the probability flow ODE (albeit at significantly higher computational cost) {cite:p}`sharrock2022sequential,geffner2023compositional,gloeckler2024allinone`. Many other generative models have been explored for NPE, ranging from flow-matching {cite:p}`dax2023flow` to consistency models {cite:p}`schmitt2024consistency`, generative adversarial networks {cite:p}`ramesh2022gatsbi`, energy-based models {cite:p}`glaser2022maximum`, and tabular foundation models {cite:p}`vetter2025effortless`. We expect that any major advances in (conditional) generative modelling will directly carry over to improvements in NPE.

### Improving the flexibility of simulation-based inference

A core limitation of NPE with normalizing flows is that it requires the inference task to be known upfront. If, for example, the prior changes, when the simulator is modified, or when the data representation changes, NPE has to be re-run in order to perform inference (including running simulations and training the neural network).

To overcome this, many recent methods have improved the flexibility of NPE. First, in order to perform inference given any number of i.i.d. trials, permutation-invariant embedding networks can be used, as described above {cite:p}`radev2020bayesflow`. Alternatively, diffusion models enable NPE to estimate the posterior distribution given any number of i.i.d. datapoints, even when the neural network was trained on pairs of parameters and a single simulation result {cite:p}`geffner2023compositional,linhart2026diffusion,gloeckler2024compositional`. Second, in order to enable more flexible specification of prior or simulator, it has been proposed to amortize over hyperparameters of prior or simulator {cite:p}`starostin2025fast,elsemuller2024sensitivityaware,muller-icml23a`. In that case, the inference network is conditioned not only on data, but also on additional values describing the prior or simulator. At inference time, one can then perform inference for any prior and for any simulator. Third, in some cases and only for some observations, some parameters may be known and should not be inferred, but instead should be kept fixed {cite:p}`deistler2022energy,gloeckler2024allinone`. In other cases, data may not always be complete, and parts of an observation might be missing. In order to flexibly perform amortized inference in these cases, several methods enable the estimation of any parameter conditional of NPE and systematically deal with missing data {cite:p}`rozet2021arbitrary,deistler2022energy,gloeckler2024allinone,verma2025robust`.

### Improving robustness to model misspecification

A fundamental challenge for SBI emerges when the simulator is *misspecified*, i.e., when it does not match reality. In the most extreme case, the observation {math}`x_o` cannot be generated by the simulator (i.e., the combination of prior and simulator) with any set of parameters and with any level of simulator noise.

This can have a drastic impact on SBI: While one might expect that, for misspecified observations, the approximate posterior will have mass in parameter regions that generate predictives that are close to the observation, or that the approximate posterior should be very uncertain, this is not the case. Indeed, neural network-based methods for SBI have been shown to react erratically to the misspecified observations and can produce non-sensible posteriors {cite:p}`cannon2022investigating`.

Several methods have been proposed to improve the robustness of SBI in these scenarios, including adding noise to the simulation outputs {cite:p}`ward_robust_2022`, learning statistics of the observations {cite:p}`huang2023learning` such that they are not misspecified, training on unlabelled {cite:p}`mishra2025robust` and labelled {cite:p}`wehenkel2024addressing` observations when available, or changing the target of inference {cite:p}`gao2024generalized`.

### Conservative and calibrated posterior approximations

Approximate SBI posteriors can be overconfident[^CHnLu5pqlH][^CHnLu5pqlH]: This occurs when the approximate posterior is narrower than the ground truth., particularly when simulations are limited, motivating methods that promote conservative or calibrated uncertainty estimates {cite:p}`hermans2022crisis`. Balanced neural ratio estimation modifies the NRE training objective to favour conservative approximations, while preserving its Bayes-optimal solution {cite:p}`delaunoy2022towards`, by effectively regularising the estimate to be closer to the prior. This balancing principle has also been extended to NPE and contrastive NRE {cite:p}`delaunoy2023balancing`. Alternatively, differentiable relaxations of coverage error can be included directly in the training objectives of amortized SBI methods, encouraging calibrated posteriors with moderate additional computational cost {cite:p}`falkiewicz2023calibrating`. However, expected coverage is a necessary rather than sufficient diagnostic: an approximation can exhibit nominal coverage while still differing substantially from the true posterior. Ratio coverage plots address some of these blind spots by comparing the approximate and true posteriors through an estimated density ratio {cite:p}`lipp2026generalizing`.

+++
## Discussion

Many processes in the natural sciences and in engineering are best described by complex, stochastic, and potentially non-differentiable simulators. For many such models, the likelihood can become excessively expensive or even impossible to evaluate. In order to perform inference with traditional Bayesian inference methods such as MCMC, scientists and engineers had to modify their model such that the likelihood can be efficiently evaluated, but which comes at the cost of reducing the fidelity of the simulator.

Simulation-based inference (SBI) makes Bayesian inference accessible to the widest class of models: It can be applied to any black-box simulator for which one can run forward simulations. In recent years, neural networks have largely improved the accuracy and applicability of SBI. These methods generate a database of parameters and corresponding simulation outputs, and then train neural networks to learn the statistical relationship between these quantities. After training, the neural network can be evaluated at any observation and enables us to infer the posterior distribution without further simulations or retraining.

SBI with neural networks has already been applied to perform inference in a wide range of disciplines in science and engineering. We expect that the continued progress in improving the accuracy, flexibility, and robustness of these approaches will have a profound impact on many fields, and we hope that it will enable new scientific discoveries.