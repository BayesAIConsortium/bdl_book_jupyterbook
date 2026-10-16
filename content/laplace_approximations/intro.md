(cha:laplace:introduction)=
# Introduction to Laplace approximations

+++
Astute readers of the preceding chapter will have noticed that the methods introduced in [](#part:sampling_methods) come with a concerning amount of computational and memory overhead. Even a perfectly efficient Monte Carlo method, one that draws an i.i.d. sample from the posterior in every single step, still requires us to *store* multiple samples—entire copies of the parameter vector—in memory. At prediction time, we then have to compute the forward pass of the model for each of these samples. The number {math}`k` of samples enters at least multiplicatively into the computational cost {math}`c` of whatever it is we are doing with our neural network: We will pay a price of {math}`\Omega(kc)` (the Landau {math}`\Omega`. Yes, that’s a *lower* bound), both in memory and computation, to turn a non-Bayesian computation into a Bayesian one. At a time when prominent non-Bayesian deep models are already maxing out the capacity of the best multi-GPU workstations, this is a nonstarter, and arguably the main reason why Bayesian deep learning has acquired a reputation for unacceptably high computational cost.

This Part, however, will introduce a different, more practically minded way to endow neural networks with Bayesian uncertainty. Its history is just as old as that of sampling methods, and was born out of computational necessity in the early days of deep learning {cite:p}`mackay1992bayesian`. In fact, it was originally invented in the early 19th century, long before digital computers {cite:p}`laplace1820theorie`, to turn intractable integrals into tractable ones, at any cost. As computation is once again a bottleneck in the deep learning age, Laplace approximations have re-emerged as a practical tool {cite:p}`NEURIPS2019_b3bbccd6,daxberger2021laplace,immer2021scalable` with new twists that push the envelope of automatic differentiation {cite:p}`Dangel2020BackPACK,dangel2025position` and numerical linear algebra {cite:p}`cinquin2024fsp`. It also offers new perspectives on deep learning from the perspective of differential geometry {cite:p}`NEURIPS2023_395371f7,roy2024reparameterization`. As we will see, it now comes with great power, and grave dangers.

:::{figure} assets/laplace_intro/fig_laplace_intro.png
:label: fig:laplace:sketch
:align: center

A linearized Laplace approximation on the most basic feed-forward perceptron $f(x,\theta) = \theta_2 \cdot \operatorname{ReLU}(\theta_1 x + 1)$ for the dataset $X=(1,-1),Y=(-1,-1)$, with regularizer $\Omega(\theta)=0.2 \cdot \|\theta\|^2$ and square loss. The grey contour lines denote the energy function. The black dot represents the minimum $\theta_*$; the thick black ellipses denote one (solid) and two (dashed) standard deviations of the Laplace approximation. The Gaussian approximation defined by the Laplace approximation clearly captures the *local* geometry, but not the full *global* shape of the posterior.
:::

(sec:laplace:textbook-derivation)=
## Laplace approximations are local and differential

+++
As in previous chapters, consider a differential program {math}`f: \mathbb{R}^{D}\times\mathbb{R}^{P}\to\mathbb{R}^{C}` mapping inputs {math}`x\in\mathbb{R}^{D}` to outputs {math}`f(x,\theta)\in\mathbb{R}^{C}`, parametrised by {math}`\theta\in\mathbb{R}^{P}`. Consider the regularized empirical risk {math}`\regrisk\in\mathbb{R}` defined as

```{math}
:label: eq:laplace:regularized-empirical-risk

\regrisk(\theta) = \sum_{n=1} ^N \ell(f(x_n,\theta),y_n) + \Omega(\theta).
```

with loss function {math}`\ell\colon \mathbb{R}^{C} \times \mathbb{R}^{C} \to \mathbb{R}` and regularizer {math}`\Omega\colon \mathbb{R}^{P} \to \mathbb{R}`, both strictly convex and twice continuously differentiable.[^footnote-1]

Also as in previous chapters, we interpret this function as an energy function, the unnormalised negative logarithm of a posterior:

```{math}
:label: eq:laplace:energy

\regrisk(\theta) = -\log p(Y \mid X, \theta) - \log p(\theta) + \text{const} = -\log p(\theta\mid \mathcal{D}) + \text{const}.
```

:::{prf:definition}

+++
A Laplace approximation is any algorithm within a general class of methods that approximate {math}`\regrisk(\theta)` with a quadratic function

```{math}
\regrisk(\theta) \approx \frac{1}{2}(\theta-\theta_*)^\top \Psi(\theta_*) (\theta-\theta_*) + \text{const},
```

where {math}`\Psi(\theta_*)\in\mathbb{R}^{P\times P}` is a second-order linear differential operator acting on {math}`\regrisk(\theta_*)` at the point {math}`\theta_*\in\mathbb{R}^{P}`. A Laplace approximation is thus defined by the tuple {math}`(\theta_*,\Psi)`, which we will call a **point estimate** and its **curvature**, respectively.
:::

+++
Under the assumption of Equation [](#eq:laplace:energy), any Laplace approximation is directly associated with an approximate **Gaussian measure on the parameter space**,

```{math}
:label: eq:laplace:energy-approximation

p(\theta\mid\mathcal{D}) =\frac{1}{Z(\mathcal{D})}\exp(-\regrisk(\theta))
```

```{math}
:label: eq:laplace:taylor-approximation

p(\theta\mid\mathcal{D}) \approx \frac{1}{Z(\mathcal{D})}\exp\left(-\frac{1}{2}(\theta-\theta_*)^\top \Psi(\theta_*) (\theta-\theta_*)\right)
```

```{math}
:label: eq:laplace:gauss-posterior

p(\theta\mid\mathcal{D}) = \mathcal{N}(\theta;\theta_*, \Psi^{-1}(\theta_*)).
```

+++
The textbook, but impractical, version of the Laplace approximation arises from the Taylor expansion at a local minimizer of the regularized risk:

```{math}
:label: eq:laplace:hessian

[\Psi(\theta)]_{ij} = \left[\nabla_\theta \nabla_\theta ^\top  \regrisk(\theta)\right]_{ij}
    = \frac{\partial^2 \regrisk(\theta)}{\partial \theta_i \partial \theta_j}, \qquad
    \theta_* = \operatorname*{arg\,min}_{\theta\in\mathbb{R}^{P}} \regrisk(\theta).
```

That is, we pick a point {math}`\theta_*` where the gradient vanishes, and compute the Hessian there (the matrix {math}`\Psi(\theta_*)` being the matrix representation of the aforementioned operator, evaluated at {math}`\theta_*`).

## What is to like about Laplace?

+++
Central to the appeal of Laplace approximations is that they are an *analytic post-processing* step that can be tagged onto the established way of training a deep model:

:::{admonition} How to construct a Laplace approximation:

+++
1.  Train your deep network in any way you know and trust, using whatever optimiser you like. Wait for your training loop to terminate and return the point estimate {math}`\theta_*`.

2.  After training has finished (!), compute {math}`\Psi(\theta_*)`. Since it is a second-order differential operator, it can be computed with automatic differentiation, using your favourite auto-diff framework.
:::

+++
And that’s it! You now have a “Bayesian uncertainty” estimate in parameter space, in the form of the Gaussian from Equation [](#eq:laplace:gauss-posterior). Let us spell out the advantages of this recipe, some more obvious, some less so, explicitly:

*   **You get to keep your point-estimate**. People who train neural networks, especially if it is a large model, spend extreme resources to find a good value {math}`\theta_*`. Even if this particular value may not hold up to the theoretical demands of Bayesian analysis, it nevertheless embodies a significant investment, so it is appealing to keep working with this valuable object. For Laplace, this point estimate turns directly into a posterior mean. And to state the obvious: The computation overhead of this part is strictly *zero* over whatever the training cost to begin with.

*   **You get to use established tools**. Constructing the Gaussian of Equation [](#eq:laplace:gauss-posterior), requires precisely two computational tools: *Automatic differentiation* to get {math}`\Psi`, and *numerical linear algebra*, to find its (approximate) inverse. These are arguably *the* most powerful, reliable, optimised functionalities available to the machine learning engineer. They can be made extremely numerically robust and conveniently automated, or tailored for radical computational efficiency. And, using probabilistic numerical linear algebra, the discretisation and iteration error of these internal computations can be tracked for downstream consideration (More about numerical linear algebra tricks in Chapter [](#cha:laplace:linearized) and [](#part:scalability)).

*   **Laplace is post hoc**. Since {math}`\Psi` is only needed after {math}`\theta_*` has been found, its cost is not a function of the complexity of training. In fact, the construction of a Laplace approximation is effectively separated from training entirely, and can be done on a pre-trained model downloaded from the Internet. The only caveat is that, since {math}`\Psi` is a function of the regularized risk, we need access to the training data {math}`\mathcal{D}` – or some compressed representation thereof.

*   **Laplace is analytical**. Laplace approximations return a Gaussian probability measure, whose parameters are computed using optimization, automatic differentiation, and linear algebra. That is, a deterministic quantity – a probability density function – constructed deterministically. There is no source of randomness involved, apart from the potentially stochastic nature of training to find {math}`\theta_*`. As we will see in Chapter [](#cha:laplace:analytic_functionality), this makes Laplace approximations interesting for downstream applications and decisions that require analytic marginalization over the parameters.

*   **Deep theoretical meaning**. We will find in Chapters [](#cha:laplace:linearized) and [](#cha:laplace:diffgeo) that the local differential nature of the curvature at the heart of Laplace approximations gives them profound theoretical connections, and intuitive interpretations. In fact, they open up a whole self-contained theory of deep learning founded on the differential geometry of the energy manifold associated with {math}`\regrisk`.

*   **Laplace turns every deep network into a Gaussian process**. In Chapter [](#cha:laplace:linearized), we will see that the Gaussian measure {math}`\mathcal{N}(\theta;\theta_*, \Psi^{-1}(\theta_*))` constructed by the Laplace approximation can be analytically pushed forward to the output space of the network {math}`f(x,\theta)`, where it manifests itself as a Gaussian process, whose mean-function is exactly the trained network {math}`f(x,\theta_*)`, and whose covariance function (kernel) is given by a structured object, a *tangent kernel* (developed further in [](#part:kernel_methods)). This is a powerful perspective for both practical and conceptual reasons: On the practical side, we can more or less directly transport a lot of notions from the Gaussian process literature to deep learning, recovering advanced functionality sorely missing from non-Bayesian deep learning. And on the conceptual side, the tangent kernel contains very specific terms that capture modular, interpretable statements about the role of the training data, the network architecture, and regularization that make deep learning more transparent and interpretable.

## Practical Considerations

+++
On first encountering Laplace approximations in deep learning, people tend to have one of two, mutually almost contradictory reactions:

### Isn't this too simplistic?

+++
A Taylor-expansion? Really? From a fundamentalist Bayesian perspective, a quadratic function is a poor approximation of the posterior, with all its non-convex nonlinear glory (cf. [](#fig:laplace:sketch)). But compared to the day-to-day practice of training deep networks, Laplace approximations are a big step forward, from a *zeroth order* approximation (a point estimate) to a *second-order* approximation (a Gaussian probability measure). They offer an easy gateway into the Bayesian world for the deep learning practitioner: Just take your trained model, compute a curvature, and Bob’s your uncle.

We should also be careful not to appeal to unrealistic ideals, but compare to realistic alternatives instead. The true posterior may be the theoretical target of all Bayesian analysis, but is completely intractable. All Bayesian methods that are actually accessible are approximations, of different nature. For example, if we draw an ensemble of {math}`S\ll P` samples {math}`[\theta_s]_{s=1}^S` from the posterior, and use it to construct a posterior covariance {math}`\hat{\Psi}^{-1} = \frac{1}{S} \sum_{s=1}^S (\theta_s-\theta_*)(\theta_s-\theta_*)^\top` over the parameters {math}`\theta`, that matrix {math}`\hat{\Psi}` will be of rank at most {math}`S`. If we use this object as a proxy for “posterior uncertainty”, we may severely over-estimate confidence intervals. A Laplace approximation might overlook nonlocal aspects of the posterior’s geometry, but it can construct an approximate covariance matrix of full rank.

### Isn't this too expensive?

+++
*Hang on, we need a Hessian over the parameter space? That’s a matrix of size {math}`P\times P`, so just storing it will create a memory overhead of {math}`\mathcal{O}(P^2)`. Oh, and then we need to compute its inverse!? That will incur a computational cost of {math}`\mathcal{O}(P^3)`, right?* The flaw of such thoughts is that they use worst case statements about objects that are entirely under the user’s control.

It is better to think of the curvature {math}`\Psi` as a *linear operator* {cite:p}`dangel2025position`, a function mapping vectors to vectors. Such a function need not require the explicit instantiation of a matrix in memory. Chapters [](#cha:laplace:linearized) and [](#cha:laplace:analytic_functionality) will introduce several possible choices for this operator, spanning a range of computational and memory costs. These include the generalised Gauss-Newton approximation ([Section {name}](#sec:laplace:fisher)), Kronecker-factored approximate curvature, and diagonal approximations to the Hessian ([Section {name}](#sec:laplace:kfac)). In fact, in [Section {name}](#sec:laplace:calibrated_uncertainty) we will see that even the trivial choice {math}`\Psi=\alpha I` for an arbitrary {math}`\alpha \in \mathbb{R}_+` (i.e. just replacing the point estimate with an arbitrary Gaussian measure) already fixes some tangible, nontrivial pathologies of non-Bayesian deep learning.

Likewise, numerical linear algebra offers a rich toolbox of iterative, sparse, sketched, etc. methods to solve linear systems of equations of the form {math}`\Psi x = b` for {math}`x` without the need to even explicitly form an estimate of the matrix inverse {math}`\Psi^{-1}` ([Section {name}](#subsec:laplace:lowrank)). And there are probabilistic versions of these methods that can partially track and estimate the error of their output {cite:p}`{e.g.}NEURIPS2022_4683beb6,cockayne2019bayesian,NEURIPS2020_4afd521d`. The memory and compute cost of Laplace approximations can thus be completely controlled by the user, opening up a wide spectrum of algorithms of varying quality and cost.

On the flip side, this also means that constructing Laplace approximations, just like training deep networks, is not a turnkey operation, and requires care and expertise. But emerging software tools increasingly encapsulate these choices, and provide abstraction layers that allow constructing Laplace approximations with a few lines of code.

## All approximations are wrong

+++
Despite all the arguments in their favour, Laplace approximations remain just that: approximations. Their local nature – the fact that they only use differential quantities evaluated at a single point in the parameter space – is both their principal strength and their principal weakness. As [](#fig:laplace:sketch) shows, the Gaussian posterior created by a Laplace approximation may not capture the geometry of the posterior – not even within the region of the parameter space delineated by one or two standard deviations of that Gaussian. And when intricate numerical routines are used to operate on the curvature operator {math}`\Psi`, their own errors can also severely erode the quality of the approximation.

Laplace approximations should thus best be viewed as an important tool of first resort. At least their most basic variants can be constructed on virtually any deep architecture, often with minimal overhead in both implementation and computation. They can be a powerful diagnostic tool, elucidating the interplay between data and architecture; and offer structured, controllable functionality for everyday requirements like uncertainty quantification, model adaptation, continual learning, and more. But whenever the stakes are high, for example in safety critical applications, or when uncertainty is used to draw scientific conclusions, additional safeguards are needed. In such settings, Laplace approximations may be paired with some of the other frameworks described in this book, to provide a more robust and calibrated uncertainty estimate.

:::{admonition} Takeaways:

+++
A Laplace approximation is a quadratic approximation to the log posterior associated with a deep model, interpreted as a Gaussian posterior on the parameter space, or a Gaussian process posterior on the predictive (logit) outputs of the model. Such approximations are constructed by first training a point estimate for the parameters in the “normal” way, then computing the curvature operator (or kernel) post hoc. By this nature, Laplace approximations are *local* and *differential*. On the one hand, this is a weakness, because they fail to capture *global*, *non-linear* structure in the posterior. On the other hand, this is a strength, because it allows the use of powerful automatic differentiation and matrix-free, probabilistic linear algebra, and provides theoretical descriptions of deep learning in the terms of differential geometry. The following chapters will provide further details on these aspects.
:::

[^footnote-1]: Note that convexity here is only assumed for the loss and regularizer. The full loss $\regrisk(\theta)$ is not generally convex, since the network $f$ is a nonlinear function of the parameters $\theta$.
