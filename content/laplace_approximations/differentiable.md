(cha:laplace:model_selection)=
# Model selection with the marginal likelihood

+++
The main intractability of high-dimensional Bayesian inference, as mentioned earlier in the book, stems from the difficulty of computing the normalising constant of the posterior distribution—the *marginal likelihood* (or *evidence*). While the MCMC methods presented before circumvent this problem by allowing us to sample from an *unnormalised* posterior distribution, the Laplace approximation offers the benefit that we can compute this quantity in closed form due to the Gaussianity. This not only enables inference and predictions (as shown in the previous chapters), but has one additional important use case: *model selection*. In this chapter, we will motivate why the marginal likelihood is a useful model selection criterion, explain how it is computed with the Laplace approximation, and present several practical use cases for model selection in Bayesian deep learning.

(sec:laplace:marginal_likelihood)=
## Marginal likelihood for model selection

+++
There are several perspectives to understand why the marginal likelihood can be a useful criterion for model selection. We will review some different ones in this section to hopefully convince you that this pesky normalisation constant is not only a nuisance but a powerful tool when wielded in the right way.

### Hierarchical Bayes

+++
Let us look at the familiar Bayes’ rule, but with an explicit conditioning on our model of choice, which we will denote by {math}`\model`. This model can encompass both the prior and the likelihood, so in the case of Bayesian deep learning, for instance, the neural network architecture, the likelihood noise level, the prior variance, or its distributional form. In short, the model {math}`\model` contains everything we have to choose actively before we can start to perform inference. The model-conditioned Bayes’ rule looks like this:

```{math}
:label: eq:laplace:model_cond_post

p(\theta \mid \mathcal{D}, \model) = \frac{p(\theta \mid \model) \, p(\mathcal{D} \mid \theta, \model)}{p(\mathcal{D} \mid \model)}
```

If we assume now that we do not actually know how to select the model in advance and would like to be Bayesian about it, we could attempt to treat {math}`\model` as a random variable in a hierarchical Bayesian model, the posterior of which would take the form

```{math}
:label: eq:laplace:hyperposterior

p(\model \mid \mathcal{D}) = \frac{p(\model) \, p(\mathcal{D} \mid \model)}{\int p(\model) \, p(\mathcal{D} \mid \model) \, \mathrm{d}\model}
```

If we further assume that *a priori*, we do not have any preference for any particular model, that is, {math}`p(\model)` is uniform across models (brushing under the rug the fact that this may be an improper prior if the space of models is not integrable), this yields

```{math}
p(\model \mid \mathcal{D}) \propto p(\mathcal{D} \mid \model)
```

We now see that the hierarchical posterior over models is proportional to the marginal likelihood term in Equation [](#eq:laplace:model_cond_post). Therefore, if we want to select the model that maximises the posterior (i.e., a MAP estimate over models), we just need to optimise {math}`\model` with respect to the marginal likelihood! Since this is essentially just performing a maximum likelihood selection on the second level of this Bayesian hierarchy, it is also often called *maximum likelihood II (ML-II)* model selection.

:::{figure} assets/laplace_differentiable/occam.png
:label: fig:laplace:occam
:align: center

A visualisation of the Occam's razor effect of the marginal likelihood. The simplest model can assign the largest likelihoods, but just to a small number of datasets. For the most complex model, it is the other way around. For a given dataset $\mathcal{D}$, the marginal likelihood selects an intermediate model that is just complex enough to model these data.
:::

### Occam's razor

+++
Apart from the interpretation as maximum likelihood in a hierarchical Bayesian model, we can also motivate from first principles why the marginal likelihood as a model selection criterion leads to an automatic *Occam’s razor* effect. The principle of Occam’s razor states that among a set of hypotheses (or models) that explain the same data equally well, one should always prefer the simplest one {cite:p}`sober2015ockham`. The connection to the marginal likelihood has already been made in David MacKay’s popular textbook {cite:p}`mackay2003information`, based on his thesis {cite:p}`mackay1992thesis`, and has been further developed by {cite:t}`rasmussen2000occam`.

The argument is relatively simple: since the term {math}`p(\mathcal{D} \mid \model)` has to be a normalised distribution over all possible datasets {math}`\mathcal{D}` for any given model {math}`\model`, this means that models can either assign a lot of likelihood to a small number of datasets or a small amount of likelihood to many datasets. The former would intuitively be a *simple* model, such as a linear one, whereas the latter would be a *complex* model, such as a large neural network. For a dataset that can be modelled well by a linear function, the linear model will therefore be able to assign a larger likelihood than the neural network, because its likelihood is not distributed over all the non-linear datasets. Generally, this means that the model with the largest marginal likelihood for a given dataset will typically be a model that is just complex enough to model this dataset, but not more complex, thus implementing Occam’s razor. This intuition is visualised in [](#fig:laplace:occam).

### Cross-validation

+++
A popular non-Bayesian model selection criterion is cross-validation. The idea is that you split your dataset into {math}`\mathcal{D} = \mathcal{D}_{\mathrm{train}} \cup \mathcal{D}_{\mathrm{val}}`, with disjoint training and validation sets {math}`\mathcal{D}_{\mathrm{train}}` and {math}`\mathcal{D}_{\mathrm{val}}`. We can then estimate the generalisation risk (i.e., the expected loss on unseen data) by training a model on {math}`\mathcal{D}_{\mathrm{train}}` and then evaluating its error on {math}`\mathcal{D}_{\mathrm{val}}`. This fits the intuition that we are ultimately interested in applying our model to unseen test data {math}`\mathcal{D}_{\mathrm{test}}`, so we want to know how well the model predicts on datapoints it has not seen during training.

In their seminal paper, {cite:t}`fong2020marginal` show that the log marginal likelihood on {math}`\mathcal{D}` is equal to the average validation log likelihood over all possible cross-validation splits, that is,

```{math}
\log p(\mathcal{D}) = \mathbb{E}_{\mathcal{D}_{\mathrm{train}}, \mathcal{D}_{\mathrm{val}} \sim \mathrm{split}(\mathcal{D})}\left[\log \mathbb{E}_{\theta \sim p(\theta \mid \mathcal{D}_{\mathrm{train}})}\left[p(\mathcal{D}_{\mathrm{val}} \mid \theta)\right]\right]
```

where {math}`\mathrm{split}(\mathcal{D})` is a shorthand for writing a mechanism that first uniformly samples a dataset size split {math}`m < N` and then samples a uniformly random permutation of assigning {math}`N - m` points to {math}`\mathcal{D}_{\mathrm{train}}` and {math}`m` points to {math}`\mathcal{D}_{\mathrm{val}}`. Note that for notational convenience, we have suppressed the explicit conditioning on {math}`\model`, but you can indeed read all the {math}`p(\cdot)` here as implicitly being {math}`p(\cdot \mid \model)`.

Obviously, actually performing this exhaustive cross-validation procedure would require a computational cost scaling combinatorially with the dataset size {math}`N`, so nobody would actually compute this in practice. However, we see that with the marginal likelihood, we get the result of this exhaustive procedure by solving just a single integral over the training dataset. This is another argument for the marginal likelihood as a model selection criterion: if you believe that cross-validation is a good way to evaluate the potential generalisation capabilities of your model, the marginal likelihood will not only compute one cross-validation for you, but *all* of them at the same time!

### PAC-Bayes

+++
Another way to study the generalisation behaviour of Bayesian predictors is to use the framework of PAC-Bayes bounds {cite:p}`shawe-taylor1997pac,mcallester1998pacbayesian`. While we will get a more thorough introduction to PAC-Bayes bounds in later chapters (specifically in Chapter [](#chap:ensemble_methods_pac_bayesian_ensembles)), for now it suffices to say that they are mathematical statements that bound the expected true risk of our (approximate) posterior—which we cannot know because we do not have access to the true data-generating process—with the expected empirical risk that we can actually measure. These bounds generically look something like this:

:::{prf:proposition} Generic PAC-Bayes bound [@alquier2024user]

+++
For any (approximate) posterior {math}`q(\theta)`, choice of prior {math}`p(\theta)`, and choice of {math}`\delta \in (0,1)`, the expected true risk is bounded with probability at least {math}`1 - \delta` over sampling of datasets {math}`\mathcal{D}` of size {math}`N` from the data-generating process as

```{math}
\mathbb{E}_{\theta \sim q(\theta)}\left[\mathcal{L}(\theta)\right] \leq \mathbb{E}_{\theta \sim q(\theta)}\left[\emprisk{\theta}{\mathcal{D}}\right] + \mathcal{C}(q(\theta), p(\theta), N, \delta)
```
:::

+++
Here, {math}`\mathcal{C}(q(\theta), p(\theta), N, \delta)` is a term that measures the model complexity. In a simple case, this could, for instance, be {cite:p}`mcallester1998pacbayesian,perez2021tighter`

```{math}
\mathcal{C}(q(\theta), p(\theta), N, \delta) = \sqrt{\frac{D_{KL}\left(q(\theta)\middle\|p(\theta)\right) + \log \frac{2 \sqrt{N}}{\delta}}{2 N}}
```

This bound fits several key intuitions: it gets tighter as the size of the dataset grows because the model can overfit less; it gets looser if we reduce {math}`\delta` because then the bound needs to hold with higher probability; and it gets looser as the KL-divergence between posterior and prior grows because then the model can overfit more.

In summary, to get a small bound on the expected risk, you need a posterior that predicts well in terms of empirical risk and at the same time is as simple as possible in terms of the complexity measure {math}`\mathcal{C}`. If this again sounds like Occam’s razor to you, you will not be surprised to learn that PAC-Bayes bounds can be formally connected to the marginal likelihood (indeed, this section would probably not be part of this chapter otherwise).

If we take the posterior in the bound to be the true Bayes posterior, {cite:t}`germain2016pac` derive the following PAC-Bayes bound. Note that the true posterior is often intractable and can only be approximated, in which case the bound will only hold approximately. Note also that it only holds under the technical condition that the log likelihood is sub-Gamma; a sub-Gamma loss is one whose tails are upper-bounded by the tails of a Gamma distribution, essentially ensuring that the loss does not grow unexpectedly fast. This holds, for instance, for Bayesian linear regression with a Gaussian likelihood.

:::{prf:proposition} Marginal likelihood PAC-Bayes bound [@germain2016pac]

+++
For any choice of prior {math}`p(\theta)`, and choice of {math}`\delta \in (0,1)`, and if the negative log likelihood loss {math}`\ell(\theta, \mathcal{D}) = - \log p(\mathcal{D} \mid \theta)` is sub-Gamma with variance factor {math}`s^2` and scale {math}`c < 1`, the expected true risk under the Bayes posterior {math}`p(\theta \mid \mathcal{D})` is bounded with probability at least {math}`1 - \delta` over sampling of datasets {math}`\mathcal{D}` of size {math}`N` from the data-generating process as

```{math}
\mathbb{E}_{\theta \sim p(\theta \mid \mathcal{D})}\left[\mathcal{L}(\theta)\right] \leq \frac{s^2}{2 (1-c)} - \frac{1}{N} \log \left( \delta \, p(\mathcal{D}) \right)
```
:::

+++
We therefore see that maximising the marginal likelihood directly minimises this bound on the generalisation error, providing another strong (frequentist) argument for why it is a reasonable criterion for model selection. Remember, however, as a caveat, that this result only strictly holds for sub-Gamma likelihoods, such as for linearised Laplace in the regression setting, but not in the case of full Laplace inference in BNNs, especially in the classification setting. For those cases, the bound does not provide a proper guarantee, but still an intuitive argument for why the Occam’s razor effect of the marginal likelihood can improve generalisation.

(sec:laplace:diff_mle)=
## Differentiable marginal likelihood with Laplace

+++
So now that we have convinced ourselves that the marginal likelihood is a great criterion for model selection, we are left with the problem of actually computing (or approximating) it. Indeed, this is one of the main challenges of Bayesian inference, which is why, in practice, the marginal likelihood is not often used for this purpose. However, as promised above, the Laplace approximation has the lucky property that we can actually compute the marginal likelihood in closed form!

To understand how we can do that, we will need to look under the hood of the Laplace approximation in Equation [](#eq:laplace:taylor-approximation). Let us remind ourselves that we are aiming to approximate the posterior

```{math}
p(\theta \mid \mathcal{D}) = \frac{p(\theta) \, p(\mathcal{D} \mid \theta)}{p(\mathcal{D})} = \frac{1}{p(\mathcal{D})} \, \tilde{p}(\theta)
```

where we have now written the unnormalised posterior as {math}`\tilde{p}(\theta)`. In the context of model selection, feel free to mentally condition all of these distributions on {math}`\model` as in Equation [](#eq:laplace:model_cond_post) above. We will omit this conditioning here to not needlessly clutter the notation. Note also that we can choose the regularised risk from Equation [](#eq:laplace:energy) to be {math}`\regrisk(\theta) = - \log \tilde{p}(\theta)`.

If we approximate {math}`\log \tilde{p}(\theta)` around a point {math}`\theta_*`, this gives

```{math}
\log \tilde{p}(\theta) \approx \log \tilde{p}(\theta_*) + \nabla_{\theta_*} \log \tilde{p}(\theta_*)^\top (\theta-\theta_*) - \frac{1}{2}(\theta-\theta_*)^\top \Psi(\theta_*) (\theta-\theta_*) + \mathcal{O}(\theta^3)
```

where {math}`\Psi(\theta)` is the same Hessian operator as in Equation [](#eq:laplace:hessian). Note that {math}`\Psi` is the Hessian of {math}`- \log \tilde{p}(\theta)`, hence the sign change in the Taylor expansion.

If we now choose {math}`\theta_*` to be a maximiser of the (log) posterior, such that the first-order term vanishes because {math}`\nabla_\theta \log \tilde{p}(\theta) = 0`, and ignore the third and higher order terms, we can approximate {math}`\tilde{p}(\theta)` with a distribution {math}`q(\theta)`, which we define as

```{math}
\begin{align}
    q(\theta) &= \frac{1}{Z_q} \, \exp \brackets{\log \tilde{p}(\theta_*) - \frac{1}{2}(\theta-\theta_*)^\top \Psi(\theta_*) (\theta-\theta_*)} \\
    &= \frac{\tilde{p}(\theta_*)}{Z_q} \, \exp \brackets{ - \frac{1}{2}(\theta-\theta_*)^\top \Psi(\theta_*) (\theta-\theta_*)}
\end{align}
```

Now, instead of computing the true marginal likelihood {math}`p(\mathcal{D}) = \int \tilde{p}(\theta) \mathrm{d} \theta`, we can just approximate it with the normaliser of our approximate distribution, that is {math}`p(\mathcal{D}) \approx q(\mathcal{D}) = Z_q`.

Since we see from the quadratic form in the exponential that {math}`q(\theta)` is multivariate Gaussian with mean {math}`\theta_*` and precision matrix {math}`\Psi(\theta_*)`, we can look up the normalising constant, which is {math}`\sqrt{(2\pi)^P \det \Psi(\theta_*)^{-1}}`, where {math}`P` is the dimensionality of the parameter space. This means that {math}`\tilde{p}(\theta_*) / Z_q = (2\pi)^{- P / 2} \det \Psi(\theta_*)^{1/2}`. Therefore, our approximate log marginal likelihood is

```{math}
:label: eq:laplace:log-marglik

\begin{align}
    \log Z_q = \log q(\mathcal{D}) &= \log \tilde{p}(\theta_*) - \frac{1}{2} \log \det \Psi(\theta_*) + \frac{P}{2} \log 2 \pi \\
    &= \log p(\theta_*) + \log p(\mathcal{D} \mid \theta_*) - \frac{1}{2} \log \det \Psi(\theta_*) + \text{const}

\end{align}
```

Note that in the last line, we assumed that the number of parameters {math}`P` is the same across the different models {math}`\model` that are being compared. If the model selection is meant to be performed across models with different numbers of parameters, this term naturally is not constant and still needs to be considered in the marginal likelihood.

### Computational considerations

+++
Similar to the inversion {math}`\Psi^{-1}` discussed in previous chapters, the determinant {math}`\det \Psi` also has a computational cost of {math}`\mathcal{O}(P^3)` if the matrix is of size {math}`P \times P`. As discussed before, this can get very expensive when the dimensionality of the parameter space is large, that is, for large neural network models. However, luckily, we can use some computational tricks and approximations to reduce the computational cost, such as linearisation and the empirical Fisher information, as well as diagonal or low-rank approximations (see Chapter [](#cha:laplace:linearized) and Chapter [](#chap:scalability_scalable_laplace) for details). Since none of these considerations are unique to the marginal likelihood computation, we will not discuss them at length in this chapter.

### Post-hoc vs. online model selection

+++
When selecting models using the marginal likelihood, the most naïve way is to perform *post-hoc* model selection. For instance, we can train two models, {math}`\model_0` and {math}`\model_1`, and then compute their respective (log) marginal likelihoods. For illustration purposes, let us say that {math}`\model_0` is a fully-connected neural network and {math}`\model_1` is a convolutional neural network. We train them both on some image dataset (e.g., MNIST) and then compute {math}`\log q(\mathcal{D} \mid \model_0) - \log q(\mathcal{D} \mid \model_1)`. If this quantity is positive, we select {math}`\model_0`, if it is negative, we select {math}`\model_1`. Note that this difference between log marginal likelihoods is equivalent to the likelihood ratio test, which, according to the famous Neyman-Pearson lemma {cite:p}`neyman1933problem`, is the uniformly most powerful hypothesis test.

This selection approach gives us all the benefits of the marginal likelihood described in the previous section, but it still requires that we train each of the different models we want to test. Indeed, for selecting some parts of the model, we can do much better than that! While the neural network architecture is a discrete choice that we have to test post hoc, there are hyperparameters of the neural network with respect to which we can actually compute gradients of the marginal likelihood and then optimise them *online*. These include, for instance, the variance(s) of the prior or the likelihood noise. We will denote as {math}`\model^\partial` the subset of parameters for which gradients {math}`\nabla_{\model^\partial} \log q(\mathcal{D} \mid \model)` exist. We can then optimise these parameters by performing simple gradient ascent like

```{math}
\model^\partial \leftarrow \model^\partial + \gamma \nabla_{\model^\partial} \log q(\mathcal{D} \mid \model)
```

where {math}`\gamma` is a step size. This allows us to select all these model hyperparameters within a single training run, without the need to train several models, as would be the case for approaches such as cross-validation.

However, this approach comes with a few caveats: First, recall that the marginal likelihood depends on the Hessian {math}`\Psi(\theta_*)`, so on the second derivatives of the loss. If we want to take gradients of the marginal likelihood, we need to ensure that these are differentiable with respect to the hyperparameters, which is not true for all activation functions. Particularly, it is not true for the ReLU activation function, so we need to ensure to use smoother ones if we want to perform online model selection (e.g., sigmoid, tanh, Swish, GeLU, ELU, etc.). Second, since the marginal likelihood depends on the entire dataset {math}`\mathcal{D}`, computing it requires an entire pass over the dataset, so in deep-learning parlance, one epoch. Since the best parameters {math}`\theta*` generally depend on the choice of {math}`\model^\partial`, we typically interleave the training, that is, we train {math}`\theta` for one or more epochs and then make one or several gradient steps on the marginal likelihood as written above, each of which corresponds to an additional epoch on the data. Depending on the frequency of these updates, the training will therefore require more epochs; in practice, typically around twice as many. Finally, dropping the first-order term in the Taylor expansion above required {math}`\theta_*` to be a true optimum of the loss, where the gradients vanish, but one might wonder whether standard optimisers actually find such optima in deep learning. Indeed, in practice, the optima we find do not truly have zero gradients everywhere. However, it has been shown empirically that this generally does not impair the model selection with the marginal likelihood {cite:p}`immer2021scalable` and there are even some theoretical arguments for why this might be the case {cite:p}`lin2023online`.

:::{figure}
:label: fig:laplace:toy-model-selection

```{image} assets/laplace_differentiable/marglik-opt.png
:width: 45%
```

```{image} assets/laplace_differentiable/marglik-pred.png
:width: 45%
```

Using the Laplace-approximated marginal likelihood for online model selection of the prior variances on a small Bayesian neural network on a 2D classification task. We see that the marginal likelihood converges to a local optimum over the course of training and that the optimised model is better regularised than a naïvely trained (overfitting) model. {cite:p}`{Adapted from}immer2021scalable`
:::

### Simple example

+++
As a simple illustrative example of online model selection with the marginal likelihood, we will consider a small fully-connected neural network for a synthetic 2D classification task. We place a Gaussian prior over the weights and we assume that the prior variance is shared within each layer. That is, the {math}`i`th weight in the {math}`\ell`th layer will have the prior {math}`p(\theta_{\ell i}) = \mathcal{N}(0, \sigma^2_\ell)`. Our hyperparameters to select are now these layerwise variances, so {math}`\model^\partial = \{\sigma^2_\ell\}_{\ell = 1}^L`.

We see in [](#fig:laplace:toy-model-selection) that our estimate of the marginal likelihood increases monotonically over the course of training and seems to converge to a local optimum. We also see that the prior variances indeed change during training and also seem to converge. The final model achieves a lower training accuracy compared to a naïvely trained model, but a higher test accuracy. This indicates that our selected model is better regularised and overfits less, in line with the Occam’s razor effect of the marginal likelihood. The visual smoothness of its decision boundary supports this conclusion.

### On the risk of overfitting

+++
Based on the previous example and our arguments above, you may now be convinced that the Laplace marginal likelihood will always select the best model for you, no matter the circumstances. Unfortunately, you would be wrong about that. We do not want to hide the caveat that maximum marginal likelihood (ML-II)—much like standard maximum likelihood (ML-I)—carries the inherent risk of overfitting. In our previous examples, such as choosing between two model architectures or optimising layerwise prior variances, this has not been apparent because the number of models compared (or number of hyperparameters optimised) was small compared to both the number of parameters of the model and the number of datapoints. However, if we were to optimise a number of hyperparameters (or compare a number of models) larger than {math}`P` and {math}`N`, we may well overfit the model to the data.

In such cases, just like in standard maximum likelihood, one remedy is to make the hyperprior {math}`p(\model)` in Equation [](#eq:laplace:hyperposterior) more informative and thus regularise the optimisation. We could, for instance, have a separate prior for each weight {math}`p(\theta_{\ell i}) = \mathcal{N}(\mu_{\ell i}, \sigma^2_{\ell i})` that we optimise with the marginal likelihood, but then have strong hyperpriors {math}`p(\mu)` and {math}`p(\sigma^2)` that regularise these two parameters to some reasonable values unless the marginal likelihood really provides strong evidence for different ones. For a more in-depth discussion on the risks of overfitting with the marginal likelihood in model selection and the use of hyperpriors, see {cite:t}`ober21a`.

Another remedy, proposed by {cite:t}`lotfi2022bayesian`, is to compute the marginal likelihood not on the whole dataset, but only on a subset, similar to cross-validation. The authors call this the *conditional log marginal likelihood* (CLML). While this can in practice indeed reduce the overfitting effects, it inherits a lot of the computational challenges of standard cross-validation, most importantly the need to choose a validation split or to perform the procedure several times across different splits.

### Connection to flat minima

+++
A lot of the arguments we made above in [Section {name}](#sec:laplace:marginal_likelihood) hold for the true marginal likelihood {math}`p(\mathcal{D})`. Based on some of the arguments in Chapter [](#cha:laplace:introduction), you might wonder whether our approximation {math}`q(\mathcal{D})` is not too crude for model selection purposes. If the true BNN posterior has exponentially many modes, each of which has potentially a non-Gaussian geometry, should we really believe that approximating the marginal likelihood with a single Gaussian on one mode is enough to judge the quality of the model? Indeed, the proof of the metaphorical pudding is in the eating, and we will see in the following section that the Laplace marginal likelihood can effectively be used for many model-selection use cases in practice. However, before we get to the empirics, there is one more conceptual argument we can offer to hopefully appease the sceptical reader.

If we have another look at our Laplace marginal likelihood criterion in Equation [](#eq:laplace:log-marglik), we see that it has three parts: the prior, the likelihood, and the log determinant of the Hessian. That means that models will achieve a large marginal likelihood estimate if they are probable under the prior, fit the data well (in terms of the likelihood), and have a small determinant of the Hessian. The latter of these is clearly the most cryptic. What does a small determinant mean? Recall that we assume that we are at a local optimum, or have at least linearised our network or added a sufficiently strong prior, such that the Hessian is positive definite. That means that all its eigenvalues are positive. Since the determinant is the product of the eigenvalues, a small determinant means that the eigenvalues of the Hessian are small, which in turn means that the loss function (our negative log posterior) has low curvature in the region around {math}`\theta_*`—it is relatively flat.

:::{figure} assets/laplace_differentiable/flat-minima.png
:label: fig:laplace:flat-minima
:align: center

Here we show a loss landscape with a flat and a narrow minimum. As the loss landscape shifts from $\mathcal{D}_{\text{train}}$ to $\mathcal{D}_{\text{test}}$, the solution $\theta_*^{\mathrm{flat}}$ still achieves a good performance, whereas the solution $\theta_*^{\mathrm{sharp}}$ does not.
:::

+++
The idea that such flat minima can lead to better generalisation has been around for a while {cite:p}`hochreiter1997flat`. In fact, it is quite intuitive, since it essentially means that small changes in {math}`\theta_*` do not lead to large changes in the loss, so even if we are a bit off from the true optimum and even if the loss landscape shifts a bit due to distribution shift in the test data, we will probably still achieve a relatively low test error. This intuition is also sketched in [](#fig:laplace:flat-minima). If this combination of train error (likelihood) and complexity measure (flatness) sounds like a PAC-Bayes bound to you, you are on the right track: flat minima have indeed been connected to PAC-Bayesian results {cite:p}`haddouche2025pac`. Moreover, the deep learning community has embraced flatness-encouraging training criteria—similar to our marginal likelihood—for its purposes, such as the famous sharpness-aware minimisation [SAM, @foret2021sharpnessaware] or its Bayesian counterpart [bSAM, @moellenhoff2023sam].

## Use cases

+++
As we have seen above, the Laplace marginal likelihood can be effectively used to select differentiable hyperparameters online, such as the prior variances or the likelihood noise, and to compare non-differentiable design choices post hoc, such as neural network architectures. These capabilities have been discovered by {cite:t}`immer2021scalable` for modern deep learning models, and they have shown that this type of model selection is often cheaper and more effective than cross-validation. However, this type of hyperparameter tuning is by far not the only way the marginal likelihood can be used for model selection in deep learning. In recent years, different use cases have been developed, some of which have the potential to unlock entirely new application areas. While we cannot possibly provide a comprehensive overview of all of them, we will introduce a selection of the most interesting and creative ones in this section.

### Feature selection

+++
A common use case for model selection in Gaussian process models is feature selection via the automatic relevance determination (ARD) kernel {cite:p}`welch1992screening,Rasmussen2006Gaussian` (see also Chapter [](#chap:kernel_methods_intro)). This is a kernel in which each feature dimension has its own associated kernel length scale. These length scales can then be tuned using the marginal likelihood. One can perform a similar kind of feature selection in BNNs with the Laplace approximation, by performing model selection on the input weights. This is naturally more reasonable for data where the different input features have distinct semantic meaning, such as tabular data, than where they do not, such as images. One can even extend this idea to neural additive models (NAMs) {cite:p}`agarwal2021neural`—that is, models where each input feature is passed to a separate neural network module—by performing model selection on the entire modules. {cite:t}`bouchiat2024improving` have shown that the Laplace marginal likelihood provides an effective way to select features in Bayesian NAMs and even further improve their interpretability through its Occam’s razor effect.

### Neural network pruning

+++
The aforementioned feature selection, in its simplest implementation, amounts to removing weights from the input layer. This suggests the idea that one might also be able to remove weights from deeper layers. Indeed, this is the basic premise of the research field of neural network pruning {cite:p}`cheng2024survey`. There are two major approaches to pruning: unstructured pruning, where weights across the whole weight matrix are removed, and structured pruning, where entire rows or columns of the weight matrix are removed (this corresponds to removing neurons from the layer). Note that while the latter approach is more aggressive, it is the only one that immediately leads to computational gains, since the former approach just makes the matrix more sparse, but not smaller. {cite:t}`dhahri2024shaving` have shown that the Laplace marginal likelihood can be used to improve pruning in both settings, through its Occam’s razor effect, which encourages the neural networks to only use just as many weights as are needed for a given task. They have also shown that the Laplace posterior offers a simple yet effective pruning criterion: the log posterior density evaluated at 0.

### Invariance learning

+++
While the previous two use cases are still intuitively connected to our original model selection examples, as well as to existing applications in other models such as Gaussian processes, {cite:t}`immer2022invariance` propose to use the marginal likelihood for a more creative task: learning invariances from data. While invariances are commonplace in the context of designing invariant neural network architectures by hand (e.g., convolutional neural networks), there are many applications where invariances may be present in the data but the user does not know them. Following the approach from {cite:t}`vanderwilk2018learning`, {cite:t}`immer2022invariance` suggest using the following parameterisation of a generic invariant function:

```{math}
\hat{f}(x, \theta, \eta) = \mathbb{E}_{x' \sim p(x' \mid x, \eta)}\left[f(x', \theta)\right]
```

Here, {math}`f(\cdot)` is the neural network, {math}`\hat{f}(\cdot)` is the desired invariant function, {math}`p(x' \mid x, \eta)` is a data augmentation distribution, with respect to which the function should be invariant, and {math}`\eta` are hyperparameters controlling this augmentation distribution. The Laplace marginal likelihood can now be used to select {math}`\eta` online during training and therefore enables learning a function that discovers the invariances directly from the data within a single training run. {cite:t}`immer2022invariance` show that this can faithfully recover known invariances and can drastically improve data efficiency on real datasets.

### Probing

+++
In addition to the bona fide model selection use cases introduced above, finally, the marginal likelihood can even be used for applications that—on the surface—do not look like model selection, such as probing. Probing is a common interpretability tool in language models, and it tries to answer the question of whether the latent representations of these models encode certain properties of their inputs. For instance, people might wonder “Does BERT encode the concept of verb tense?”. To answer this, there is a long line of work trying to take these latent encodings and fitting different simple readout functions to them, such as linear functions, to predict the property of interest, in this case, verb tense {cite:p}`belinkov2022probing`. But what if the encodings do not encode the property linearly but quadratically? Then the linear probe might not perform well, but arguably, the property is still encoded. However, a quadratic model is more complex than a linear one, so maybe one should count this property as less well encoded than in a set of encodings where a linear probe would suffice. That again sounds suspiciously like Occam’s razor, and indeed {cite:t}`immer2022probing` have shown that one can treat the problem of *jointly* selecting encodings and probes to predict the desired property as a model selection problem and use the Laplace marginal likelihood for a fair apples-to-apples comparison between different models. This shows that the tools presented in this chapter are much more widely applicable than one might naïvely expect, when one is a bit more creative in the interpretation of what *model selection* might look like.