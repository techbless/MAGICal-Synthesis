
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

## MAGICal Synthesis: Memory-Efficient Approach for Generative Semiconductor Package Image Construction

[https://doi.org/10.6117/kmeps.2023.30.4.069](https://doi.org/10.6117/kmeps.2023.30.4.069)


## Framework Architecture

### Sub-Region Patch Based Approach

We cropped the full-size images into ($N \times N$) sub-regions with the resolution $(\frac{W}{N} \times \frac{H}{N})$.

### Training Steps

step1. crop the full-size images into sub regions with empirical function $\psi(x, c')$.

step3. feed the random noise vector and location of sub-regions ($c'$) to generator.

step4. feed the real $x$ and $\tilde{x}$ and $c'$ to the critic and train D.

step5. train G to deceive the critic.

step5. generate the sub-regions in window with same duplicated latent vector.

step6. calculate the boundary loss and train G


### Natural Boundary and Coherent Sub-regions

> proposed in our research   



- We added the boundary loss to minimize the differences among the lines of 1 pixels where the sub-regions meet each other, which induce the framework to learn the natural boundary.
- It is similar to `mse loss` like below.

$$
L_{boundary} = \lambda_b \times \left( \frac{1}{n-1} \sum_{i=0}^{n-1} (\hat{r}\_{i, \mathrm{end}} - \hat{r}\_{i+1, \mathrm{start}})^2 + \frac{1}{n-1} \sum_{i=0}^{n-1} (\hat{c}\_{i, \mathrm{end}} - \hat{c}\_{i+1, \mathrm{start}})^2 \right)
$$

$$
\frac{1}{n-1} \sum_{i=0}^{n-1} (\hat{r}\_{i, \text{end}} - \hat{r}\_{i+1, \text{start}})^2 + \frac{1}{n-1} \sum_{i=0}^{n-1} (\hat{c}\_{i, \text{end}} - \hat{c}\_{i+1, \text{start}})^2
$$

$$
L_B = \frac{\lambda_B}{n-1} \left( \sum_{i=0}^{n-1} \left( \hat{r}\_{i,\mathrm{end}} - \hat{r}\_{i+1,\mathrm{start}} \right)^2 + \sum_{i=0}^{n-1} \left( \hat{c}\_{i,\mathrm{end}} - \hat{c}\_{i+1,\mathrm{start}} \right)^2 \right)
$$


- $\hat{r}_{i, (start|end)}$: the last $n$ column (the right|leftmost) vector of ith of concatenated sub regions vertically
- $n$: the number of sub-regions in a row or a column.
- $\lambda_b$: the weight of $L_{boundary}$, which controls the strength of $L_{boundary}$


   

### Wasserstein GAN

$$
L_W = \mathbb{E}_{\mathbf{x} \sim \mathbb{P}\_r}[D(\mathbf{x})] - \mathbb{E}_{\mathbf{\tilde{x}} \sim \mathbb{P}\_g}[D(\mathbf{\tilde{x}})]
$$



$$
L_{GP} = \lambda\_{GP} \mathbb{E}_{\mathbf{\hat{x}} \sim \mathbb{P}\_{\hat{x}}} \left[ \left( \|\nabla\_{\mathbf{\hat{x}}} D(\mathbf{\hat{x}})\|\_2 - 1 \right)^2 \right]
$$



- $\tilde{x}$: Generated (fake) data points produced by the generator
- $\hat{x}$: Interpolated points between real and generated data, used specifically in the gradient penalty term of WGAN-GP

### Auxiliary Classifier

It is used to classify the sub-region’s location.

$$
L_C = \mathbb{E}[\log P(C = c | X_{\text{real}})] + \mathbb{E}[\log P(C = c | X_{\text{fake}})]
$$

### Combined loss function.

$$
L = L_W + L_{GP} + L_C + L_{boundary}
$$

```
@article{Chang2023MAGICalSynthesis,
  author    = {Yunbin Chang and Wonyong Choi and Keejun Han},
  title     = {MAGICal Synthesis: Memory-Efficient Approach for Generative Semiconductor Package Image Construction},
  journal   = {Journal of the Microelectronics and Packaging Society},
  year      = {2023},
  volume    = {30},
  number    = {4},
  pages     = {69--78},
  doi       = {10.6117/kmeps.2023.30.4.069},
  publisher = {The Korean Microelectronics and Packaging Society},
  issn      = {1226-9360, 2287-7525},
}
```
