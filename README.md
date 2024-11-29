## MAGICal Synthesis: Memory-Efficient Approach for Generative Semiconductor Package Image Construction

[https://doi.org/10.6117/kmeps.2023.30.4.069](https://doi.org/10.6117/kmeps.2023.30.4.069)


Abstract: With the rapid growth of artificial intelligence, the demand for semiconductors is enormously increasing
everywhere. To ensure the manufacturing quality and quantity simultaneously, the importance of automatic defect detection
during the packaging process has been re-visited by adapting various deep learning-based methodologies into automatic
packaging defect inspection. Deep learning (DL) models require a large amount of data for training, but due to the nature
of the semiconductor industry where security is important, sharing and labeling of relevant data is challenging, making
it difficult for model training. In this study, we propose a new framework for securing sufficient data for DL models
with fewer computing resources through a divide-and-conquer approach. The proposed method divides high-resolution
images into pre-defined sub-regions and assigns conditional labels to each region, then trains individual sub-regions and
boundaries with boundary loss inducing the globally coherent and seamless images. Afterwards, full-size image is
reconstructed by combining divided sub-regions. The experimental results show that the images obtained through this
research have high efficiency, consistency, quality, and generality

Keywords: Data Augmentation, Generative Adversarial Networks, Artificial Intelligence, Performance Optimization

## Framework Architecture
> proposed in our research
> 
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

- We added the boundary loss to minimize the differences among the lines of 1 pixels where the sub-regions meet each other, which induce the framework to learn the natural boundary.
- It is similar to `mse loss` like below.


$$
L_B = \frac{\lambda_B}{n} \left( \sum_{i=0}^{n-1} \left( \hat{r}\_{i,\mathrm{bottom}} - \hat{r}\_{i+1,\mathrm{top}} \right)^2 + \sum_{i=0}^{n-1} \left( \hat{c}\_{i,\mathrm{right}} - \hat{c}\_{i+1,\mathrm{left}} \right)^2 \right)
$$


- $\hat{r}_{i, (start|end)}$: the last $n$ column (the right|leftmost) vector of ith of concatenated sub regions vertically
- $\hat{c}_{i, (top|bottom)}$: the last $n$ row (the top|bottommost) vector of ith of concatenated sub regions horizontally
- $n$: the number of sub-regions in a row or a column.
- $\lambda_b$: the weight of $L_{boundary}$, which controls the strength of $L_{boundary}$


   

### Wasserstein GAN

$$ L_W = \mathbb{E}\_{ \mathbf{x} \sim \mathbb{P}\_r}[D( \mathbf{x})] - \mathbb{E}\_{ \mathbf{ \tilde{x}} \sim \mathbb{P}\_g}[D( \mathbf{ \tilde{x}})] $$


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
