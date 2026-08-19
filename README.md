# Diffusion Posterior Sampling under speckle noise

An extension of [Diffusion Posterior Sampling](https://github.com/DPS2022/diffusion-posterior-sampling)
(Chung et al., ICLR 2023) to a **coherent imaging** forward model, where the dominant corruption is
multiplicative rather than additive.

DPS solves inverse problems with a pre-trained diffusion model by adding a likelihood gradient to
each reverse step, and the released implementation covers Gaussian and Poisson noise. Coherent
imaging (SAR, ultrasound, optical coherence tomography) fits neither: there the corruption is
**speckle**, a multiplicative random field. This repository is the work of adapting the method to
that case.

> **Status: work in progress.** The forward model, the measurement operator and the
> likelihood-gradient estimator are implemented. No reconstruction results are checked in yet, and
> the section [What is and is not wired up](#what-is-and-is-not-wired-up) says exactly where things
> stand.

## Forward model

```
y = A(x . w) + z
```

- `x` is the image to recover.
- `w ~ N(mu, sigma_w^2)` is the multiplicative **speckle** field, applied pixelwise.
- `A` is a **DCT low-pass operator**: take the 2D DCT, keep the low-frequency square set by
  `mask_rate`, invert. This stands in for the band limit of a coherent imaging system.
- `z ~ N(0, sigma_z^2)` is additive sensor noise.

At the configured operating point (`mask_rate = 0.8`, `sigma_w = 1.0`, `sigma_z = 0.025`) the
operator keeps a square of side `H * sqrt(0.8)`, so the problem is band limited *and*
multiplicatively corrupted at the same time.

## What this adds on top of upstream DPS

| Piece | File | What it is |
|---|---|---|
| `DCTOperator` | `measurements.py` | DCT low-pass measurement operator with a configurable mask rate |
| `SpeckleNoise` | `measurements.py` | the noise model, registered under the name `speckle` |
| `nll_grad` | `measurements.py` | Monte Carlo estimator of the speckle negative log-likelihood gradient |
| `conjugate_gradient`, `B_operator` | `measurements.py` | the matrix-free linear solve that estimator needs |
| speckle branch | `condition_methods.py` | routing in `grad_and_value` for the speckle case |
| speckle measurement path | `sample_condition.py` | applies the multiplicative field before the operator |
| `dct_speckle_config.yaml` | config | the operating point |

Everything else is upstream DPS, which is itself built on OpenAI's guided-diffusion. The blocks
added here are marked with banner comments in the source.

## The likelihood gradient

Under speckle the measurement likelihood is not Gaussian in `x`, so the plain DPS residual
gradient is not the right object. Writing

```
B(x) = A^T diag(x^2) A + sigma_z^2 I
```

the gradient of the negative log-likelihood involves `B(x)^{-1}`, which is never formed
explicitly. `nll_grad` estimates it with a Hutchinson-style Monte Carlo trace term, averaging
`A(B^{-1} A v) . v` over `num_ite_MC` Gaussian probes `v`, plus a data term applying the same
solve to the residual. Each application of `B^{-1}` is a matrix-free conjugate gradient solve
(25 iterations, tolerance 1e-6).

## What is and is not wired up

Being explicit, because it changes what a run from this configuration actually measures.

- The Monte Carlo gradient call in `condition_methods.py` is **implemented but commented out**.
  The speckle branch currently falls back to the plain DPS residual gradient, the gradient of
  `||y - A(x_0_hat)||`. A run therefore measures DPS against a speckle-corrupted measurement, not
  DPS with a speckle likelihood.
- `nll_grad` carries an open question in a comment: it needs `x_0_hat`, which depends on the
  timestep and the score, while the current call site passes `x`.
- No quantitative reconstruction results (PSNR, SSIM) are recorded here.

## Running it

These files are drop-in replacements inside the DPS repository, not a standalone package.

1. Clone [diffusion-posterior-sampling](https://github.com/DPS2022/diffusion-posterior-sampling)
   and follow its environment setup.
2. Download the pre-trained 256x256 FFHQ checkpoint `ffhq_10m.pt` from the DPS release links and
   put it in `models/`.
3. Copy the files from here into place:
   - `measurements.py` and `condition_methods.py` into `guided_diffusion/`
   - `sample_condition.py` at the repository root
   - `dct_speckle_config.yaml` into `configs/`
4. `pip install torch-dct`
5. Run:

```bash
python sample_condition.py \
  --model_config=configs/model_config.yaml \
  --diffusion_config=configs/diffusion_config.yaml \
  --task_config=configs/dct_speckle_config.yaml \
  --gpu=0 \
  --save_dir=./results
```

## References

- Chung, Kim, McCann, Klasky, Ye (2023). *Diffusion Posterior Sampling for General Noisy Inverse
  Problems.* ICLR 2023.
- Dhariwal, Nichol (2021). *Diffusion Models Beat GANs on Image Synthesis* (guided-diffusion).
