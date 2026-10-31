# Model and source licence audit

Audited 2026-10-31. This backend uses **TinyLuma-UNet-v1**, an original small
additive-skip UNet trained from scratch. Its source and released weights are
MIT. There are no imported pretrained tensors, Apple models, Apple-compiled
programs, calibration dumps or camera captures in the training pipeline.

## Candidate decision

[FastDVDnet at c8fdf618](https://github.com/m-tassano/fastdvdnet/tree/c8fdf6182a0340e89dd18f5df25b47337cbede6f)
has an MIT root licence; its README describes training on DAVIS. This audit
does not establish a separate
licence/provenance chain for each downloadable pretrained checkpoint. No code
or weights from it are used. Training a smaller original UNet removes that
checkpoint ambiguity and reduces the operator set to five convolutions,
ReLU, nearest upsampling, addition, subtraction and clipping.

## Data and splits

Only these four images are loaded from scikit-image **0.25.2**. The
[versioned data documentation](https://scikit-image.org/docs/0.25.x/api/skimage.data.html)
and [tagged fetchers](https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/data/_fetchers.py)
identify their licences and attribution.

| Split | Image | Creator / source | Rights |
| --- | --- | --- | --- |
| Train | astronaut | NASA, Eileen Collins portrait | Public domain |
| Train | coffee | Rachel Michetti, Pikolo Espresso Bar | CC0 |
| Held-out test | chelsea | Stefan van der Walt | CC0 |
| Held-out test | hubble_deep_field | NASA, Hubble | Public domain |

`training.json` and the measurement receipt identify decoded RGB bytes by
SHA-256, library version, split, seed and noise process. The training script
never loads the held-out images. Data are fetched by the pinned package, not
redistributed here. No endorsement by these creators is implied.

`open-luma-awgn-v1` is a reproducible synthetic noisy dataset derived from the
held-out pair: full-range BT.709 luma, clipped Gaussian noise with sigma
15/25/40 on the uint8 scale, followed by uint8 quantization. It is a small
quality smoke check, not a real-sensor benchmark. Noise seeds and clean-image
identities are in the receipt. There is no claim of camera or temporal quality.

## Code and build dependencies

| Component | Licence | Role |
| --- | --- | --- |
| New backend, scripts, weights | MIT | Original implementation and training output |
| Existing libane and binding | MIT | Execution and fallback policy |
| mil-hwx-compiler | MIT | Open source program emission |
| coremltools 9.0 | BSD-3-Clause | Typed MIL construction and source package export |
| PyTorch 2.7.1 CPU | BSD-3-Clause | Training only |
| ONNX 1.19.1 | Apache-2.0 | CPU graph serialization |
| ONNX Runtime 1.23.2 | MIT | Model CPU evaluation |
| NumPy 2.2.6, scikit-image 0.25.2 | BSD-3-Clause | Array math and open test data |
| CPython | PSF-2.0 | C ABI bridge interpreter |

These libraries are dependencies, not vendored source; their distributions
retain their own notices. Installed wheels may include separately licensed
transitive libraries. This is a model/code/data audit, not a complete distro
SBOM. There is no unclear third-party weight licence left in the selected model.
The captured compiler diagnostics retain their original MIT copyright notice
in `receipts/compiler-LICENSE.txt`.
