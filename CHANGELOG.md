# Changelog

## 1.0.0

Reference release of the code behind Izzatullah et al. (2024), Geophysics 89(2), R137.

The package was reorganised and documented. Numerical results are unchanged: regression tests compare every public function with outputs saved from the earlier code, and both example notebooks give the same results on a GPU.

### Changed

- The kernels, the PyLops-to-PyTorch operator and the posterior now live in `kernels.py`, `operators.py` and `posterior.py`. They are still available from `pnpsvgd.svgd`, so the notebooks' imports keep working.
- `TorchOperator` is a plain class with `apply`, `shape` and `dtype`. It no longer subclasses `pylops.LinearOperator`.
- The first argument of `PnPSVGD` is now called `denoiser` (it was `Denoiser`). Positional calls are unaffected.
- `SVGD` no longer requires a `scheduler` argument.
- The DRUNet helpers run on the device and in the precision of the model's parameters instead of always calling `.cuda().half()`. With a CUDA float16 model, as in the notebooks, they behave exactly as before.
- `basicblock.py` keeps only the layers that DRUNet uses. Weight names are unchanged, so the published `drunet_gray.pth` loads as before.
- `install_env.sh` downloads the DRUNet weights from the KAIR v1.0 release and checks their SHA-256. This file is byte-identical to the copy used for the paper.

### Fixed

- `utils.callback0` referred to an undefined variable.
- `PrimalDual_` failed when a callback was used with `gfirst=False`.
- The Hess notebook relied on `time` leaking out of a star import. It now imports `time` itself.
- The Volve notebook saved its results before computing `minv_t`, so it could not run from top to bottom. The save cell is now the last cell.

### Added

- Docstrings with parameters, equations and references for every public function.
- CPU regression tests, doctests and a GitHub Actions workflow.
- `LICENSE`, `THIRD_PARTY_NOTICES.md` for the adapted KAIR code, and this changelog.
