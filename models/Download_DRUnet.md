# DRUNet weights

The notebooks load `models/drunet_gray.pth`, the grayscale DRUNet denoiser of Zhang et al. (2022).
`install_env.sh` downloads it. To get it by hand:

```bash
curl -fL -o models/drunet_gray.pth https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth
```

Expected SHA-256: `e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4`.

Zhang, K., Li, Y., Zuo, W., Zhang, L., Van Gool, L., and Timofte, R. (2022). Plug-and-play image
restoration with deep denoiser prior. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44(10).
