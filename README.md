# diffusion_remap

| 时间 | 分支名 | 实验内容 | 实验记录 | 实验结论 |
| ------| ------ | ------ | ------ | -----: |
| | master | bs=24&bs=128的对比 | todo | bs=24下遮挡区域，large motion情况下，补全效果不佳；bs=128整体ok |
| 20230811 | master | bs=128(sintel训练)模型在GHOF-Clean上的PSNR对比 | 客观指标：<span style="color:blue">some *identity_psnr: avg:21.8951 - backWarp_psnr: avg:28.9480 - fordWarp_naive_psnr: avg:20.8678 - fordWarp_dm_psnr: avg:26.9163* text</span>.，主观效果: <span style="color:red">some *s3://lhp/diffusion-remap/qualitative_results/20230811-bs128-sintelTrain-GhofTest/* text</span>. | forward-warp + diffusion的主观，客观效果均不如naive backward-warp |
