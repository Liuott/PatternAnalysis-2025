import os
import pandas as pd

combined = pd.DataFrame([
    {"split": "keras_slices_validate", "file": "__summary__", "psnr": 28.1, "ssim": 0.7919},
    {"split": "keras_slices_test",     "file": "__summary__", "psnr": 28.1, "ssim": 0.8048},
])

# 建议放到 workdir 下
out_dir = "recognition/hipmri-VQVAE-48302304/runs"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "metrics_combined.csv")
combined.to_csv(out_path, index=False, float_format="%.4f")

print("Saved to:", out_path)
print(combined.to_string(index=False))