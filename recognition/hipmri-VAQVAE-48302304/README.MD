














module load Anaconda3 cuda/12.1 # Rangpur 示例
conda create -y -n hipmri_vq python=3.10
conda activate hipmri_vq
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install nibabel numpy scipy scikit-image opencv-python matplotlib tqdm pyyaml
pip install torchmetrics==1.4.1