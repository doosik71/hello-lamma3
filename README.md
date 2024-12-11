## LLAMA 3 Example

## Check version compatibility

- [Tensorflow](https://www.tensorflow.org/install/source?hl=ko#tested_build_configurations)

## Current version

| NVidia | CUDA  | Cuda Toolkit | cuDNN |   PyTorch   | Tensorflow |
| :----: | :---: | :----------: | :---: | :---------: | :--------: |
|  535   | 12.2  |    11.5.1    |  9.6  | 2.5.1+cu124 |   2.18.0   |

## Check version

```bash
ubuntu-drivers devices
nvidia-smi
nvcc --version
# apt show nvidia-cuda-toolkit
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
python --version
python -c "import torch; print(torch.__version__)"
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip check  
```

## Install CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

## Install CUDA Toolkit (OLD)

```bash
# sudo apt update
# sudo apt install nvidia-cuda-toolkit
```

## Install CUDNN

```bash
wget https://developer.download.nvidia.com/compute/cudnn/9.6.0/local_installers/cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.6.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.6.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```

## Install Tensorflow

```bash
python3 -m pip install 'tensorflow[and-cuda]'
```

## Install PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Login

```bash
huggingface-cli login
```

## Run

```bash
python3 example.py
```
