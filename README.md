# MoQ-NAS

## Environment Configuration

The following steps are used to configure the environment for the project.

- Miniconda Installation
- Conda Environment Creation
- Package Installation

**Notes**: 
- An NVIDIA GPU is required to run the project. 
- The project is tested on Ubuntu 22.04 LTS with NVIDIA L40S GPU.

### Miniconda Installation

Install Miniconda in the home directory. Refer to the [Miniconda Installation Guide](https://docs.anaconda.com/free/miniconda/#quick-command-line-install) for more information.

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

### Conda Environment Creation

```bash
conda create -n moqnas python=3.10
conda activate moqnas
```

### Package Installation

```bash
#conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

pip install -r requirements.txt
```
