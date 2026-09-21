# Deployment setup

This guide installs the DODGER deployment code for simulation. The G1 onboard
computer uses ROS 2 Foxy and requires the additional instructions in
[ROS 2 Foxy setup](ros2_foxy_setup.md).

## 1. Clone the release

```bash
cd ~
git clone https://github.com/anonymousroboticsauthor/dodger.git
cd ~/dodger/deployment
```

All commands in the deployment documentation assume this directory layout.
Do not copy `build/`, `install/`, or `log/` directories from another checkout;
these contain absolute paths and must be regenerated locally.

## 2. System dependencies

The simulation setup is supported on Ubuntu 22.04 with an NVIDIA GPU and a
recent driver.

```bash
sudo apt update
sudo apt install -y \
    build-essential cmake git \
    libyaml-cpp-dev libboost-all-dev libeigen3-dev \
    libspdlog-dev libfmt-dev
```

## 3. Python environment

Install Miniconda if it is not already available:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
source ~/.bashrc
```

Create the environment and install the Python package:

```bash
conda create -n unitree_rl_mjlab python=3.11
conda activate unitree_rl_mjlab
cd ~/dodger/deployment
pip install -e .
```

The environment name remains `unitree_rl_mjlab` because it is a software
package identifier, independent of the repository directory name.

## 4. ROS 2 perception

Perception, RViz, the goal interface, and `g1_ctrl` must use the system ROS
Python environment rather than Conda.

- [Perception prerequisites and workspace build](../ros2/README.md)
- [G1 onboard computer: Ubuntu 20.04 and ROS 2 Foxy](ros2_foxy_setup.md)
- [Robot-to-operator network and visualization](ros2_teleop.md)

## 5. Build and run

The controller links against messages built in the ROS 2 workspace, so build
the perception workspace before configuring `g1_ctrl`. Complete simulation and
hardware commands are documented in the
[Navigation runtime guide](../deploy/robots/g1/config/policy/navigation/v1/RUN.md).

