# DRLFailureMonitor

Research Artifact of the paper "DRLFailureMonitor: A Dynamic Failure Monitoring Approach for Deep Reinforcement Learning System"
" 

## Installation
* Bulid from source code
```bash
git clone https://github.com/CAgent05/DRLFailureMonitor.git
cd DRLFailureMonitor
pip install -r requirements.txt
```

## Steps to run the code
Effectiveness of DRLFailureMonitor
### 1. Training Data collection
```python
python DataCollection.py --dataset BipedalWalkerHC --episodes 3000 --nsteps 20
```
* **dataset:** BipedalWaklkerHC, Hopper, InvertedDoublePendulum, Walker2d, Humanoid.
* **episodes:** Number of episodes to collect the data. The default value is 3000.
* **nsteps:** Number of steps in each episode. When nsteps is 20, we also save state-only time series for comparison with state-action data.

### 2. Training Todynet Model
```python
python Todynet/src/train.py --dataset BipedalWalkerHCAC --nsteps 20 --epochs 100
```
* **dataset:** BipedalWaklkerHCAC, HopperAC, InvertedDoublePendulumAC, Walker2dAC, HumanoidAC. 
* **epochs:** Number of epochs to train the model. The default value is 100.

### 3. Online Monitor
```python
python TodyNet/OnlineMonitor.py --dataset BipedalWalkerHCAC --nsteps 20
```

### 4. Data Analysis
```python
python result/DataAnalysis.py --dataset BipedalWalkerHC --nsteps 20 --alg TodyNet
```
* **alg:** TodyNet, MLSTM-FCN, OS-CNN, WEASEL

## Comparative Study

### MLSTM-FCN
```python
python MLSTM-FCN/train.py --dataset BipedalWalkerHCAC --nsteps 20 --epochs 100
python MLSTM-FCN/OnlineMonitor.py --dataset BipedalWalkerHCAC --nsteps 20
```

### OS-CNN
```python
python OS-CNN/train.py --dataset BipedalWalkerHCAC --nsteps 20 --epochs 100
python OS-CNN/OnlineMonitor.py --dataset BipedalWalkerHCAC --nsteps 20
```

### WEASEL
```python
python WEASEL_MUSE/train.py --dataset BipedalWalkerHCAC --nsteps 20
python WEASEL_MUSE/OnlineMonitor.py --dataset BipedalWalkerHCAC --nsteps 20
```

