# Learning Global Object-Centric Representations via Disentangled Slot Attention


This repository is the implementation of Learning Global Object-Centric Representations via Disentangled Slot Attention (GOLD).

### Dependencies

```
torch			1.11.0
torchvision		0.12.0
h5py			3.6.0
numpy			1.21.2
tensorboard		2.6.0       
matplotlib		3.7.3
sklearn			1.3.1
json5			0.9.14
yaml			0.2.5                          
```


### Training

To train the model in the paper, one can run this command:

```
bash scripts/train.sh
```

Check `train_ocl.py` and `train_vqvae.py` to see the full list of training arguments.

To modify the hyperparameters, one can edit the file `scripts/train.sh`.


### Testing

To test the model in the paper, one can run this command:

```
bash scripts/test.sh
```

### Outputs

The training code produces Tensorboard logs. To see these logs, run Tensorboard on the logging directory. These logs contain the training loss curves and visualizations of reconstructions and object attention maps.

### Code Files

This repository provides the following floders and files.

\- `scripts` contains the training and testing script.

\- `vqvae` provides helper classes and functions of VQVAE.

\- `train_ocl.py` contains parameters and training process of the Global Object-Centric Learning module.

\- `ocl.py` provides the model class for the Global Object-Centric Learning module.

\- `train_vqvae.py` contains parameters and training process of the Image Encoder-Eecoder module.

\- `preocl_vqvae.py` provides the model class for the Image Encoder-Eecoder module.

\- `data_img_h5.py` contains the dataset class.

\- `utils.py` provides helper classes and functions.

\- `metrics.py` provides helper classes and metrics calculation functions.

\- `plot_decompose.py` provides helper classes and plot the decompose visual functions.

\- `test_ocl.py` provides qualitative and quantitative test results of GOLD on Scene Decomposition as well as quantitative results of Object Identification.

\- `test_gold.py` provides the visualization results of GOLD on Global Object-Centric Learning, Attributes Disentanglement and the Individual Object Generation.
