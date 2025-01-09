# TTA-for-3D-Fetal-Subcortical-Segmentation

![GitHub Logo](/figures/TTA_schematic.png)

Source code for "Exploring Test Time Adaptation for Subcortical Segmentation of the Fetal Brain in 3D Ultrasound". Accepted for publication in IEEE International Symposium on Biomedical Imaging (ISBI) 2025.

## Details

Source code:
* `text_time_adaptation` - Contains the implementation of each test-time adaptation (TTA) method described in the paper (histogram matching, TENT, EntropyKL, LayerInspect).
* `metrics.py` - Contains the metrics used to evaluate the performance of each model (Dice, Surface Dice).
* `model_loaders.py` - Contains code to load model objects for our base model and for each test-time adapted model. 
* `unet_architecture.py` - Architecture for the base UNet model used in our experiments. https://github.com/lindehesse/FetalSubcortSegm_Code.

Model Loader parameters:
* `device` - `torch.device` for the model
* `model_weights_path` - Path to the checkpointed base model weights
* `subject_list` - List of preprocessed volumes to feed into the model 
* `lr` - learning rate
* `steps` - Number of backwards passes of TENT/EntropyKL/LayerInspect before making a final prediction
* `source_data_activations_path` - Path to precomputed source data activations for LayerInspect
* `lambd` - Lambda value for EntropyKL
* `use_KL` - When `True`, use EntropyKL with LayerInspect. Otherwise, use TENT
* `num_to_update` - Number of layers to update in LayerInspect

## Dependencies
```
numpy==1.24.4
scikit_image==0.19.3
SimpleITK==2.4.0
torch==2.1.1
torch==2.1.2
torchio==0.19.2

surface_distance:
# git clone https://github.com/lindehesse/surface-distance.git
# pip install surface-distance/
```
