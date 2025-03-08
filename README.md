# Multimodal Land Cover Classification with Edge-Aware ViT and Diffusion Model

## Overview
This project presents a multimodal architecture for enhancing land cover classification using the EuroSAT dataset. It integrates two models:
- **Edge-Aware Vision Transformer (ViT)**: Captures edges in satellite images to enhance feature extraction.
- **Diffusion Model**: Removes noise, specifically cloudy noise, to improve image clarity before classification.

Both models are trained separately and then combined to achieve superior classification performance. The test images used here are randomly snipped from google maps targeting rivers and buildings in India.

## Dataset
We use the **EuroSAT** dataset, a collection of satellite images from Sentinel-2, which contains 10 classes related to land use and land cover. Create a directory for the data and use the appropriate paths in the program. Also do the same for models too as pretrained models are too big to upload.

## Methodology
1. **Preprocessing**:
   - Introduce synthetic cloudy noise into the EuroSAT dataset using Perlin noise.
   - Normalize and augment the dataset.

2. **Diffusion Model**:
   - A U-Net-based model trained to remove cloudy noise from satellite images.
   - Uses MSE loss to reconstruct clear images from noisy inputs.

3. **Edge-Aware ViT**:
   - A ViT-based model trained on edge-enhanced representations of the EuroSAT dataset.
   - Uses a classification head to predict land cover classes from enhanced images.

4. **Fusion of Models**:
   - The Diffusion model enhances satellite images before classification.
   - The Edge-Aware ViT processes the refined images to improve land cover classification.

### Dependencies
Install the required libraries:
```bash
pip install torch torchvision numpy perlin-noise
```
## Results
- Enhanced classification accuracy compared to standalone ViT or Diffusion model.
- Improved robustness against occlusions and noisy satellite images.

## Future Work
- Explore alternative noise removal techniques.
- Optimize the fusion strategy for better performance.
- Apply the approach to diverse remote sensing datasets.
- The model here was trained on smaller subsets of the EuroSAT dataset due to computational expenses and might provide better performance if trained on a larger dataset.

## Author
Prithviraajan Senthilkumar.

Licensed under MIT
