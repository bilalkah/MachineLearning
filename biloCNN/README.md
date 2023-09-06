# biloCNN

In this project the main goal is to build a multi-tasking CNN based model that feeded with PointCloud2 data and RGB camera image and would generate 6-DoF odometry estimation. 

## Codebase Structure
```
/builder
  /cfg : Yaml files contains confihurastion of model
  layers.py : Layers and blocks
  model.py : Builder class
  optimizers.py : Optimizers
  loss.py : Loss
```

## Jenkins

test test test test test test

## 6-DoF
- Translation 
  - X-axis
  - Y-axis
  - Z-axis
- Rotation
  - X-axis
  - Y-axis
  - Z-axis


# TODO List
- [x] PointCloud2 Compression in fixed size
- [ ] Represent PointCloud2 in polar coordinate system or 2D flatten image
- [ ] Fixing codebase of models and layers

