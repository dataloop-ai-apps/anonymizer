# Image Anonymization

A Dataloop pipeline node application for anonymizing images by blurring or masking specific objects based on annotations, models, and labels.

## Overview

This app provides a pipeline node to anonymize sensitive content in images while preserving the original annotations. It supports multiple anonymization strategies and can process images based on specific model predictions or annotation labels.

## Features

- **Multiple Anonymization Types**:
  - `replace`: Overwrites the original image with the anonymized version
  - `remove`: Creates an anonymized version and removes the original
  - `keep`: Creates an anonymized version while keeping the original (with linked metadata)

- **Flexible Object Detection**:
  - Filter objects by model IDs
  - Filter objects by annotation labels
  - Supports multiple annotation types:
    - Bounding boxes
    - Polygons
    - Segmentation masks

- **Anonymization Options**:
  - `blur`: Gaussian blur with configurable intensity
  - `fill`: Simple masking by filling the annotation with a mask


## Usage

1. Configure the node inside a pipeline with appropriate parameters
2. The app will process each image by:
   - Identifying objects of interest based on models and/or labels
   - Creating masks for the identified objects
   - Applying blur or masking to the objects
   - Handling the output based on the specified anonymization type
   - Preserving all annotations in the output image
