# mods-rectify
Script that rectifies the MODS dataset and its ground truth annotations.

## Installation
1. Clone the repository
2. Install the requirements: `pip3 install -r requirements.txt`

## Usage

Usage: `python3 mods-rectify.py <config-file>`


## Config file
Config file format: YAML

### Keys:
- `dataset_path`: path to the dataset
- `out_path`: path to the output directory
- `output_imgs`: whether to output the rectified images
- `visualise`: whether to visualise the rectification

### Example:
```yaml
dataset_path: '~/mods/'
out_path: '~/mods_rectified/'
output_imgs: True
visualise: False
```
