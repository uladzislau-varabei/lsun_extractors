# LSUN extractors

A useful script to extract data from LSUN datasets. It allows to extract only a subset of database, 
create wide versions of images, set jpg quality of output images and process data files efficiently to avoid huge RAM usage. 
For more details see file `lsun_scripts.py`.

*Note:* databse images are compressed in jpeg with quality 75.

This repo is based on a script `dataset_tool.py` from StyleGAN project: https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py

## Useful links

- The official LSUN web page where data can be downloaded: https://www.yf.io/p/lsun
- Base repo to download and process LSUN data: https://github.com/fyu/lsun
- PyTorch code to parse LSUN databases: https://pytorch.org/vision/stable/_modules/torchvision/datasets/lsun.html
