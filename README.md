# NEMG-IQA

## Prerequisites

Ubuntu 18.04

Python 3.6.10

The sufficient requirements of libs in Python could be seen in ./env.list.txt

## Training from scratch

1. Generate Json score files.

   You can generate score files by running prepare_score_file.py in ./DatasetScores

   ```shell
   python prepare_score_file.py
   ```

   The default setting would generate 10 splits per dataset(CSIQ,TID2013,LIVE).

2. Configure your setting

   You should configure your score file and dataset path before training. Please open the ./configuration/{dataset}_config.json and change the image_dir and score_file term

3. Training

   You could train the model by the following command:

   ```shell
   python trainer.py --config configuration/{dataset}_config.json
   ```



## Reference

The Resnet-50 is adapted from Pytorch model zoo 

The implementation of SSIM is from https://github.com/Po-Hsun-Su/pytorch-ssim

​    

