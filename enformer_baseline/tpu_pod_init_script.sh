#!/bin/bash -l

sudo pip3 install silence_tensorflow
sudo pip3 install tensorflow-addons
sudo pip3 install matplotlib==3.4.3
sudo pip3 install pandas
sudo pip3 install dm-sonnet==2.0.0
sudo pip3 install tensorflow-probability
sudo pip3 install seaborn
sudo pip3 install einops
sudo pip3 install tqdm
sudo pip3 install scipy
sudo pip3 install wandb
sudo pip3 install plotly
sudo pip3 install tensorboard-plugin-profile==2.4.0

wandb login
