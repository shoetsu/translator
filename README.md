# Translator

The imprementation of the model to extract values for various attributes from the input sentence based on [Pointer Networks](https://arxiv.org/abs/1506.03134).

### Preparation
- Install python 2.7.14 and requirements

```bash
pip install -r requirements.txt
```
- Install nltk packages.

```bash
python scripts/setup_nltk.py
```

- **(optional)** Download Stanford POS tagger and add its path to your environment variable.

```bash
wget -O stanford-postagger.zip https://nlp.stanford.edu/software/stanford-postagger-2017-06-09.zip 
unzip stanford-postagger.zip 
echo "export STANFORD_POSTAGGER_ROOT=$(pwd)/stanford-postagger" >> ~/.bash_profile
source ~/.bash_profile
```


### How to run 
##### training
```bash
  ./run.sh {checkpoint_path} train [config_path]
  # if [config_path] won't be specified, configs/config will be used as default.

```

 ##### testing
```
  ./run.sh {checkpoint_path} test 
  # The config used for training a model will be restored in its checkpoint and automatically referred in testing.
```


For convinience, several arguments listed in **run.sh** can be dynamically specified at runtime as follows and they have higher priority than those in the config file.
```
 ./run.sh checkpoints/tmp train configs/config --batch_size=100 --test_data_path=dataset/test.csv
```

### Items and directories
```
├── checkpoints      # The directory to restore your trained models.
├── configs          # The directory to put your initial config files in.
├── dataset          # The directory to put your train/valid/test dataset in.
│   └── embeddings   # The directory to put your pretrained embeddings in.
├── main.ipynb       # The file for jupyter notebook demo.
├── requirements.txt # The list of the libraries you need to install.
├── run.sh           # The main script to run this model.
├── scripts          # Miscellaneous scripts mainly to analyze the results.
└── src              # The main codes of this project.
```

### Other useful tools
##### Tensorboard
Tensorboard is the official tool for visualization of Tensorflow's computation graphs and the result of your experiments. If you need it, I recommend you to see the process of Tensorboard on the remote server from your local browser by doing port forwarding in the way as follows, or with a few additional settings to your .ssh/config file.
```
  ssh -L 18889:localhost:8889 {your_remote_server}
  tensorboard --logdir={checkpoint_path} --port=8889
  # And access to localhost:18889 on your browser.
```


##### Jupyter notebook
I prepared a simple demo script that can be run by jupyter notebook.
You can try it by the similar way as tensorboard.
```
  ssh -L 18888:localhost:8888 {your_remote_server}
  cd {this_directory}
  jupyter notebook --port=8888
  # And access to localhost:18888 on your browser.
```
