This repo contains an implementation of the paper [Mention Annotations Alone Enable Efficient Domain Adaptation for Coreference Resolution](https://arxiv.org/abs/2210.07602). Most of the code is a modified version of [this repo](https://github.com/mandarjoshi90/coref/tree/master).
# Setup
- Install python3 requirements ``pip install -r requirements.txt``
- ``export data_dir=</path/to/data_dir>``
- ``./setup_all.sh`` builds custom kernels
# Data Preparation
Given .conll files as input run ``python minimize.py config/minimize.conf`` to generate .jsonlines files. Models take .jsonlines formatted files as input.
- To set up OntoNotes, run ``setup_training.sh``
- To set up the i2b2/VA corpus, first obtain access to the raw data [here](https://portal.dbmi.hms.harvard.edu/). We found [this repo](https://github.com/mxhofer/i2b2_2009-to-CoNLL.git) useful for parsing files into the .conll format.
# Training Models
- Experiment configurations are found in ``config/experiments.conf``.
- To train a model, run ``python src/train.py <experiment-name> config/experiments.conf``
# Evaluating Models
- To evaluate the model, run ``python src/evaluate.py <experiment-name> config/experiments.conf``
- This should produce a file for each task that the model was trained on (e.g. coreference, mention detection) corresponding to task-specific model performance.


 
