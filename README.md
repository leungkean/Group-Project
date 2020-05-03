# Question Generation using Paragraphs
<h1>Setup</h1>

1) Install Pytorch, Torchvision, transformers, tensorboard
(pip3 install pytorch torchvision transformers tensorboard) 

2) Download SciQ dataset: https://allenai.org/data/sciq and unzip it in the directory in which you unzipped
this in a file titled Sciq

3) Run python pre_processing_sciq.py

4) Now run src.py and the model will randomlly select items from the test data and predict an output using greedy search  

## Sources
- https://www.kaggle.com/stanfordu/stanford-question-answering-dataset
- https://www.aclweb.org/anthology/D18-1424.pdf

## PDF Overleaf Doc
- https://www.overleaf.com/project/5e99fb9f43ff67000121a464
