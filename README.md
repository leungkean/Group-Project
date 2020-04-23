# Question Generation using Paragraphs
<h1>Setup</h1>

1) Install Pytorch, Torchvision, transformers, tensorboard
(pip install pytorch torchvision transformers tensorboard) 

2) Download SciQ dataset: https://allenai.org/data/sciq and unzip it in the directory in which you unzipped
this in a file titled Sciq

3) run python pre_processing_sciq.py 

4) Now in src.py it is setup to train the model. 

With a little modification to the forward pass (in particular taking the que on how to 
output the actual output of the model in the print statements in the train() function) a 
system can be setup to reliably test the model of the test set that is created during pre_processing. The 
data the is put into the train function can simply be changed by changing the path that is put into it in the 
build_model_and_train_function

# Source Code
- (https://github.com/byrdofafeather/ResearchTestingBed)

# Sources:
- https://www.kaggle.com/stanfordu/stanford-question-answering-dataset
- https://www.aclweb.org/anthology/D18-1424.pdf
