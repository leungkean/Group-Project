# Question Generation using Paragraphs Code Setup
<h1>Setup</h1>

1) Install Pytorch, Torchvision, transformers, tensorboard
`pip3 install torch torchvision transformers tensorboard`

2) Download the SciQ dataset: https://allenai.org/data/sciq and unzip it in the directory in which you unzipped
this project in a file titled **Sciq**

3) Run `python pre_processing_sciq.py`

4) Now run `python src.py` and the model will randomly select items from the test data and predict an output using greedy search  

Note: Final paper is given in FinalProjectPaper.pdf
