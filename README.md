# CommitFit

## Environment
Code demo for paper Boosting Commit Classification with Contrastive Learning
In order to easily reproduce our method, we use ```pip freeze > requirements.txt``` to output the packages used in our experimental environment.
you can try ```pip install -r requirements.txt``` to obtain the similar environment with us.

## Datasets
The datasets we used can be find from the dataset folder 
Commit_datasets.csv is the dataset with 3 labels, you can also find download it straight from https://zenodo.org/record/4266643#.X6vERuLPxPY

negative+CC-900repos and positive+CC-900repos are the dataset with 2 labels, you can also find download it straight from https://github.com/davidleejy/wnut21-cotrain. And we concatenated and randomly shuffled them as dataset.csv 

All experiment results are stored in the notebook folder, for example, the folder with the name E-2-5 means, the E(experiment) result on 2(dataset with 2 labels) with 5 shots(5 samples for each label). 
## Directory
To avoid the file path-related errors, you are recommended to organize the script as follows: <img src="https://github.com/AppleMax1992/CommitFit/assets/77500295/7002f3f1-ba09-42c6-b2ac-b87ad534d39c" width="100" height="100">


