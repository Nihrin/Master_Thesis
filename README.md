# Master Thesis: Robustness of Classical and Credal Classifiers
This GitHub repository contains all classifiers, data sets, and results that were used in the resulting master thesis. All results are reproducible and usage of the created models should be easy when taking the [main.py](main.py) file as an example.

### Prerequisites
All code was created in a Python 3.10 environment. All required packages that can be pip installed can be found in the [requirements.txt](requirements.txt) file. To make use of the credal sum-product networks you will need to put the [cdeeprobkit](/models/cdeeprob) folder into the site-packages folder of your Python installation.

Some small edits were made to the original deeprobkit library were made to make use of it. In the file deeprob/spn/learning/splitting/gvs.py, lines 181-189 need to be replaced by the following code:

```
if distributions[i].LEAF_TYPE == LeafType.DISCRETE: 
    b1 = domains[i] + [domains[i][-1] + (domains[i][-1] - domains[i][-2])]
elif distributions[i].LEAF_TYPE == LeafType.CONTINUOUS:
    _, b1 = np.histogram(x1, bins='scott')
else:
    raise ValueError("Leaf distribution must be either discrete or continuous")

if distributions[j].LEAF_TYPE == LeafType.DISCRETE:
    b2 = domains[j] + [domains[j][-1] + (domains[j][-1] - domains[j][-2])]
elif distributions[j].LEAF_TYPE == LeafType.CONTINUOUS:
    _, b2 = np.histogram(x2, bins='scott')
else:
    raise ValueError("Leaf distribution must be either discrete or continuous")
```

### Running the code
In the [main.py](main.py) file there are some folder directories that should be edited when running the code on your own computer. Other than that all code should run as given. When modifying the code it is possible that errors may occur for which the code was not tested.

### Results
The [Results](/Results) folder contains all raw results created by the [main.py](main.py) file, as well as summarized results and tables used in the report, see [clean results](/Results/_clean_results.xlsx).
