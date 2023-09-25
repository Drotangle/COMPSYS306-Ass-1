## Included files:

### Report:
- COMPSYS-306 Assignment Part 1 Report.pdf:
	- My report detailing the machine learning process in detail

### Python files:
- main.py:
	- This is the file where all functions for the machine learning may be run from (just uncomment a line to make it run)
	- Also, contains functions that handle most dataset extraction / formatting, analysis, and data pre-processing
		- This also includes saving the split datasets to disk
- mlp_model.py:
	- Contains function to make an mlp model and use valdiation set to look at results, as well as one that uses the testing dataset to look at the results
		- Note that these two also apply a final feature extraction step of Histogram of Oriented Gradients to the data
	- Also contains functions to save and load the models to and from the disk
- svm_model.py:
	- Has the same equivalent functions as mlp_model.py, except that it includes an individual testing method, to ensure my results were working properly
- show_time.py:
	- Includes a single function, used by all files, that shows a message of the "start" or "finish" time for debugging

### .joblib files:
- x_training.joblib:
	- the dataset to test on once hyperparameter tuning is finished
- y_training.joblib:
	- the labels of the testing dataset, to help analyse how good the final model is
- mlp_model.joblib:
	- the complete, hyper-parameter tuned and tested Multi-Layer Perceptron model
- svm_model.joblib:
	- the complete, hyper-parameter tuned and tested Support Vector Machine model