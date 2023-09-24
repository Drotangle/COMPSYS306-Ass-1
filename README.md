# COMPSYS306-Ass-1
Part 1 of the assignment for COMPSYS 306 - AI and Machine Learning

## Some notes about what some parts mean:
- feature extraction: does this mean we remove the *background* and look for
features?
	- not sure what these would be other than just putting in the pixels
	- i think this is just done with the MLP model?
	- but if we are using SVM we might use HOG (histogram of ...)
- normalization and standardization: we would likely need to do this as all images have different value
ranges!
	- I'll do normalization as data does not follow standard distribution
	- Actually, could do HOG (in topic notes) instead - might actually use along with it as feature extraction?
	- is normalization implicitly done??? - this seems to be confirmed off of a link
		- https://stackoverflow.com/questions/44257947/skimage-weird-results-of-resize-function
- training and validation dataset - this seems to be where we split it up, we do some training then look at the validation dataset and tune hyperparameters as needed?
	- testing only comes in at end for those other things like f1 score and stuff? - yep! validation is for tuning hyperparameters
 - for mlp, looks like we need to define stuff in tensorflow but not actuall do it in tensorflow - this includes layers and activation functions!
	- we can tune activation functions, but have a look at layer sizes!
 	- for the solver, we are using an optimized version of stochastic gradient descent, due to the size of my data


## data analysis notes:
- from the bar chart, the frequency of images in categories is very unbalanced

## Things I may need to change:
- if we are splitting data 3 ways, train, validate, test, wouldn't we keep it in this split as we tune model with train and validate then use test?
- also note that my current accuracy with the model is 6.663% (this is low but randomly guessing signs is ~2.67%)
