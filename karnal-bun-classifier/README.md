# Karnal Bunt SVM Classifier

Here I perform automated analysis of microscopic imagery to detect the presence of Karnal bunt spores.  Using image processing and a SVM classifier, spores and non-spores are differentiated. 

* Run trainingExemplars.m (performs feature extraction on training images and saves features in training.mat)
* Run validationExemplars.m (performs feature extraction on validation images and saves features in validation.mat)
* Run main_ver2.m (trains SVM classifier with training.mat, and classifies objects in validation images using validation images features stores in validation.mat. Then, it performs feature extraction on test images and classifies each object)
