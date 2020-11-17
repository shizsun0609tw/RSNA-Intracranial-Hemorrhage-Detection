# RSNA-Intracranial-Hemorrhage-Detection

A project about RSNA-Intracranial-Hemorrhage-Detection  
(Kaggle:https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection)

dataset:  
* (epidural/healthy/intraparenchymal/subarachnoid/subdural) each class has 1000 CT data  

use:  
* use preprocessing.py, filter.py to preprocessing the training dataset  
* use VGGmodel.py or seresnet.py to train model  
  
detail:  
* preprocess:  
  * split CT data to 3 channel PNG data  
* filter:  
  * image preprocessing by opencv(noise reduction, clipping, enhacement, binarization)  
* VGGmodel:  
  * VGG16 model  
* sersnet:  
  * resnet14 model  
  
  
Don't forget to change the data path in code
