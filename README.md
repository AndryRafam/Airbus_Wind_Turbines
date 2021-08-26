# Airbus_Wind_Turbines

In this project, we are trying to build some classifiers in order to discriminate (classify) wind turbines from grounds.
##### The dataset can be downloaded from here: https://www.kaggle.com/airbusgeo/airbus-wind-turbines-patches

### We are essentially using two methods in order to achieve our goal.
    - The first one is to build a CNN (Convolutional Neural Network) from scratch.
    - The second one is to use transfer learning (feature extraction).
    
### Result of the experiment

#### CNN from scratch (Perso folder) with data augmentation:
    - Test Accuracy: 98.76 % (over 3 epochs)
    - Test Loss: 5.33 % (over 3 epochs)
#### MobileNetV2 - Transfer Learning (feature extraction) with data augmentation:
    - Test Accuracy: 98.36 % (over 1 epoch)
    - Test Loss: 4.78 % (over 1 epoch)
#### EfficientNetB3 - Transfer Learning (feature extraction) with data augmentation:
    - Test Accuracy: 97.20 % (over 3 epochs)
    - Test Loss: 7.55 % (over 3 epochs)

# ACKNOWLEDGEMENT
This little project would not have been possible without Kaggle and Airbus DS GEO S.A. Thank you for all your hard work and the fun during the tagging and hacking sessions.
