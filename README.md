# Convolutional neural networks for classification of American Sign Language (ASL)
## Description
We present implementations of a deep Convolutional Neural Network (CNN) model that can accurately translate images of sign language of hand
to alphabets. With an input size of 200x200 pixels’ image, the model can achieve an accuracy of over 97% with 1.17 ms inference time. We also introduce a method which utilizes the pre-trained pose estimation model MediaPipe to mark hand keypoints and skeletons from images, using them as additional features to enhance the accuracy of our model. This practice in computer vision could enhance communication experience for individuals with speech disability and eliminate discrimination to people who rely on sign language for communication.

## Prerequisites
* Latest version of Python 3 with pip:

  ```
  pip install torch torchvision pillow matplotlib numpy scipy pandas scikit-learn mediapipe
  ```
* Jupyter Notebook
* Dataset: Please download at https://www.kaggle.com/datasets/grassknoted/asl-alphabet/data

## File Description:

* **CNN_for_ASL_Classification.ipynb** : Main notebook, include everything. Please just clone and run.
* **evaluation.ipynb**: For loading the pre-trained model and evaluating, if necessary.
* **best_models/**: Pre-trained parameters.
* **scripts/** : Clean version of scipts.

  
