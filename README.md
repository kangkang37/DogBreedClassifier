# Dog Breed Classifier

use Vision Transformer pre-trained Model, use 'dog breed identification' dataset to fine tuning the model.

## dataset
download dataset from kaggle: https://www.kaggle.com/c/dog-breed-identification/data

## files
DogBreedClassifer.ipynb: the main file to build vit model, and use dataset to fine-tuning model.
After fine tuning, save the model as `{model_name}.pt`.

app.py: Use the flask framework to build a simple web interface to upload images and predict the breed of the dog in the picture.

test_images: images to test

templates: Front-end web page code

## run 
first, run the `DogBreedClassifer.ipynb` and save the trained model in `.pt` file.

second, `python app.py`, it will load `.pt` file and make prediction.   
click the link to enter the local web page, upload the images and click the predict button to predict the animal breed.
