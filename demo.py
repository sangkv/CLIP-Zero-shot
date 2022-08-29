import os
from PIL import Image
from models import ImageClassifier

# Examples for categories
categories = ['a man', 'a woman', 'flowers', 'a house', 'a little girl', 'A flock of birds', 'a bird']
# Initialize the classifier with the above category
classifier = ImageClassifier(categories=categories)

for file in os.listdir('data/'):
    image = Image.open('data/'+file)
    result, probability = classifier.predict(image)

    print("\nThe predicted result of image '%s' is: "%file, result, " ", probability)

