import os
import time
from PIL import Image
from models import ImageClassifier

# Examples for categories
categories = ['a man', 'a woman', 'flowers', 'a house', 'a little girl', 'A flock of birds', 'a bird']
# Initialize the classifier with the above category
classifier = ImageClassifier(categories=categories)

listFiles = os.listdir('data/')
sumTime = 0
for file in listFiles:
    image = Image.open('data/'+file)

    t0 = time.time()
    result, probability = classifier.predict(image)
    t1 = time.time()
    sumTime = sumTime + (t1-t0)

    print("\nThe predicted result of image '%s' is: "%file, result, ", Probability: ", probability)
print("\n\nSum time: ", sumTime, ", The average time: ", (sumTime/len(listFiles)))