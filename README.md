# keras-fbcnn
Keras implementation of FBCNN-style Bilinear CNN

Bilinear means that a neural embedding is multiplied by itself, creating an outer product matrix.
The values in the matrix are many linear combinations of the input values.


See the notebook for how it is used. This technique has proven very effective for classifying pathology slides, I think because medical slice images have a very strong multi-octave textural nature. But it's also good on the Stanford Dogs dataset!

This implementation is hard-coded for 2D CNN, but is extensible to any deep learning network embedding.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7500315/

This code is lifted and repackaged from https://github.com/NiFangBaAGe/FBCNN , supplied by the authors of the paper.

## But what does BilinearCNN do, really?

The features in a set of CNN feature maps form a vector space. An individual feature in this vector space is an abstract concept: the different features are overlapping fields of values. But, suppose instead that there are 7 discrete features in the dataset, and we create many vector samples with these 7 features modulated combinatorically. We then create a filter that examines all of the samples and decides that a few of the modulator combinations are very good ways to rank the features in importance.

This modulation and ranking is, deep down, achieved by placing a BilinearCNN2D layer followed by Dense FFN layer. The notebook demonstrates this technique applied to classifying the Stanford Dogs dataset. It's surprisingly effective.

## Should I use a Bilinear CNN outer product layer in my production image-processing network?
Bilinear CNN is not cheap! In the demonstration notebook, adding it to EfficientNetB0 for 224x244 images has the following numbers:

Original: 4m total weights, graph compilation 24 seconds, training epoch 33 seconds
Bilinear: 200m total weights, graph compilation 10 seconds, training epoch 56 seconds

Bilinear prediction time is presumably also affected. Bilinear CNN is a technique that allows us to throw a lot more CPU and memory at an image-processing network, and make it slightly more accurate. Such tools are handy to have in the practitioner's toolbox, but can be too expensive for a particular application.
