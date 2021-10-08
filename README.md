# keras-fbcnn
Keras implementation of FBCNN-style Bilinear CNN

Bilinear means that a neural embedding is multiplied by itself, creating an outer product matrix.
The values in the matrix are many linear combinations of the input values.


See the notebook for how it is used. This technique has proven very effective for classifying pathology slides, I think because medical slice images have a very strong multi-octave textural nature. But it's also good on the Stanford Dogs dataset!

This implementation is hard-coded for 2D CNN, but is extensible to any deep learning network embedding.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7500315/

This code is lifted and repackaged from https://github.com/NiFangBaAGe/FBCNN , supplied by the authors of the paper.

## But what does BilinearCNN do, really?

The features in a set of CNN feature maps form a vector space. An individual feature in this space is an abstract concept. But, suppose instead that there are 7 discrete features in the dataset, and you create many vector samples with these 7 features modulated combinatorically. You could then create a filter that examines all of the samples and decides that a few of the modulator combinations are very good ways to rank the features in importance.

This modulation and ranking is achieved by placing a BilinearCNN2D layer followed by Dense FFN layer.
