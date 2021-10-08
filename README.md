# keras-fbcnn
Keras implementation of FBCNN-style Bilinear CNN

Bilinear means that a neural embedding is multiplied by itself, creating an outer product matrix.
The values in the matrix are many linear combinations of the input values.
The effect of this in neural networks is to arrange many combinations of the features in the embedding.
This output is fed to a simple FFN "Dense" layer.

See the notebook for how it is used.

This implementation is hard-coded for 2D CNN, but is extensible to any deep learning network embedding.

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7500315/
