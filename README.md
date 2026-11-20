# DNA_ML

In this project, sequence of DNA dinucleotide is predicted based on the geometry and / or sequence of the neighboring nucleotides within DNA hexamers using ML models. 

The pipeline includes data processing, preparation of PyTorch-compatible datasets, several model architectures, custom loss functions, Trainer objejct that handles training, evaluation, weights saving/louding, as well as tools for auotmatic grid-search of hyperparameters and visualization of the predictions. 

Structural data of DNA hexamers are derived from molecular dyncamics simulations conducted within hexABC project (https://doi.org/10.1038/s41467-026-74390-5)

For more detailed description of the pipeline inspect jupyter notebook: ``DNA_sequence_prediction.ipynb``.
