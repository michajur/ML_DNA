# DNA_ML

Project explores whether DNA dinucleotide sequences can be predicted from the local geometry and/or sequence of neighboring nucleotides within DNA hexamers using machine learning.

The **PyTorch-based pipeline** includes data preprocessing, preparation of datasets and several MLP and CNN architectures, including variants that exploit DNA reverse-complement symmetry. Models support both 16-class dinucleotide prediction and separate 4-class nucleotide predictions, with custom loss functions designed to improve distinction between similar nucleotides and enforce prediction consistency within complementary sequences.

The pipeline also provides a configurable **Trainer** class supporting CPU/GPU training, evaluation, early stopping, learning-rate scheduling, and model checkpointing. Additional utilities enable automated hyperparameter grid searches, experiment configuration tracking, and prediction analysis using confusion matrices.

The structural data were obtained from MD simulations conducted within the [hexABC project](https://doi.org/10.1038/s41467-026-74390-5).

For a more detailed description see the Jupyter notebook `DNA_sequence_prediction.ipynb`.