# NucleoNN

I. README
------------
Title: Nucleotide Skip-gram Neural Network (NucleoNN) 
Date: 2018-02-20

Description: represents alleles as distributed vectors that encode for associations with other alleles within a given window during the course of evolution, and applies artificial neural networks for feature learning from genetic sequences to learn these vectors as allele embeddings.

Install: This package requires TensorFlow (Version 1.2.1 or higher), Python (Version 3.5.1 or higher) and R (Version 3.3.1 or higher).



II. AUTHORS
------------
Author: Hyunjin Shim
Reference: Shim, H., Feature learning of virus genome evolution with the nucleotide skip-gram neural network (submitted)
Maintainer: Hyunjin Shim <jinenstar@gmail.com>



III. File list
------------
data_process.Rmd		processes raw DNA sequence data into NucleoNN input data format
process_data_echo.py		processes NucleoNN input data into TensorFlow input format
nucleotide_skipgram.py		trains TensorFlow input data for learning allele embeddings in TensorFlow
PCA_nearsest.Rmd		visualises PCA components of allele embeddings by correlation plots and hierarchical clustering



IV. ANALYSIS
-------------
Run: [Step 1] import raw DNA sequence data; [Step 2] find minor alleles above sequence error; [Step 3] create input data for NucleoNN; [Step 4] train NucleoNN input data in TensorFlow; [Step 5] visualise allele embeddings using TensorBoard; [Step 6] download PCA components from TensorBoard; [Step 7] visualise PCA components by correlation plots; [Step 8] apply hierarchical clustering; [Step 9] cluster validation and biological interpretation

Examples: Worked example for each function with the necessary inputs and data formats is available at gist.github.com/hjshim/.

NucleoNN input format:
0 = major allele
1 = minor/derived allele (standing variation or de novo mutation)
Allele set = set of M minor alleles rising above sequencing error (encoded as 1,2, …, M)
Window size = extra zeros inserted in between sequence samples to avoid overlapping sequences due to wide window sizes (seq1, 0,…,0, seq2, 0,…,0, ..., 0,…,0, seqT)

To run TensorFlow:
python nucleotide_skipgram.py

To visualize TensorBoard:
tensorboard --logdir=./directoryname



IV. LICENSE
------------------------
License: GPL-2 | GPL-3