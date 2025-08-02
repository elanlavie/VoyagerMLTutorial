# 🛰️ Voyager Signal Similarity Search with β-VAE (Colab Tutorial)

# [CLICK HERE to open the notebook in Google Colab in your browser!](https://colab.research.google.com/github/elanlavie/VoyagerMLTutorial/blob/main/VoyagerMLTutorial.ipynb)

## 👋 Introduction
In this tutorial, we will use real Voyager-1 radio spectrogram data from the Breakthrough Listen public archive to demonstrate a content-based signal search. 

## 📚 Background
We aim to find signals that look similar to a given “query” signal in Voyager's spectrograms. This is a lot like a [reverse Google Image Search](https://images.google.com) if you've ever tried that!

<div align="center"><br><img src="images/GoogleImageSearch.png" alt="GoogleImageSearch" width="400"><br></div>

<br>For example, if you wanted to find an image of another orange/grey spotted calico cat, you could upload your image to Google Image Search and it will return similar hits: <br>

<div align="center"><br><img src="images/CatImageSearch.png" alt="GoogleImageSearch" width="400"><br></div>

<br>We'll be doing something very similar, but instead of uploading an image of a cat, we'll upload an image of a signal we believe looks like the one Voyager (or an alien!) emits:<br>

<div align="center"><br><img src="images/VoyagerSignal.png" alt="VoyagerSignal" width="400"><br></div>

Then the machine learning algorithm, called a Beta-Variational Autoencoder (β-VAE), will look through a bunch of data that we want it to search for similar looking signals! If all goes well, it will return the top 10 lookalikes ("hits"), which we can then check out for ourselves to see if they really are extraterrestrial spacecraft!!

<br>The dataset consists of six 5-minute scans of Voyager 1 acquired by the Green Bank Telescope in July 2020 (3 scans pointed ON the Voyager spacecraft and 3 OFF target). For simplicity, we will ignore the ON/OFF labeling (i.e. we won’t perform interference filtering by removing OFF-target signals. Instead, we’ll treat all scans uniformly as a pool of spectrogram data in which to search for lookalike signals.<br>

To better understand how the machine learning algorithm works, it may be helpful to watch:
1) This wonderful video from Carnegie Mellon University to develop one's intuition on "latent space": [Cat's in Latent Space](https://www.youtube.com/watch?v=hb-KT66rCT8) :)
2) This short video on Convolutional Neural Networks (like a β-VAE): [Convolutional Neural Networks](https://www.youtube.com/watch?v=LEAFHHNk8II)
3) This longer 3Blue1Brown video on Neural Networks more generally: [But What is a Neural Network](https://www.youtube.com/watch?v=aircAruvnKk)

## 🗺️ Overview
We follow a simplified version of the method by Ma et al. (2023). First, we use an energy detection approach to extract small signal snippets from the full spectrogram – essentially finding time- frequency patches with power significantly above the noise floor. These snippets (for example, containing narrowband Doppler-drifted carrier signals) will be our “catalog” of candidate signals. Next, we load a pretrained β-Variational Autoencoder (β-VAE) encoder model based on the one created by Peter Ma et al (same weights, but possibly a slightly different architecture). This neural network encodes each spectrogram snippet into a low-dimensional latent feature vector that captures its salient morphological features (shape, bandwidth, drift rate, etc.). Finally, given a selected query snippet, we compute the cosine similarity between the query’s embedding and all other snippet embeddings to retrieve the most similar signals. We will display the top 10 matches as images and discuss the results.

## ✅ Requirements
This notebook uses TensorFlow (with GPU support) and common scientific Python packages (NumPy, h5py, etc.) which are pre-installed in Colab. We will also install Blimpy (Breakthrough Listen I/O toolkit) to help read .h5 spectrogram files. The β-VAE encoder model will be downloaded and loaded with Keras. Let's get started by importing necessary libraries and preparing the data!
