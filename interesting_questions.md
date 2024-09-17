# Interesting questions
## Priority ideas
* Train an autoencoder on ImageNet
* Develop a VAE version

## Important ideas
* Compare the reconstructions of different latent space dimensions (i.e. how much better is 128 than 32?)
* How do small changes to the latent representation influence the reconstructed image?
* Do similar images (or similar cows rather) have similar latent space represenations?
* For a version trained on imagenet: Can we build another model that can predict tags (i.e. trees, cars) on these representations? More generally, how can we "read" these powerfull abstractions of images?
* Sampling: Are there any kind of symmetries that the representations have to have in order to produce meaningfull decodings?
* Build another autoencoder with a very different architecture (no convolutions). How does that change the reconstructions?
    * We can simply use a vision transformer for the encoder, the decoder is the difficult part

## Medium term
* Write a paper