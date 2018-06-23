# Image-Processing
## Objective
Make an application that notices what input image is by using image processing.

## References
* [Convolutional Neural Networks (CNNs / ConvNets)](http://cs231n.github.io/convolutional-networks/)
* [Epoch vs Batch Size vs Iterations](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
* [Keras Home](https://keras.io/)
* [Keras Image Processing Doc](https://keras.io/preprocessing/image/)
* [How to Use The Pre-Trained VGG Model to Classify Objects in Photographs](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/)
* [classifier from little data script 1](https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)
* [classifier from little data script 2](https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069)
* [classifier from little data script 3](https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975)
* [How to Make Predictions with Keras](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)
* [How to deploy keras model](https://www.youtube.com/watch?v=f6Bf3gl4hWY)
* [Machine Learning with Python : Image Classifier using VGG16 Model - Part 1: Theory](https://www.techkingdom.org/single-post/2017/11/07/Machine-Learning-with-Python-Image-Classifier-using-VGG16-Model---Coming-Soon)
* [Data set Wikipedia](https://en.wikipedia.org/wiki/Data_set)

## About Dataset
### What is a data set?
A data set (or dataset) is a collection of data. Most commonly a data set corresponds to the contents of a single database table, or a single statistical data matrix, where every column of the table represents a particular variable, and each row corresponds to a given member of the data set in question.

### How can we get a image data set?
1. Type "Image tag" in https://image.google.com. And click "inspect" after clicking right click.
2. Copy and paste below code at console tab. (You maybe copy twice) Then you will get "url.txt".
In "url.txt", there are image urls.
``` javascript
// pull down jquery into the JavaScript console
var script = document.createElement('script');
script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
document.getElementsByTagName('head')[0].appendChild(script);

// grab the URLs
var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });

// write the URls to file (one per line)
var textToSave = urls.toArray().join('\n');
var hiddenElement = document.createElement('a');
hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
hiddenElement.target = '_blank';
hiddenElement.download = 'urls.txt';
hiddenElement.click();
```
3. Using download.py, download images in url.txt. (You should delete images that you failed to download. These are not images.)
4. Because there are many thing you don't want, you should clean(refine) data set.

## What is the keras?
Keras is a high-level neural networks API, written in Python and capable of
running on top of TensorFlow, CNTK, or Theano. It was developed with a focus
on enabling fast experimentation. Being able to go from idea to result with
the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

* Allows for easy and fast prototyping (through user friendliness, modularity, and extensibility).
* Supports both convolutional networks and recurrent networks, as well as combinations of the two.
* Runs seamlessly on CPU and GPU.

## Flow Chart of this program
![Flow chart of this program](/images/flow_chart.png?raw=true)

## Flow Chart of VGG model
![Flow chart of VGG model](/images/Plot-of-Layers-in-the-VGG-Model.png?raw=true)

## Terminologies
### Epoch
One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.

### Batch Size
Total number of training examples present in a single batch.

### Iterations
Iterations is the number of batches needed to complete one epoch.

