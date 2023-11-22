# Automatic-Voice-Recognition
Implementation of Deep Learning to Perform Speaker Recognition.

## Note to Users: Since gradio only accepts wav files as inputs you would need to convert mp3 files to ones acceptable by gradio. In order to use mp3 files for testing please make sure you have ffmpeg installed and added to your path.You can find tutorials on how to do that here https://phoenixnap.com/kb/ffmpeg-windows

Speaker recognition is the process of automatically recognizing who is speaking 
on the basis of individual information included in speech waves. This 
technique makes it possible to use the speaker's voice to verify their identity 
and control access to services such as voice dialing, banking by telephone, 
telephone shopping, database access services, information services, voice 
mail, security control for confidential information areas, and remote access to 
computers. Design an Automatic Speaker Recognition System which can be 
used for several security applications. 

Automatic voice detection using CNN and Spectrogram images 
libraries used 
Tensorflow
Opencv
tensorflow
numpy 
matplotlib
cv2 as cv
librosa
gradio

This project aim was to use Convolutional neural networks to be able to classify voices of an individual amongst other voices 
The data sets we  used were the LJSpeech dataset and the cvcorpus dataset.The LJspeech dataset contains a voice of a particular speaker whose voice was the main subject that we were using against the rest.

We first needed to pass the audio files through librosa(a python dependency that produces spectrogram images based on audio input) to be able to generate the spectrogram images of each sound file .
We then had to group these images into different folders(CVcorpus folder and LJSpeech folder) to be able train them
We then created our machine learning model that we would be using to train the images to be able to run the predictions 
For the training we used transfer learning technique where we utilized the  VGG16 pretrained model 
We also used the pre trained weights of the model to reduce training time . also we didn’t include the last few layers of the model because we were going customize it to fit our problem 
We added  a flatten layer on top of our pretrained model, continued with dense layer and finished it up with a final dense layer with 1 neuron (because we are going to do a binary classification ) 
We compiled the model using the sgd optimizer and a loss function 'binary crossentropy'
We then trained the model with 10 epochs and achieved an accuracy of 100 within a time span of 79mins 52.2s 
After Trianing we saved the model so we can use it for prediction in the future


## TESTING MODEL BY PASSING IN A VOICE FILE
To test our model we imported various dependencies such as 
Opencv 
tkinter
matplotlib 

We then created a wen-gui using graio to prvide a GUI to be able for users to upload their voice files 
We then converted the voice file using the librosa library to get the spectrogram image 
then after we preprocessed the spectrogram image
After preprocessing we use tensorflow to load the model we saved earlier and then use the loaded model to predict the preprocessed image 
After prediction the model generates a 1 or a zero to indicate if the voice file matches the one we trained or not
the prediction is then seen on the gui as a diaglouge message "voice matched" or "voice does not match" based on the output of the prediction.

