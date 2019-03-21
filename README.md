# shazam-air
~~music recognition and classifier project just like shazam but lighter and simpler~~

Music classifier project.


# Installation
    The Step 2 probably takes a few minutes, just be patient.
## 1.clone this repository
`$ git clone https://github.com/vsc-hvdc/shazam-air.git`
## 2.create a conda environment using environment.yml in the project directory
`$ conda env create -f environment.yml -n $YOUR_ENV_NAME`
## 3.activate the environment
`$ source activate $YOUR_ENV_NAME`
## 4.add package to python search path
`$ conda develop /Your/Project/Path/shazam-air/src`


# Run Demos
    Note: all the demo dataset, model and wav files are provided under the `/data` directory. 
    Feel free to record or import your own demo file though.
    
## 1.audioDemo
Just run the script in `demos/` directory

You should be able to see the plotting of two example audio file in time & frequency domain
![raw](asset/raw_chunk.png)
![rec](asset/rec_chunk.png)
![raw_spec](asset/raw_spec.png)
![rec_spec](asset/rec_spec.png)
## 2.musicAI
Run the script in `demos/` directory

You should be able to read the output of prediction electronic dance music genre.

You can choose to run it with example recordings `/dubsteps.wav` and `/future.wav` under `demos/demo_chunks/`.

But the model's accuracy is yet to improve, which will be discussed later.

The recommended method to run your model is to contruct your model outside the project, save your model data, and load it in this project. The interface for training your model is reserved but not completed now. Hope I can finish it sooner.

You can read the result of prediction from the last line of terminal output like this

`Huh, this song sounds like BigRoom`
## 3.train
You can try training your own model by running the `train.py` script.

This project have already include the dataset to train the model. This dataset is downloaded from [kaggle](https://en.wikipedia.org/wiki/Kaggle), See more infomation about it [here](https://www.kaggle.com/caparrini/beatsdataset)

You can see the PCA and LDA analysis of our dataset.

![pca](asset/pca.png)
![lda](asset/lda.png)

You should be able to see the training process in the terminal output.

```
Construct the model
Start training
Epoch 1/100
2019-03-21 11:12:32.030091: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2300/2300 [==============================] - 1s 250us/step - loss: 0.0417 - acc: 0.0478
Epoch 2/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0416 - acc: 0.0509
Epoch 3/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0415 - acc: 0.0648
Epoch 4/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0415 - acc: 0.0683
Epoch 5/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0414 - acc: 0.0648
Epoch 6/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0411 - acc: 0.0796
Epoch 7/100
2300/2300 [==============================] - 0s 79us/step - loss: 0.0408 - acc: 0.0909
.
.
.
Epoch 97/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0368 - acc: 0.2748
Epoch 98/100
2300/2300 [==============================] - 0s 79us/step - loss: 0.0368 - acc: 0.2683
Epoch 99/100
2300/2300 [==============================] - 0s 79us/step - loss: 0.0371 - acc: 0.2643
Epoch 100/100
2300/2300 [==============================] - 0s 78us/step - loss: 0.0373 - acc: 0.2435
```

Now you can get a model named `demo_exp.h5` in `data/model/` directory

The best model I trained reachs a accuracy of 80%. It's a good number for a dataset of only 2300 data when you try to classifiy 23 classes.

The advice is to create the dataset of your own (Yet the most tricky and impossible part is how you can access enough amount of music data), the original dataset is not fully reliable, but it's the best I can find.

Or you can just keep trying Tweaking Your Model!

![](asset/meme.gif)
## 4.real-time spectrum demo
This demo is recommended running on the jupyter notebook because I met some strange bugs when trying to run it by script.

You can open the terminal and run 
`$ jupyter notebook`, which should be installed when you create the env for this project in **installation step2**.

Open `real_time_demo.ipynb`, run through the demo, and you should be able to see a dynamic sepctrum while playing the demo chunk.

I found matplotlib extremely not good for real-time plotting cuz it's really slow and memory-consuming. The demo is the best I can do now, but still have obvious delay. Maybe we can try other library like openGL.

The spectrum is not ideal now cuz the y-axis is not fixed. Right now I can't figure out how to disable its auto-scaling setting.

![REAL-TIME DEMO](asset/rt-spec.gif)
