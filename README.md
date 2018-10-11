# Tasks
This project is built to investigate Echo-State-Networks' performance 
in **chaotic time series prediction** and **singing voice detection**.
# Structure
`function.py` and `ESN.py` defined basic functions and classes which 
will be used in the relevant tasks. Other `.py` files are the scripts I
 used for impletenting experiments.
If you want to start a new file using ESN, please `import function.py` `import ESN.py` first.
# ESN
## Introduction
The basic model of ESN see http://www.scholarpedia.org/article/Echo_state_network.
The model used in this project is a simplified version without feedback connection.
## How to use ESN
1. Set appropriate parameters for ESN, then initialize the ESN using:
```Python
esn=ESN(n_inputs=1,n_outputs=1)
esn.initweights()
```
2. Pre-update ESN for 64-100 steps, and then discard the states.
*This is to discard the transient of ESN, subsequently ensure the internal states converge to the value which 
is merely determined by the input*
```Python
esn.update(inputSignal[:,:100],1)
del esn.allstate
del esn.state
```

3. Update ESN with input signals. Note to set the `ifrestart` parameter to 0.
```Python
esn.update(inputSignal,0)
```
4. Fit ESN with target signals.
```Python
esn.fit(inputSignal,targetSignal,0) 
```
if you want to use regularization, use the function `esn.ridgeRegres()`

5. Update the output signals.
```Python
esn.update(outputSignal,1)
```

6. Make prediction
```Python
esn.predict()
```

# Chaotic timeseries prediction
See `predictTimeshift.py` for example.

# Singing voice detection
1. Please download Jamendo dataset and put the annotation in `.\sources\annote`, put 
the audio files in `.\sources\audio_train`
2. Run the file `Extract_features.py`. 
3. Run the file `main.py` 