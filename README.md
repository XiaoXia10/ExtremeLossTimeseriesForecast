# Forecasting Extreme Events in Time Series: Introducing the Extreme Loss Function for Neural Networks
Xiao Xia Liang, Dany Lauzon, Julien Straubhaar, Erwan Gloaguen, Maxime Claprood, Philippe Renard, and Reed Maxwell

This paper presents a novel loss function, the extreme loss function, for DNN models solving regression-type problems. 
The extreme loss function can be implemented in pytorch and tensorflow. 
Here, we tested the extreme loss function to forecast extreme events in karst spring discharges and groundwater heads in monitoring wells. 
The 2 models selected to test the extreme loss function are the [Graph WaveNet](https://arxiv.org/pdf/1906.00121) ([code](https://github.com/sshleifer/Graph-WaveNet))and the [GRU auto-encoder](https://arxiv.org/pdf/1406.1078).

The Graph WaveNet model has been adapted for hydrogeological data; this model is developed with pytorch.
The GRU auto-encoder is developed with tensorflow-keras.

The MPS DeeSse package can be found [here](https://github.com/randlab/geone).

### Data

All datasets used in this study can be found [here](https://www.dropbox.com/scl/fo/9z66v6y2qmxzldoqebr5a/AL9RviqcOfNo-lQHMh7QuGs?rlkey=6yxpkicxyuotd7xb3ts84gjig&st=bfdq708q&dl=0).

An example of the MPS code for generating the missing data for the Yamaska data can be found [here](https://www.dropbox.com/scl/fi/c0zdpxx0zvjt9sql8sigu/MPS_Yamaska_example.ipynb?rlkey=c2pzw0vq1a04oplqcg7l5vuw3&st=p18xejis&dl=0)

### Package requirements

- python 3
- pandas (2.1.1)
- numpy (1.26.4)
- scikit-learn (1.4.0)
- scipy (1.13.1)
- torch (2.1.2+cu121)
- tensorflow (2.14.0)
- [geone](https://github.com/randlab/geone)(1.2.17)

