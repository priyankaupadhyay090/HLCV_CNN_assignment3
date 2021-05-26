# HLCV_CNN_assignment3

## Notes

1. Q1.C, **VisualFilter(model)** function is modified to accept an additional _boolean_ argument: before.
   When _before_ = False, the visualized filters are saved with a different filename.  
   The functional call in line at the bottom of ``ex3_convnet.py`` is changed to add this addition argument.  
2. Q2.A, To activate **BatchNormalization**, specify True/False on the command line:  
   ``python ex3_convnet.py -n True``
   ``nn.BatchNorm2d`` applied _only_ to each of the 5 convolution layers.  
   The parameter for ``nn.BatchNorm2d`` is the out_channel parameter of the previous ``nn.conv2d`` layer, i.e. 
   the value of ``h_size``.
   
3. Q3.B, To specify dropout value: type a value between [0.1,0.9] on the command line:  
``python ex3_convnet.py -d 0.5``
   
3. Q3.A, To specify how many transform methods to add to compose function for data augmentation, specify int
values between [0,4]; 0 deselects all methods.
   
   
## Configure Command Line Arguments for Experiments

To configure hyperparameter values for experiments, specify options below

```
usage: ex3_convnet.py [-h] [-e EPOCH] [-n NORM] [-d DROPOUT] [-j JITTER]
[-a AUGMENT] [-v DISP] [-s E_STOP] [-c COMMENT]

ex3 convnet param options

optional arguments:
-h, --help            show this help message and exit

-e EPOCH, --epoch EPOCH Number of epochs [default = 20]
-n NORM, --norm NORM  Turn on Batch Normalization [True/False]
-d DROPOUT, --dropout DROPOUT Specify dropout p-value .e.g values between [0.1,0.9]
-j JITTER, --jitter JITTER Specify ColorJitter param [default = 0.2]
-a AUGMENT, --augment AUGMENT How many data augmentation techniques to add to
compose e.g. values between [1-4], 4 uses all transform techniques
-v DISP, --disp DISP  Show plots to display [default = False; plots are saved without display]
-s E_STOP, --e_stop E_STOP Apply early stop [default = False]
-c COMMENT, --comment Run comment for wandb run name [default = "q1_3"]
```
   
