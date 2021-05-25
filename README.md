# HLCV_CNN_assignment3

## Notes

1. Q1.C, **VisualFilter(model)** function is modified to accept an additional _boolean_ argument: before.
   When _before_ = False, the visualized filters are saved with a different filename.  
   The functional call in line at the bottom of ``ex3_convnet.py`` is changed to add this addition argument.  
2. Q2.A, To activate **BatchNormalization**, uncomment the line above model initialization, line #245: ``norm_
layer = True``.  
   ``nn.BatchNorm2d`` applied _only_ to each of the 5 convolution layers.  
   The parameter for ``nn.BatchNorm2d`` is the out_channel parameter of the previous ``nn.conv2d`` layer, i.e. 
   the value of ``h_size``.
   
