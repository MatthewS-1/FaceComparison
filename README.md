# FaceComparison
This respository uses a siamese neural network to compare faces and determine whether the two faces are similar\
I have made a couple changes with data augmentation, kernels, and layers, so this is a bit outdated\
It is important to note that both images are fed into the same model to produce an encoding

# Progress Log
Noticed that the loss and accuracy seemed to jump around\
This problem seems to be because the default learning rate is too high.
Model uses RMSprop which has a learning rate of 0.001; lowering this learning rate to 0.00035 seemed to bring the best performance
