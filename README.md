Abstract — 

This study is conducted to create a software for DMI and MIDI supported musical keyboard with the help of Hand Gesture and Motion Recognition system to do multitasking real time action while musicians are performing in their keyboard.

Keywords — 

Convolution Neural Network, Deep Learning, DAW, VST, OpenCV

Objective — 

To introduce hand gesture features which are compatible with DAWs (Digital Audio Workstation) and VSTs (Virtual Studio Technology) software for achieving track switching, volume modulation and special effects control during live performance.

<img width="432" height="236" alt="image" src="https://github.com/user-attachments/assets/08df2c8a-dbdb-47a8-ba23-5f0d1a0caf58" />


Result — 

The deep learning model was trained for 10 epochs, where each epoch represents one complete iteration over the entire training dataset. The training process involves adjusting the weights of the neural network in order to minimize the loss function, which measures the difference between the predicted output and the actual output.
The results indicate that the model achieved high accuracy for both the training and validation datasets. The training accuracy reached 99.88%, while the validation accuracy reached 99.92%. This suggests that the model has learned to accurately classify images of hand gestures into their respective categories.
In addition to accuracy, the output also displays the loss values for both the training and validation datasets. The loss represents the difference between the predicted output and the actual output, and a lower loss value indicates better performance. The model achieved a low training loss of 0.0044 and a validation loss of 0.0024. 
These low loss values suggest that the model is performing well and is not overfitting the training data.
 
Finally, the output also displays the results of evaluating the model on a separate test dataset. The test accuracy reached 99.92%, which is comparable to the training and validation accuracy, indicating that the model is performing well on unseen data. The low-test loss value of 0.0024 further confirms the good performance of the model on the test dataset.
While using this model on our software we have noticed that the lag or latency of the process is between the range of 40 ms to 50 ms.
