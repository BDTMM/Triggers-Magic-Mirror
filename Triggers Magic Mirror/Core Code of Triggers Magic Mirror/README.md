Usage Instructions：

1. Use `Generate_gradient.py` to load the model that has been pre-trained for 10 epochs and generate the backdoor gradients. 
2. The generated gradients are saved in the `saved_models` folder.
3. Run `TMM_inversion.py` to load the pre-trained model and backdoor gradients, and perform backdoor sample reconstruction.
4. Save the reconstructed image in the `result` folder