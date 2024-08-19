Usage Instructions：

1. Use `Generate gradient.py` to load the model that has been pre-trained for 10 epochs and generate the backdoor gradients. The generated gradients are saved in the `saved_models` folder.
2. Run `TMM inversion.py` to load the pre-trained model and backdoor gradients, and perform backdoor sample reconstruction.