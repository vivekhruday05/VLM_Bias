# VLM_Bias

## Directory Stucture:

`clip_cda_different_freeze.py:` This file is used to finetune clip (for CDA method) with different settings in freezing the model. Also, the data it finetunes on is by default on whole anti-stereotypical data, if you want it on some subset of the data or some other type of data, change the _CLIPTrainDataset_ class from the same.

`clip_eval_hf.py:` This file evaluates the finetuned model on 6 different prompts and outputs RA_m, RA_f, RA_avg, GG. By default, this scripts tries to evaluates model with the path "fine_tuned_clip_hf". If you want to evaluate some other model or some model from HF, just change the model and processor names from the script.

`results_cda_anti_stereo.txt:` These contains results on 5 different settings where the finetuned settings (using CDA) used all the anti-streotypical data.

`results_cda_male_anti_stereo.txt:` These contains results on 5 different settings where the finetuned settings (using CDA) used only the anti-streotypical and male data.

`train_data.json:` Contains the data that we could use as training data for different settings.

`val_data.json:` Contains the data that we could use as validation data for different settings.

`test_data.json:` Contains the data that we could use as test data for different settings.
