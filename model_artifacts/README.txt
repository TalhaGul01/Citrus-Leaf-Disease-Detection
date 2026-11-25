MODEL FILES REQUIRED
--------------------
Copy these two files from your Google Colab session into the project root:
- citrus_disease_model.pkl
- citrus_label_encoder.pkl

They are not included in this ZIP because the trained binary files were not supplied in this chat.
The Flask application will not start until both files are added.
Check the scikit-learn version used to train and load the model; scikit-learn pickle files
may be incompatible across versions. Prefer the same version in Colab and locally.
Only load pickle/joblib files that you created or trust.
