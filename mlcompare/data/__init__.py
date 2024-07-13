import os

# Set empty environment variables for downloading data from Kaggle
# To avoid importing the library throwing an error
os.environ["KAGGLE_USERNAME"] = ""
os.environ["KAGGLE_KEY"] = ""
