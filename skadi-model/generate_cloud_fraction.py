import pandas as pd
import numpy as np
from functions import open_input_file
from equations import cloud_fraction

def main():
    
    LWin_data = open_input_file("LWdown")

    cloud_fraction_data = cloud_fraction(LWin_data)
    cloud_fraction_data = cloud_fraction_data.to_frame(name="cloud_fraction")
    
    # Export the DataFrame to a CSV file
    cloud_fraction_data.to_csv("../data/synthetic_dataset/cloud_fraction.csv")
    print("Cloud Fraction successfully saved in data/synthetic_dataset.")

if __name__ == "__main__":
    main()

