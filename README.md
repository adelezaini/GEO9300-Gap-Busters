# GEO9300 Gap-Busters: Comparative Analysis of Machine Learning Methods for Gap-Filling Micrometeorological Data

Six machine learning algorithms for gap-filling latent heat flux (LE) data in micrometeorological observations are evaluated. The gap-filling method is ispired by the work of Vekuri et al. (2023), testing the ML algorithms performance on a synthetic dataset with superimposed realistic gaps.

This repository contains the code used for the course project *GEO9300: Geophysical Data Science* (2024)

**Contributors:** Malin Ahlbäck, Eivind Wiik Ånes, Adele Zaini.

## Abstract

Micrometeorological observations often contain missing data due to post-processing, instrument errors, or power outages, making gap-filling essential. This project compares six machine learning algorithms:

- Bayesian Additive Regression Trees (BART)  
- Long Short-Term Memory (LSTM)  
- Neural Network (NN)  
- Random Forest (RF)  
- eXtreme Gradient Boost (XGBoost)  
- Multiple Linear Regression (MLR)  

These methods are evaluated on their performance with random and structured data gaps. Our findings indicate that **XGBoost** and **Random Forest** perform best in terms of accuracy and robustness, particularly for long data gaps (>2 days).

## Repository Structure

TBD


## Key Features

1. **Synthetic Data Generation**  
   A custom model (*Skadi*) simulates micrometeorological variables for gap-filling tests. Input variables are typically available from model and reanalyis product outputs (e.g. incoming shortwave and longwave radiation, and near-surface air temperature).

2. **Gap Simulation**  
   Missing data is modeled based on real-world scenarios, such as power outages and post-processing filters, with both random and structured gap patterns.  

3. **Machine Learning Algorithms**  
   Models are tuned and evaluated using `scikit-learn`, `tensorflow.keras`, and `xgboost`. Evaluation metrics include Mean Absolute Error (MAE), Mean Squared Error (MSE), and R².

4. **Comparative Analysis**  
   Performance is assessed across varying data gap lengths, highlighting strengths and limitations for each algorithm.  


## Getting Started

### Prerequisites

TBD 

- Python >= 3.8  
- Required libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `tensorflow`, `matplotlib`  

### Installation
TBD

Clone the repository and install dependencies:

```bash
git clone https://github.com/adelezaini/GEO9300-Gap-Busters.git
cd GEO9300-Gap-Busters
pip install -r requirements.txt
```

## Running the Code

TBD

1. **Synthetic Data Generation**  
   Run `skadi_model.py` to generate synthetic micrometeorological data.  

   ```bash
   python scripts/skadi_model.py
   ```
   
2. **Gap Simulation**
3. **Model Training and Evaluation**
4. **Results Visualization**

## Citation

If you use this code, please cite the course project:  

**Ahlbäck, M., Ånes, E.W., & Zaini, A.** (2024). *Mind the Gap! A Comparative Analysis of Machine Learning Methods for Gap-Filling Micrometeorological Data.* GEO9300: Geophysical Data Science, University of Oslo.

## Contributing

🚧 Any contribution is welcome! Please open [pull requests](https://github.com/adelezaini/GEO9300-Gap-Busters/pulls) or use the [issue tracker](https://github.com/adelezaini/GEO9300-Gap-Busters/issues) to report, comment and suggest.


## License

📋 The code is released under the terms of the [MIT Licence](https://opensource.org/licenses/MIT). See the file [LICENSE.md](https://github.com/adelezaini/GEO9300-Gap-Busters/blob/master/LICENSE.md).

