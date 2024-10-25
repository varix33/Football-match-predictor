# Football Match Results Predictor

This repository contains the code and resources for a project that predicts the outcomes of international football matches. Developed using machine learning techniques, the predictor forecasts the winner of FIFA championship games based on historical match data and team rankings.

## Project Overview

The goal of this project is to create a predictive model for football matches between national teams. By analyzing team performance, historical match results, and ranking data, we aimed to develop an accurate model capable of predicting match winners. Key steps included data preprocessing, feature engineering, model selection, and hyperparameter tuning.

## Data Sources

Three datasets were used for training and prediction:
1. **International Football Results (1872-2023)**: Records of international matches with details like date, teams, scores, and tournament type.
2. **FIFA World Ranking (1992-2023)**: Historical rankings and points of national teams.
3. **Recent Rankings Data (2023)**: A custom dataset with the latest FIFA rankings not covered in the main ranking dataset.

## Key Steps

1. **Data Preprocessing**: Cleaned and standardized team names, removed irrelevant entries, and merged datasets based on match dates.
2. **Feature Engineering**: Generated features such as recent win rates, average goals, and team performance indicators for more accurate predictions.
3. **Model Training**: Trained multiple models, including Random Forest, SVM, Adaboost, and Multilayer Perceptron, and evaluated them for classification accuracy.
4. **Hyperparameter Tuning**: Used grid search and cross-validation to optimize hyperparameters, selecting SVM as the final model for its accuracy and performance.

## Project Structure

- `src/` - Includes trained model script and the graphical user interface developed with Tkinter.
- `main.py` - To run all the scripts.
- `preprocessing/` - Contains processed datasets and scripts for data cleaning and features engineering.
- `graphical data/` - Images of graphical results obtained on data and models.
- `docs/` - The report of the project.
- `rankingData.csv + rera.csv + reraBase.csv + tournament_teams.csv` - Cleaned datasets .

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/football-predictor.git
   cd football-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the predictor**:
   - Use `./main.py` to train the model with preprocessed data.

## Graphical Interface

A GUI is available for simulating tournament stages and viewing prediction outcomes. Users can load match data files and visualize tournament progress, including winning probabilities.

## Results and Conclusions

The SVM model achieved the best accuracy without overfitting, making it the preferred choice for predicting match outcomes. The final model considers recent performance metrics and team rankings to generate reliable predictions.

## Authors

- Karl Alwyn Sop Djonkam
- Bruneau Antoine
- Di Placido Anna

Supervised by: Enrico Formenti, University of Nice Côte d'Azur

## License

This project is licensed under the MIT License.

--- 

This README includes an overview, instructions for using the project, and a description of the project structure and purpose. Let me know if you'd like adjustments for specific details!

[Consult the report](docs/Football_Predictor_Report.pdf)
