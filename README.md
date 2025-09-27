Air Quality Health Impact Classification
This project uses machine learning to classify health impact levels based on air quality data. It applies both Logistic Regression and Random Forest models to evaluate predictive performance and visualize results,

📁 Project Structure
- air_quality_health_impact_data.csv: Dataset containing air quality metrics and health impact labels
- main.py: Core script for data preprocessing, model training, evaluation, and visualization

🧰 Technologies Used
- Python 3.8+
- pandas, numpy – Data manipulation
- matplotlib, seaborn – Visualization
- scikit-learn – Modeling and evaluation
  
🚀 How It Works
- Data Loading
- Reads the CSV file and inspects missing values and unique entries
- Preprocessing
- Fills missing values with column means
- Splits features (X) and target (y) where HealthImpactClass is the label
- Modeling
- run_logistic_regression(): Trains and evaluates a Logistic Regression model
- run_random_forest(): Trains and evaluates a Random Forest model
- Evaluation
- Prints accuracy, classification report, and confusion matrix
- Visualizes confusion matrix with a heatmap
- Displays class distribution with a bar chart

  
📊 Sample Output
Accuracy: 0.87
Classification Report:
              precision    recall  f1-score   support
           0       0.89      0.85      0.87       100
           1       0.85      0.89      0.87       100
Confusion Matrix:
[[85 15]
 [11 89]]


📈 Visualizations
- Confusion matrix heatmap
- Bar chart of health impact class distribution

🧪 To Run
pip install pandas numpy matplotlib seaborn scikit-learn
python main.py


📌 Notes
- Ensure the dataset file is named air_quality_health_impact_data.csv and placed in the same directory as the script.
- You can easily swap models or adjust parameters like test_size and random_state.


📬 Contact
Carlos Jamito
carlosjamit@gmail.com

For questions or suggestions, feel free to reach out or open an issue.

Let me know if you'd like to add sections for model comparison, deployment, or future improvements. I can also help you write a requirements.txt or set up a GitHub Actions workflow!


