import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Haberman's Survival dataset
df = pd.read_csv('haberman.csv', header=None)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Function to evaluate k-NN classifier
def evaluate_knn(X_train, X_test, y_train, y_test, k_values, p_values):
    results = []
    for k in k_values:
        for p in p_values:
            # Initialize and train the k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=k, p=p)
            knn.fit(X_train, y_train)

            # Make predictions on training and test sets
            y_train_pred = knn.predict(X_train)
            y_test_pred = knn.predict(X_test)

            # Compute errors
            train_error = 1 - accuracy_score(y_train, y_train_pred)
            test_error = 1 - accuracy_score(y_test, y_test_pred)

            results.append({
                'k': k,
                'p': p,
                'train_error': train_error,
                'test_error': test_error
            })
    return results

# Main experiment
k_values = [1, 3, 5, 7, 9]
p_values = [1, 2, np.inf]
n_trials = 100
average_results = []

for _ in range(n_trials):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Evaluate the k-NN classifier
    trial_results = evaluate_knn(X_train, X_test, y_train, y_test, k_values, p_values)

    # Accumulate results
    if not average_results:
        average_results = trial_results
    else:
        for i in range(len(trial_results)):
            average_results[i]['train_error'] += trial_results[i]['train_error']
            average_results[i]['test_error'] += trial_results[i]['test_error']

# Compute average errors
for result in average_results:
    result['train_error'] /= n_trials
    result['test_error'] /= n_trials

# Print results
print("Average Results:")
for result in average_results:
    print(f"k: {result['k']}, p: {result['p']}, Train Error: {result['train_error']}, Test Error: {result['test_error']}")
# Save results to an Excel file
df_results = pd.DataFrame(average_results)
df_results.to_excel('results.xlsx', index=False)