from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the circle separator dataset from the provided file path
data_path = 'circle_separator.txt'
circle_data = pd.read_csv(data_path, sep=" ", header=None)

# The dataset consists of two features and one label column
X_circle = circle_data.iloc[:, :-1].values  # Features
y_circle = circle_data.iloc[:, -1].values   # Labels

# Check the shape of the data to ensure it's loaded correctly
X_circle.shape, y_circle.shape

# Define the k values and distance metrics as per the homework's instructions
k_values = [1, 3, 5, 7, 9]
distance_metrics = {'manhattan': 1, 'euclidean': 2, 'chebyshev': np.inf}
n_trials = 100

# Function to evaluate the k-NN classifier for the given dataset
def evaluate_knn(X, y, k_values, distance_metrics, n_trials):
    results = {
        k: {metric: {'train_error': [], 'test_error': []} for metric in distance_metrics} 
        for k in k_values
    }
    
    for trial in range(n_trials):
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=trial)

        for k in k_values:
            for metric, p in distance_metrics.items():
                # Initialize and train the k-NN classifier
                knn = KNeighborsClassifier(n_neighbors=k, p=p)
                knn.fit(X_train, y_train)

                # Make predictions and compute errors
                y_train_pred = knn.predict(X_train)
                y_test_pred = knn.predict(X_test)
                train_error = 1 - accuracy_score(y_train, y_train_pred)
                test_error = 1 - accuracy_score(y_test, y_test_pred)

                results[k][metric]['train_error'].append(train_error)
                results[k][metric]['test_error'].append(test_error)

    # Calculate average errors and their differences
    avg_results = {
        k: {metric: {
            'avg_train_error': np.mean(values['train_error']),
            'avg_test_error': np.mean(values['test_error']),
            'error_difference': np.mean(values['test_error']) - np.mean(values['train_error'])
        } for metric, values in metrics.items()} for k, metrics in results.items()
    }

    return avg_results

# Evaluate the KNN classifier with the "circle separator" dataset
circle_evaluation_results = evaluate_knn(X_circle, y_circle, k_values, distance_metrics, n_trials)

# Function to print the evaluation results in a formatted manner
def print_evaluation_results(results):
    for k in sorted(results.keys()):
        print(f"\nResults for k={k}:")
        for metric in results[k]:
            avg_train_error = results[k][metric]['avg_train_error']
            avg_test_error = results[k][metric]['avg_test_error']
            error_difference = results[k][metric]['error_difference']
            print(f"  {metric.capitalize()} Distance:")
            print(f"    Average Train Error: {avg_train_error:.3f}")
            print(f"    Average Test Error: {avg_test_error:.3f}")
            print(f"    Error Difference: {error_difference:.3f}")

# Print the formatted results
print_evaluation_results(circle_evaluation_results)

# Function to plot average test error for different k values and distance metrics
def plot_errors(results):
    plt.figure(figsize=(12, 8))

    for metric, _ in distance_metrics.items():
        k_values = list(results.keys())
        test_errors = [results[k][metric]['avg_test_error'] for k in k_values]

        plt.plot(k_values, test_errors, marker='o', linestyle='-', label=f'{metric.capitalize()} Distance')

    plt.title('Average Test Error for Different k Values and Distance Metrics')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Average Test Error')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.show()

plot_errors(circle_evaluation_results)
