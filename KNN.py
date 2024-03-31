import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Haberman's Survival dataset
df = pd.read_csv('/mnt/data/haberman.csv', header=None)
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values   # Labels

# Modify the evaluation function to accurately capture requirements
def evaluate_knn(X_train, X_test, y_train, y_test, k_values, p_values):
    results = {k: {p: {'train_error': [], 'test_error': []} for p in p_values} for k in k_values}
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

            results[k][p]['train_error'].append(train_error)
            results[k][p]['test_error'].append(test_error)
    return results

# Main experiment
k_values = [1, 3, 5, 7, 9]
p_values = [1, 2, np.inf]
n_trials = 100

# Initialize results storage
all_results = {k: {p: {'train_error': [], 'test_error': []} for p in p_values} for k in k_values}

for _ in range(n_trials):
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Evaluate the k-NN classifier
    trial_results = evaluate_knn(X_train, X_test, y_train, y_test, k_values, p_values)

    # Accumulate results from this trial
    for k in k_values:
        for p in p_values:
            all_results[k][p]['train_error'].extend(trial_results[k][p]['train_error'])
            all_results[k][p]['test_error'].extend(trial_results[k][p]['test_error'])

# Compute average errors and prepare for plotting
plot_data = {'k': [], 'p': [], 'train_error': [], 'test_error': []}
for k, p_dict in all_results.items():
    for p, error_dict in p_dict.items():
        avg_train_error = np.mean(error_dict['train_error'])
        avg_test_error = np.mean(error_dict['test_error'])
        plot_data['k'].append(k)
        plot_data['p'].append(p)
        plot_data['train_error'].append(avg_train_error)
        plot_data['test_error'].append(avg_test_error)

# Plotting results
fig, ax = plt.subplots(figsize=(10, 6))
for p in p_values:
    p_data = [d for i, d in enumerate(plot_data['p']) if d == p]
    train_errors = [plot_data['train_error'][i] for i, d in enumerate(plot_data['p']) if d == p]
    test_errors = [plot_data['test_error'][i] for i, d in enumerate(plot_data['p']) if d == p]
    ax.plot(k_values, train_errors, marker='o', linestyle='-', label=f'Train Error, p={p}')
    ax.plot(k_values, test_errors, marker='x', linestyle='--', label=f'Test Error, p={p}')

ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Error')
ax.set_title('k-NN Classifier Errors for Different k and p Values')
ax.legend()

plt.show()
