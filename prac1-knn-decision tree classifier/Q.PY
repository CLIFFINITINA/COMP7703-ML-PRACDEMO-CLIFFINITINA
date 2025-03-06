import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Read data
classif_data = pd.read_csv(r'E:\ml\w3classif.csv')
regr_data = pd.read_csv(r'E:\ml\w3regr.csv')

# Data visualization - scatter plots for classification and regression data
plt.figure(figsize=(12, 5))

# Classification data
plt.subplot(1, 2, 1)
plt.scatter(classif_data.iloc[:, 0], classif_data.iloc[:, 1], c=classif_data.iloc[:, 2], cmap='viridis')
plt.title("Classification Dataset")

# Regression data
plt.subplot(1, 2, 2)
plt.scatter(regr_data.iloc[:, 0], regr_data.iloc[:, 1], c=regr_data.iloc[:, 1], cmap='coolwarm')
plt.title("Regression Dataset")

plt.show()

# Data processing - split into training and testing sets
classif_train, classif_test = train_test_split(classif_data, test_size=0.3, shuffle=True, random_state=42)
regr_train, regr_test = train_test_split(regr_data, test_size=0.3, shuffle=True, random_state=42)


# k-NN Classification
def knn_classification(k, train_data, test_data):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(train_data.iloc[:, :2], train_data.iloc[:, 2])
    train_acc = accuracy_score(train_data.iloc[:, 2], knn_clf.predict(train_data.iloc[:, :2]))
    test_acc = accuracy_score(test_data.iloc[:, 2], knn_clf.predict(test_data.iloc[:, :2]))
    print(f"k-NN Classification (k={k}) - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
    return knn_clf


# k-NN Regression
def knn_regression(k, train_data, test_data):
    knn_regr = KNeighborsRegressor(n_neighbors=k)
    knn_regr.fit(train_data.iloc[:, :1], train_data.iloc[:, 1])
    train_mse = mean_squared_error(train_data.iloc[:, 1], knn_regr.predict(train_data.iloc[:, :1]))
    test_mse = mean_squared_error(test_data.iloc[:, 1], knn_regr.predict(test_data.iloc[:, :1]))
    print(f"k-NN Regression (k={k}) - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    return knn_regr


# Decision Tree Classification
def decision_tree_classification(max_depth, train_data, test_data):
    dt_clf = DecisionTreeClassifier(max_depth=max_depth)
    dt_clf.fit(train_data.iloc[:, :2], train_data.iloc[:, 2])
    train_acc = accuracy_score(train_data.iloc[:, 2], dt_clf.predict(train_data.iloc[:, :2]))
    test_acc = accuracy_score(test_data.iloc[:, 2], dt_clf.predict(test_data.iloc[:, :2]))
    print(
        f"Decision Tree Classification (max_depth={max_depth}) - Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")
    return dt_clf


# Decision Tree Regression
def decision_tree_regression(max_depth, train_data, test_data):
    dt_regr = DecisionTreeRegressor(max_depth=max_depth)
    dt_regr.fit(train_data.iloc[:, :1], train_data.iloc[:, 1])
    train_mse = mean_squared_error(train_data.iloc[:, 1], dt_regr.predict(train_data.iloc[:, :1]))
    test_mse = mean_squared_error(test_data.iloc[:, 1], dt_regr.predict(test_data.iloc[:, :1]))
    print(f"Decision Tree Regression (max_depth={max_depth}) - Train MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}")
    return dt_regr


# Run k-NN Classification
knn_clf = knn_classification(k=3, train_data=classif_train, test_data=classif_test)

# Run k-NN Regression
knn_regr = knn_regression(k=3, train_data=regr_train, test_data=regr_test)

# Run Decision Tree Classification
dt_clf = decision_tree_classification(max_depth=5, train_data=classif_train, test_data=classif_test)

# Run Decision Tree Regression
dt_regr = decision_tree_regression(max_depth=5, train_data=regr_train, test_data=regr_test)


# Function to plot the regression model
def plot_regression(X, y, model):
    # Reshape X to be 2D for the model prediction
    X_reshaped = X.values.reshape(-1, 1)

    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, model.predict(X_reshaped), color='red', label='Predicted function')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.title('Regression with k-NN')
    plt.show()


# Plot regression predictions
plot_regression(regr_train.iloc[:, 0], regr_train.iloc[:, 1], knn_regr)

# Experiment with different k values for k-NN Classification
k_values = range(1, 21)
train_acc_list = []
test_acc_list = []
for k in k_values:
    knn_clf = knn_classification(k, classif_train, classif_test)
    train_acc_list.append(accuracy_score(classif_train.iloc[:, 2], knn_clf.predict(classif_train.iloc[:, :2])))
    test_acc_list.append(accuracy_score(classif_test.iloc[:, 2], knn_clf.predict(classif_test.iloc[:, :2])))

# Plot k values' effect on k-NN Classification accuracy
plt.plot(k_values, train_acc_list, label='Train Accuracy')
plt.plot(k_values, test_acc_list, label='Test Accuracy')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Effect of k on k-NN Classification')
plt.show()

# Experiment with different k values for k-NN Regression
train_mse_list = []
test_mse_list = []
for k in k_values:
    knn_regr = knn_regression(k, regr_train, regr_test)
    train_mse_list.append(mean_squared_error(regr_train.iloc[:, 1], knn_regr.predict(regr_train.iloc[:, :1])))
    test_mse_list.append(mean_squared_error(regr_test.iloc[:, 1], knn_regr.predict(regr_test.iloc[:, :1])))

# Plot k values' effect on k-NN Regression
plt.plot(k_values, train_mse_list, label='Train MSE')
plt.plot(k_values, test_mse_list, label='Test MSE')
plt.xlabel('k Value')
plt.ylabel('MSE')
plt.legend()
plt.title('Effect of k on k-NN Regression')
plt.show()

# Experiment with different max depth values for Decision Tree Classification
depths = range(1, 21)
train_acc_depth = []
test_acc_depth = []
for depth in depths:
    dt_clf = decision_tree_classification(max_depth=depth, train_data=classif_train, test_data=classif_test)
    train_acc_depth.append(accuracy_score(classif_train.iloc[:, 2], dt_clf.predict(classif_train.iloc[:, :2])))
    test_acc_depth.append(accuracy_score(classif_test.iloc[:, 2], dt_clf.predict(classif_test.iloc[:, :2])))

# Plot max depth's effect on Decision Tree Classification accuracy
plt.plot(depths, train_acc_depth, label='Train Accuracy')
plt.plot(depths, test_acc_depth, label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Effect of Max Depth on Decision Tree Classification')
plt.show()

# Experiment with different max depth values for Decision Tree Regression
train_mse_depth = []
test_mse_depth = []
for depth in depths:
    dt_regr = decision_tree_regression(max_depth=depth, train_data=regr_train, test_data=regr_test)
    train_mse_depth.append(mean_squared_error(regr_train.iloc[:, 1], dt_regr.predict(regr_train.iloc[:, :1])))
    test_mse_depth.append(mean_squared_error(regr_test.iloc[:, 1], dt_regr.predict(regr_test.iloc[:, :1])))

# Plot max depth's effect on Decision Tree Regression
plt.plot(depths, train_mse_depth, label='Train MSE')
plt.plot(depths, test_mse_depth, label='Test MSE')
plt.xlabel('Max Depth')
plt.ylabel('MSE')
plt.legend()
plt.title('Effect of Max Depth on Decision Tree Regression')
plt.show()
