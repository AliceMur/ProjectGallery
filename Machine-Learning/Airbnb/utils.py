import matplotlib.pyplot as plt

column_names = ["rhood_group","neighbourhood","latitude","longitude","room_type","price","minimum_nights","number_of_reviews","last_review","reviews_per_month","calculated_host_listings_count","availability_365"]

def plot_loss(history):
    h = history.history
    x_lim = len(h['loss'])
    plt.figure(figsize=(8, 8))
    plt.plot(range(x_lim), h['val_loss'], label = 'Validation Loss')
    plt.plot(range(x_lim), h['loss'], label = 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return

def plot_predictions(preds, y_test):
    plt.figure(figsize=(8, 8))
    plt.plot(preds, y_test, 'ro')
    plt.xlabel('Preds')
    plt.ylabel('Labels')
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.plot([-0.5, 0.5], [-0.5, 0.5], 'b--')
    plt.show()
    return

"""
* 'LR' Linear Regression
* 'KNN' K-Neighbors Regressor
* 'DTR' Decision Tree Regressor
* 'RF' Random Forest Regressor
* 'GBM' Gradient Boosting Regressor
"""
def compare_predictions(preds1, preds2, preds3, preds4, preds5,  y_test):
    plt.figure(figsize=(8, 8))
    plt.plot(preds1, y_test, 'ro', label='Linear Regression Model')
    plt.plot(preds2, y_test, 'go', label='K-Neighbors Regressor Model')
    plt.plot(preds3, y_test, 'bo', label='Decision Tree Regressor Model')
    plt.plot(preds4, y_test, 'co', label='Random Forest Regressor Model')
    plt.plot(preds5, y_test, 'mo', label='Gradient Boosting Regressor')
    plt.xlabel('Preds')
    plt.ylabel('Labels')

    y_min = min(min(y_test), min(preds1), min(preds2), min(preds3), min(preds4), min(preds5))
    y_max = max(max(y_test), max(preds1), max(preds2), max(preds3), max(preds4), max(preds5))
    
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.plot([y_min, y_max], [y_min, y_max], 'b--')
    plt.legend()
    plt.show()
    return

def rmse(rmse_scores):
    plt.figure(figsize=(20, 6))
    #5
    color = ['indigo', 'teal', 'olive', 'orangered', 'maroon']
    plt.bar(['Linear Regression','K-Neighbors Regressor',' Decision Tree Regressor', 'Random Forest Regressor',' Gradient Boosting Regressor'], rmse_scores, color = color)
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.title("Model Performance (RMSE)")
    plt.show()