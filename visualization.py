import matplotlib.pyplot as plt
from sckit_learn import y_test, predictions

plt.scatter(y_test,predictions)
plt.plot([y_test.min(),y_test.max()],[predictions.min(),predictions.max()],linestyle='--')
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.title('Actual vs Predicted')
plt.savefig('actual_vs_predicted.png')
plt.show()

residuals = y_test - predictions
plt.scatter(predictions,residuals)
plt.axhline(0)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residuals plot')
plt.savefig('residuals_plot.png')
plt.show()