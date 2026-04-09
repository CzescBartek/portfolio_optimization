import shap
import matplotlib.pyplot as plt

def explain_model(model, X_test, name, ticker):

    print("\n Generating SHAP ---")
    
    explainer = shap.TreeExplainer(model.model)
    
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"Influence of features (SHAP Summary) in {name}")
    plt.tight_layout()
    plt.savefig(f'plots/shap_{ticker}.png')
    plt.close()
