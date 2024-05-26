from django.shortcuts import render
from .utils import load_and_prepare_data, train_model, evaluate_model
from .explain_models import explain_model

def index(request):
    X_train, X_test, y_train, y_test = load_and_prepare_data('datasets/cleaned_data.csv')
    model = train_model(X_train, y_train)
    mse, r2 = evaluate_model(model, X_test, y_test)
    context = {'mse': mse, 'r2': r2}
    return render(request, 'index.html', context)

def explain(request):
    shap_plot_path = explain_model()
    context = {'shap_plot_path': shap_plot_path}
    return render(request, 'explain.html', context)


