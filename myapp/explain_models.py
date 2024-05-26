import os
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from .utils import load_and_prepare_data

# Matplotlib'i etkileşimli olmayan moda al
plt.switch_backend('Agg')

def explain_model():
    try:
        print("Veriler yükleniyor...")
        X_train, X_test, y_train, y_test = load_and_prepare_data('datasets/cleaned_data.csv')
        
        print("Model eğitiliyor...")
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        print("SHAP değerleri hesaplanıyor...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        print("SHAP grafiği oluşturuluyor...")
        shap.summary_plot(shap_values, X_test)
        # SHAP grafiğini statik dosyalar dizinine kaydedin
        static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        shap_plot_path = os.path.join(static_dir, 'shap_summary_plot.png')
        plt.savefig(shap_plot_path)
        plt.close()

        print(f"SHAP grafiği kaydedildi: {shap_plot_path}")
        return shap_plot_path

    except Exception as e:
        print(f"Bir hata oluştu: {e}")
        return None