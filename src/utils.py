import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, cv, scoring='balanced_accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Plota a curva de aprendizado de um modelo.

    Parâmetros:
    - model: modelo de machine learning a ser avaliado.
    - X: features do dataset.
    - y: target do dataset.
    - cv: estratégia de cross-validation.
    - scoring: métrica de avaliação.
    - n_jobs: número de jobs em paralelo (usar -1 para utilizar todos os processadores).
    - train_sizes: proporções do dataset de treinamento a serem usadas.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42
    )

    # Calcular média e desvio padrão dos scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plotar as curvas de aprendizado
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Score de Treinamento')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Score de Validação')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.1, color='g')

    plt.title('Curva de Aprendizado')
    plt.xlabel('Tamanho do Dataset de Treinamento')
    plt.ylabel(scoring)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Retornar os valores para análise adicional se necessário
    return train_sizes, train_scores_mean, test_scores_mean
