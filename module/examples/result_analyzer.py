import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


def analyze(csv_file):
    n = 9
    df = pd.read_csv(csv_file)
    # print(df.iloc[250, :])
    # return
    df['history'] = df['history'].apply(
        lambda x: np.fromstring(x.replace('\n', '').replace('[', '').replace(']', '').replace('  ', ' '), sep=','))
    losses = [np.abs(loss[-1:]).mean() for loss in df['history']]
    df['loss'] = pd.Series(losses, index=df.index).apply(lambda x: '%.2f' % x)
    best_models_by_test = df.sort_values(['test_score', 'loss'], ascending=[False, True])[:n]
    best_models_by_test.reset_index(inplace=True)
    history_of_best = best_models_by_test['history']
    indexs = best_models_by_test['index']
    print(indexs)
    best_models_by_test.drop(['history', 'index'], axis=1, inplace=True)

    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(best_models_by_test)
    return best_models_by_test, history_of_best


def plot_loss(losses, scores):
    m, n = 3, 3
    f, axs = plt.subplots(m, n, figsize=(15, 11))
    c = 0
    for i in range(m):
        for j in range(n):
            hist = losses[c]
            score = scores[c]
            axs[i][j].plot(list(range(1, 1 + len(hist))), hist)
            axs[i][j].set_title("loss: %.2e, score %.2f" % (hist[-1], score))
            axs[i][j].set_xlabel('iterations')
            axs[i][j].set_ylabel('Log(loss)')
            c += 1
