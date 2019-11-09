import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


def analyze(csv_file, n=6):
    df = pd.read_csv(csv_file)
    df['history_loss'] = df['history_loss'].apply(lambda x: np.array(eval(x)).round(2))
    losses = [loss[-1] for loss in df['history_loss']]
    df['loss'] = pd.Series(losses, index=df.index).apply(lambda x: '%.2f' % x)

    df['history_score'] = df['history_score'].apply(lambda x: np.array(eval(x)).round(2))
    scores = [score[-1] for score in df['history_score']]
    df['train_score'] = pd.Series(scores, index=df.index).apply(lambda x: '%.2f' % x)

    best_models_by_test = df.sort_values(['test_score', 'train_score'], ascending=[False, False])[:n]
    best_models_by_test.reset_index(inplace=True)
    history_of_best = best_models_by_test['history_loss']
    # indexs = best_models_by_test['index']
    best_models_by_test.drop(['history_loss', 'history_score', 'index'], axis=1, inplace=True)
    # print(df)
    # with pd.option_context('display.max_columns', None, 'display.width', None):
    #     print(best_models_by_test)
    histories = history_of_best.to_numpy()
    scores = best_models_by_test['test_score']
    # print(histories.shape)
    # print(scores.shape)
    plot_loss(histories, scores)
    return best_models_by_test


def plot_loss(losses, scores):
    losses = losses[:6]
    scores = scores[:6]
    m, n = 2, 3
    f, axs = plt.subplots(m, n, figsize=(15, 11))
    c = 0
    for i in range(m):
        for j in range(n):
            hist = losses[c]
            score = scores[c]
            axs[i][j].plot(list(range(1, 1 + len(hist))), hist)  # negative loss
            axs[i][j].set_title("loss: %.2e, score %.2f" % (hist[-1], score))
            axs[i][j].set_xlabel('iterations')
            axs[i][j].set_ylabel('Loss')
            c += 1
