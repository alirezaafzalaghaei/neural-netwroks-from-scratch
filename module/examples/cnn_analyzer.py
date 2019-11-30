import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


def cnn_analyze(csv_file, n=6, dosent_req: set = set()):
    df = pd.read_csv(csv_file)
    df["history_loss"] = df["history_loss"].apply(lambda x: np.array(eval(x)).round(2))
    losses = [loss[-1] for loss in df['history_loss']]
    df['loss'] = pd.Series(losses, index=df.index).apply(lambda x: '%.2f' % x)

    df['history_score'] = df['history_score'].apply(lambda x: np.array(eval(x)).round(2))
    scores = [score[-1] for score in df['history_score']]
    df['train_score'] = pd.Series(scores, index=df.index).apply(lambda x: '%.3f' % x)
    df["test_score"] = df["test_score"].apply(lambda x: '%.3f' % x)
    best_models_by_test = df.sort_values(['test_score', 'train_score'], ascending=[False, False])[:n]
    best_models_by_test.reset_index(inplace=True)
    history_of_best = best_models_by_test['history_loss']
    best_models_by_test.drop(['history_loss', 'history_score', 'index'], axis=1, inplace=True)
    archs = best_models_by_test['architecture']
    res = []
    for arch in archs:
        a = arch.replace('<', '"<').replace('>', '>"')
        c = []
        for layer in eval(a):
            t = layer['_type']
            cls = t[1 + t.rindex('.'):t.rindex("'")]
            reqs = set()
            if cls == 'Conv2D':
                reqs = {'filters', 'kernel_size', 'padding', 'strides', 'activation'}
            elif cls == 'Dense':
                reqs = {'units', 'activation'}
            elif cls == 'Flatten':
                continue
            elif cls == 'Dropout':
                reqs = {'rate'}
            data = {k: layer[k] for k in layer.keys() & (reqs - dosent_req)}
            params = ', '.join(map(lambda r: '='.join([r[0], str(r[1])]), data.items()))
            c.append(("%s(%s)" % (cls, params)).replace('_size', '').replace('2D', ''))
        res.append(','.join(c))
    best_models_by_test['architecture'] = res
    opt = best_models_by_test['optimizer']
    res = []
    for row in opt:
        a = row.replace('<', '"<').replace('>', '>"')
        row = eval(a)
        t = row['_type']
        cls = t[1 + t.rindex('.'):t.rindex("'")]
        res.append('%s(%.1e)' % (cls, float(row['lr'])))
    best_models_by_test['optimizer'] = res
    best_models_by_test = best_models_by_test[
        ['architecture', 'epochs', 'batch_size', 'optimizer', 'test_score', 'train_score',
         'loss']]

    histories = history_of_best.to_numpy()
    scores = best_models_by_test['test_score']
    return best_models_by_test

#
# def plot_loss(losses, scores):
#     losses = losses[:6]
#     scores = scores[:6]
#     m, n = 2, 3
#     f, axs = plt.subplots(m, n, figsize=(15, 11))
#     c = 0
#     for i in range(m):
#         for j in range(n):
#             hist = losses[c]
#             score = scores[c]
#             axs[i][j].plot(list(range(1, 1 + len(hist))), hist)
#             axs[i][j].set_title("loss: %.2e, score %.2f" % (hist[-1], float(score)))
#             axs[i][j].set_xlabel('iterations')
#             axs[i][j].set_ylabel('Loss')
#             c += 1
#
