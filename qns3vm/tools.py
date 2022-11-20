import numpy as np
import pandas as pd

def classification_error(preds, L_test) -> float:
    error = 0.0
    for i in range(len(preds)):
        error += float(abs(int(preds[i])-int(L_test[i]))) / 2.0
    error /= len(preds)
    return error


def plot_distribution(ax,clf,X_train_l, L_train_l, X_test, L_test):
    assert np.array(X_train_l).ndim >= 2
    N_l = len(X_train_l)
    N_t = len(X_test)

    X = X_train_l + X_test
    ndim = np.array(X).shape[1]
    x_bg = list()
    for d in range(ndim):
        xmin = np.amin(np.array(X)[:,d])
        xmax = np.amax(np.array(X)[:,d])
        xmargin = 0.1*(xmax-xmin)
        xmin = xmin - xmargin
        xmax = xmax + xmargin
        if d < 2:
            x_bg.append(np.linspace(xmin,xmax,100))
        else:
            x_bg.append(np.mean([xmin,xmax]))
    x_bg = np.meshgrid(*x_bg)
    X_bg = list()
    for d in range(ndim):
        X_bg.append(x_bg[d].ravel())
    X_bg = np.array(X_bg).T
    L_bg = clf.predict(X_bg)

    Zs = [list(zip(X_bg,L_bg)), list(zip(X_test,L_test)), list(zip(X_train_l,L_train_l))]

    for i, z in enumerate(Zs):
        X,y = zip(*z)
        X_ = np.array(X)
        print(X_.shape)
        df = pd.DataFrame(X_[:,:2],columns=['cordinate_x','cordinate_y'])
        df['label'] = pd.Series(y)

        labels = sorted(df['label'].unique().tolist())

        if i == 2:
            for l in labels:
                df_l = df[df['label']==l]
                ax.scatter(df_l.cordinate_x, df_l.cordinate_y, marker='x', s=100, label=f'{l}-labeled')
        elif i == 1:
            for l in labels:
                df_l = df[df['label']==l]
                ax.scatter(df_l.cordinate_x, df_l.cordinate_y, marker='o', alpha=0.2, label=f'{l}-tested')
        elif i == 0:
            for l in labels:
                df_l = df[df['label']==l]
                ax.scatter(df_l.cordinate_x, df_l.cordinate_y, alpha=0.1)
    ax.legend(loc=(1,0),ncol=2)
    ax.set_xlabel('v1')
    ax.set_ylabel('v2')

