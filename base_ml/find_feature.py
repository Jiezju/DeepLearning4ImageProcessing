'''
特征组合对加速特征学习的作用
'''

from sklearn.utils import shuffle
import matplotlib as mpl
from cycler import cycler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

mpl.rcParams['axes.prop_cycle'] = cycler(color='rb')

np.random.seed(42)
pseudoNum1 = 300
pseudoNum2 = 300
np_pho1 = 4.5 + np.random.rand(pseudoNum1) * 2
np_pho2 = 0.5 + np.random.rand(pseudoNum2) * 2
np_theta1 = np.random.rand(pseudoNum1) * 360 / 2 * np.pi
np_theta2 = np.random.rand(pseudoNum2) * 360 / 2 * np.pi

np_x1 = np_pho1 * np.cos(np_theta1)
np_y1 = np_pho1 * np.sin(np_theta1)
np_x2 = np_pho2 * np.cos(np_theta2)
np_y2 = np_pho2 * np.sin(np_theta2)

pd_circ = shuffle(pd.DataFrame({
    "X": list(np_x1) + list(np_x2),
    "Y": list(np_y1) + list(np_y2),
    "label": ["Class1" for x in range(pseudoNum1)] + ["Class2" for x in range(pseudoNum2)]
}), random_state=0).reset_index().drop(['index'], axis=1)
pd_circ0 = pd_circ.copy()
pd_circ.head()
# %%
for sub in ["Class1", "Class2"]:
    pd_sub = pd_circ[pd_circ['label'] == sub]
    plt.plot(pd_sub["X"], pd_sub["Y"], ".", label=sub)

plt.legend()
# %%
sns.pairplot(pd_circ, hue="label")
# %%
pd_circ['X_add_Y'] = pd_circ['X'] + pd_circ['Y']
pd_circ['X_time_Y'] = pd_circ['X'] * pd_circ['Y']
pd_circ['X2_add_Y2'] = pd_circ['X'] * pd_circ['X'] + pd_circ['Y'] * pd_circ['Y']
sns.pairplot(pd_circ, hue="label")
plt.show()
