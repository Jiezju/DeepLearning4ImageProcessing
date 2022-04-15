from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 假如真实情况1万个病人，有10个是有病的
np_real = np.array([0.0 for i in range(9990)] + [1.0 for i in range(10)], dtype=bool)

# 预测1， 全预测为没问题，准确率 99.90%
np_pred_allf = 0.1 * np.random.random(10000)

# 预测2:，准确预测使用情况，准确率 100%
np_pred_true = np_pred_allf.copy()
np_pred_true[-10:] = 0.99

fpr, tpr, thresholds = roc_curve(np.array(np_real, dtype=int), np_pred_allf)
AUC_value = auc(fpr, tpr)

fpr2, tpr2, thresholds2 = roc_curve(np.array(np_real, dtype=int), np_pred_true)
AUC_value2 = auc(fpr2, tpr2)

# 虽然准确率差不多，但是AUC值差异巨大
print(AUC_value, AUC_value2)

fpr2 = np.array([0] + list(fpr2))
tpr2 = np.array([0] + list(tpr2))

plt.figure(figsize=(5,5))
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve 1 (area = %0.2f)' % AUC_value)
plt.plot(fpr2, tpr2, color='g',
         lw=lw, label='ROC curve 2 (area = %0.2f)' % AUC_value2)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
