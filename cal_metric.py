from sklearn.metrics import roc_curve, confusion_matrix
import pandas as pd
import numpy as np
import warnings
from confidenceinterval import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from confidenceinterval import npv_score, ppv_score, tpr_score, fpr_score, tnr_score
from confidenceinterval.bootstrap import bootstrap_ci

warnings.filterwarnings("ignore")

def cal_sen(label_list,pre_list):
      matrix = confusion_matrix(label_list,pre_list)
      tp_sum = matrix[1, 1] # True Positive
      fn_sum = matrix[1, 0] # False Negative
      Condition_negative1 = tp_sum + fn_sum + 1e-6
      sen = tp_sum / Condition_negative1
      return sen

def cal_spe(label_list,pre_list):
      matrix = confusion_matrix(label_list,pre_list)
      tn_sum = matrix[0, 0] # True Negative
      fp_sum = matrix[0, 1] # False Positive
      Condition_negative2 = tn_sum + fp_sum + 1e-6
      spe = tn_sum / Condition_negative2
      return spe

def ConfusionResult(label_list,pre_list,pre_prob_list):
      '''
            You can also calculate the CI of precision_score/recall_score/f1_score/ppv_score/npv_score etc.
      '''
      acc, acc_ci = accuracy_score(label_list,pre_list,confidence_level=0.95)
      auc, auc_ci = roc_auc_score(label_list,pre_prob_list,confidence_level=0.95)
      sen, sen_ci= bootstrap_ci(label_list,pre_list,metric=cal_sen,confidence_level=0.95,n_resamples=1000,method='bootstrap_bca',random_state=666)
      spe, spe_ci= bootstrap_ci(label_list,pre_list,metric=cal_spe,confidence_level=0.95,n_resamples=1000,method='bootstrap_bca',random_state=666)
      return auc, auc_ci, acc, acc_ci, sen, sen_ci, spe, spe_ci

def get_data(data,target,cut_off=None):
      '''
            data: the DataFrame containing label and prob
            target: the column name of prob
            cut_off: the specific cutoff to generate predicted value
      '''
      if cut_off == None:
            fpr, tpr, thresholds = roc_curve(data['label'], data[target])
            idx = np.argmax(tpr - fpr)
            cut_off = thresholds[idx]

      data['pred'] = np.zeros(len(data))
      for i in range(0,len(data)):
            if data[target].iloc[i] < cut_off:
                  data['pred'].iloc[i] = 0
            else:
                  data['pred'].iloc[i] = 1

      result = ConfusionResult(data['label'],data['pred'],data[target])
      return result, cut_off


if __name__ == '__main__':
      train_data = pd.read_excel(r'train_data.xlsx')
      valid_data = pd.read_excel(r'valid_data.xlsx')
      test_data = pd.read_excel(r'test_data.xlsx')
      
      target = 'resnet'

      train_result, cut_off = get_data(train_data,target)
      valid_result, _ = get_data(valid_data,target,cut_off)
      test_result, _ = get_data(test_data,target,cut_off)

      print(cut_off)
      print('''
      train auc: {:.3f} {}, train acc: {:.3f} {}
      train sen: {:.3f} {}, train spe: {:.3f} {}
      
      valid auc: {:.3f} {}, valid acc: {:.3f} {}
      valid sen: {:.3f} {}, valid spe: {:.3f} {}

      test auc: {:.3f} {}, test acc: {:.3f} {}
      test sen: {:.3f} {}, test spe: {:.3f} {}
      '''.format(
      train_result[0], train_result[1], train_result[2], train_result[3],
      train_result[4], train_result[5], train_result[6], train_result[7], 
      valid_result[0], valid_result[1], valid_result[2], valid_result[3],
      valid_result[4], valid_result[5], valid_result[6], valid_result[7],
      test_result[0], test_result[1], test_result[2], test_result[3],
      test_result[4], test_result[5], test_result[6], test_result[7]
      ))
