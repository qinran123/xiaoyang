#!/usr/bin/env python
# coding: utf-8

# **#项目背景**
# 
# 心血管疾病是全球第一大死亡原因，估计每年夺走1790万人的生命，占全世界死亡人数的31%。
# 
# 心力衰竭是心血管病引起的常见事件，此数据集包含12个特征，可用于预测心力衰竭的死亡率。
# 
# 通过采取全人口战略，解决行为风险因素，如吸烟、不健康饮食和肥胖、缺乏身体活动和有害使用酒精，可以预防大多数心血管疾病。
# 
# 心血管疾病患者或心血管高危人群(由于存在高血压、糖尿病、高脂血症等一个或多个危险因素或已有疾病)需要早期发现和管理，机器学习模型可以提供很大帮助。

# 解压数据集到work目录下

# In[4]:


get_ipython().system('unzip -oq data/data106584/心力衰竭预测.zip -d work/')


# In[5]:


get_ipython().system('tree work/ -d')


# **#探索性数据分析**
# 
# 读取文件，描述性统计
# 

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly as py 
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix


df =pd.read_csv("work/heart_failure_clinical_records_dataset.csv")
#df=pd.DataFrame(df)
df.head()
df.info()
df.describe()


# #**数据介绍**
# 
# ![查看数据集内容(列名和特征值）](https://ai-studio-static-online.cdn.bcebos.com/1a061f6cf6bf4d73afa26019b62d658b7b99e8372f684e888799072bb97093a5)
# 
# （age:年龄，anaemia:红细胞或血红蛋白减少，creatinine_phosphokinase	：CPK酶，diabetes	：糖尿病，high_blood_pressure：射血分数，high_blood_pressure：高血压，platelets：血小板，serum_creatinine：血肌酐水平，sex：性别，smoking	：是否吸烟，DEATH_EVENT：是否死亡）
# 

# In[7]:


death_num = df['DEATH_EVENT'].value_counts() 
death_num = death_num.reset_index()
print(death_num)


# In[8]:


fig = px.pie(death_num, names='index', values='DEATH_EVENT')
fig.update_layout(title_text='目标变量DEATH_EVENT的分布')  
py.offline.plot(fig, filename='.目标变量DEATH_EVENT的分布.html')


# In[9]:


df = df.rename(columns={'smoking':'smk','diabetes':'dia','anaemia':'anm','platelets':'plt','high_blood_pressure':'hbp','creatinine_phosphokinase':'cpk','ejection_fraction':'ejf','serum_creatinine':'scr','serum_sodium':'sna','DEATH_EVENT':'death'})


# 将特征重命名以便后续操作

# In[10]:


df.head()


# In[11]:


df['sex'] = df['sex'].apply(lambda x: 'Female' if x==0 else 'Male')
df['smk'] = df['smk'].apply(lambda x: 'No' if x==0 else 'Yes')
df['chk'] = 1
df['dia'] = df['dia'].apply(lambda x: 'No' if x==0 else 'Yes')
df['anm'] = df['anm'].apply(lambda x: 'No' if x==0 else 'Yes')
df['hbp'] = df['hbp'].apply(lambda x: 'No' if x==0 else 'Yes')
df['death'] = df['death'].apply(lambda x: 'No' if x==0 else 'Yes')
df.info()


# In[12]:


fig,ax = plt.subplots(3,2,figsize=[10,10])
num_features_set1 = ['age', 'scr','sna']
num_features_set2 = ['plt','ejf','cpk']
for i in range(0,3):  
    sns.boxenplot(df[num_features_set1[i]],ax=ax[i,0],color='steelblue')
    sns.boxenplot(df[num_features_set2[i]],ax=ax[i,1],color='steelblue')


# 使用boxenplot直观显示 （age,plt,scr,ejf,sna,cpk)这些数据，中间的线显示平均值 如 age
# ![](https://ai-studio-static-online.cdn.bcebos.com/3858723a01504429aacd08cfa1f3843c71993f1091b74b82a8e39ca6a5cc5285)
# 

# In[13]:


fig = plt.subplots(figsize=[10,6])
bar1 = df.smk.value_counts().values
bar2 = df.hbp.value_counts().values
bar3 = df.dia.value_counts().values
bar4 = df.anm.value_counts().values
ticks = np.arange(0,3, 2)
width = 0.3
plt.bar(ticks, bar1, width=width, color='teal', label='smoker')
plt.bar(ticks+width, bar2, width=width, color='darkorange', label='high blood pressure')
plt.bar(ticks+2*width, bar3, width=width, color='limegreen', label='diabetes')
plt.bar(ticks+3*width, bar4, width=width, color='tomato', label='anaemic')
plt.xticks(ticks+1.5*width, ['Yes', 'No'])
plt.ylabel('Number of patients')
plt.legend()


# ![](https://ai-studio-static-online.cdn.bcebos.com/98786a84522444eca2e3b36d4695eff89d5972c9fb914410951d1b1c045b51b8)用图查看不同的人的患者人数
# 

# In[14]:


sns.pairplot(df[['plt', 'ejf', 'cpk', 'scr', 'sna', 'death']],
hue='death', 
palette='husl', corner=True)


# In[15]:


import paddle
import numpy as np
import paddle.vision.transforms as T
class MyImageNetDataset(paddle.io.Dataset):
    def __init__(self, index):
        super(MyImageNetDataset, self).__init__()
        self=df
    def __getitem__(self, index):
        image=df.iloc[:,index]
        label=df.iloc[0,index]
        return  image,label
    def __len__(self):
        return (len(df))


# In[16]:


train_dataset= MyImageNetDataset(0)
print(len(train_dataset))
print(train_dataset.__getitem__(0))


# In[16]:





# In[17]:


df.describe()


# In[18]:


import xgboost as xgb
import pandas as pd
import numpy as np
import logging
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
df['death'] = df['death'].apply(lambda x: 0 if x=='No' else 1)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S')
def get_kind(x: pd.Series, diff_limit: int = 8):
    x = x.astype('str')
    x = x.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    x.dropna(inplace=True)
    if x.nunique() > diff_limit:
        print(x.nunique())
        kind = 'numeric'
    else:
        kind = 'categorical'
    return kind

def check_data_y(X):
    """
    检查数据结构，数据预测变量为 0,1，并以“y”命名
    """
    if 'y' not in X.columns:
        logging.error('未检测到"y"变量，请将预测变量命名改为"y"')


class Feature_select(BaseEstimator, TransformerMixin):
    def __init__(self,
                 num_list: list = None,
                 cate_list: list = None,
                 num_method: str = 'sys',
                 cate_method: str = 'sys',
                 diff_num: int = 10,
                 pos_label: str = 1,
                 show_df: bool = False):
        self.num_list = num_list
        self.cate_list = cate_list
        self.num_method = num_method
        self.cate_method = cate_method
        self.diff_num = diff_num
        self.pos_label = pos_label
        print(self.pos_label)
        self.show_df = show_df
        self.select_list = []

    def fit(self, X, y=None):
        X = X.copy()
        from scipy import stats
        if self.num_list is None:
            self.num_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'numeric':
                    self.num_list.append(col)
        print(self.num_list)
        if self.cate_list is None:
            self.cate_list = []
            for col in X.columns:
                kind = get_kind(x=X[col], diff_limit=self.diff_num)
                if kind == 'categorical':
                    self.cate_list.append(col)
        print(self.cate_list)
        X['y'] = y
        print(X['y'])
        print("--------------------!!!!")
        print(type(X['y']))
        print(str(X['y'])==self.pos_label)
        yes = X[X['y']==self.pos_label]
        yes.reset_index(drop=True, inplace=True)
        no = X[X['y'] != self.pos_label]
        no.reset_index(drop=True, inplace=True)
        print("------------------------------")
        print(yes)
        print("-------------------------------")
        print(no)
        del X['y']
        sys_cate_list, kf_list, kf_p_list = [], [], []
        sys_num_list, t_list, p_value_list, anova_f_list, anova_p_list = [], [], [], [], []
        if self.cate_method == 'sys' or self.show_df is True:
            for obj in self.cate_list:
                value_list = list(X[obj].unique())
                value_sum = 0
                for value in value_list:
                    support_yes = (yes[yes[obj] == value].shape[0] + 1) / (yes.shape[0] + 1)
                    support_no = (no[no[obj] == value].shape[0] + 1) / (no.shape[0] + 1)
                    confidence_yes = support_yes / (support_yes + support_no)
                    value_sum += abs(2 * confidence_yes - 1) * (X[X[obj] == value].shape[0] / X.shape[0])
                sys_cate_list.append(value_sum)
                if value_sum >= 0.1:
                    self.select_list.append(obj)
            
        if self.cate_method == 'kf' or self.show_df is True:
            for obj in self.cate_list:
                df_obj = pd.get_dummies(X[obj], prefix=obj)
                df_obj['result'] = y
                #print(df_obj)
                df_obj = df_obj.groupby('result').sum()
                #print(df_obj)
                obs = df_obj.values
                #print(obs)
                kf = stats.chi2_contingency(obs)
                '''
                chi2: The test statistic
                p: p-value
                dof: Degrees of freedom
                expected: The expected frequencies, based on the marginal sums of the table.
                '''
                chi2, p, dof, expect = kf
                kf_list.append(chi2)
                kf_p_list.append(p)
                #print(p)
                if p < 0.05:
                    self.select_list.append(obj)
            print(self.select_list)

        if self.num_method == 'sys' or self.show_df is True:
            for num in self.num_list:
                mean_c1 = no[num].mean()
                std_c1 = no[num].std()
                mean_c2 = yes[num].mean()
                std_c2 = yes[num].std()
                value_sum = abs(mean_c1 - mean_c2) / (std_c1 + std_c2) * 2
                sys_num_list.append(value_sum)
                if value_sum >= 0.1:
                    self.select_list.append(num)

        if self.num_method == 't' or self.show_df is True:
            for num in self.num_list:
                t_t, t_p = stats.ttest_ind(yes[num], no[num], equal_var=False, nan_policy='omit')  # 'omit'忽略nan值执行计算
                t_list.append(t_t)
                p_value_list.append(t_p)
                if t_p < 0.05:
                    self.select_list.append(num)
                # print('attr=%s, t=%.5f, p=%.5f' % (num, t, p_value))
        if self.num_method == 'anova' or self.show_df is True:
            for num in self.num_list:
                #print(yes[num],no[num])
                anova_f, anova_p = stats.f_oneway(yes[num], no[num])
                anova_f_list.append(anova_f)
                anova_p_list.append(anova_p)
                print('attr=%s, anova_f=%.5f, anova_p=%.5f' % (num, anova_f, anova_p))
                if anova_p < 0.05:
                    self.select_list.append(num)
        if self.show_df is True:
            dic1 = {'categorical': self.cate_list, 'importance_': sys_cate_list, 'Kf-Value': kf_list,
                    'Kf-P-Value': kf_p_list}
            df = pd.DataFrame(dic1, columns=['categorical', 'importance_', 'Kf-Value', 'Kf-P-Value'])
            df.sort_values(by='Kf-P-Value', inplace=True)
            print(df)
            dic2 = {'numeric': self.num_list, 'importance_': sys_num_list, 'T-Value': t_list, 'P-value': p_value_list,
                    'Anova-F-Value': anova_f_list, 'Anova-P-value': anova_p_list}
            df = pd.DataFrame(dic2,
                              columns=['numeric', 'importance_', 'T-Value', 'P-value', 'Anova-F-Value',
                                       'Anova-P-value'])
            df.sort_values(by='Anova-P-value', inplace=True)
            print(df)
            print(self)
        self.select_list = list(set(self.select_list))
        print('After select attr:', self.select_list)
        return self

    def transform(self, X):
        X = X.copy()
        logging.info('attr select success!')
        return X[self.select_list]


# **#特征选择**
# 
# 将死亡这类标签删除 作为X 死亡标签作为y，由于 y是分类变量，我们使用卡方鉴定，X是数值型数据使用方差分析
# 特征筛选出以下特征
# 
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/278f575e67ce49c58fc6111f9320bad6ab1e114c63404d5a849ca91feaf65c3c)
# 

# In[19]:


df.describe()
X = df.drop('death', axis=1)
y = df['death'] 
#print(X,y)
fs = Feature_select(num_method='anova', cate_method='kf') 
X_selected = fs.fit_transform(X, y)
X_selected.head()


# **#模型训练**
# 
# 1.划分训练集和测试集

# In[20]:


Features = X_selected.columns
X = df[Features] 
y = df["death"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=2020)


# 2.**训练模型（lgb)，模型评估**
# 
# 评估函数结果：0.74

# In[21]:


import paddle
lgb_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', random_state=1)

parameters = {'max_depth': np.arange(2, 20, 1) }
GS = GridSearchCV(lgb_clf, param_grid=parameters, cv=10, scoring='f1', n_jobs=-1)  
GS.fit(X_train, y_train)  
print(GS.best_params_) 
print(GS.best_score_) 
# 测试集
test_pred = GS.best_estimator_.predict(X_test)
# F1-score
print("F1_score of LGBMClassifier is : ", round(f1_score(y_true=y_test, y_pred=test_pred),2)) 


# **#总结与升华**
# 
# 心力衰竭预测的实现可以极大程度帮助医生临床问诊，给医生提供一个科学的判断依据，该项目还能在模型准确率上进行改进

# In[22]:


# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
get_ipython().system('ls /home/aistudio/work')


# In[23]:


# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
get_ipython().system('mkdir /home/aistudio/external-libraries')
get_ipython().system('pip install beautifulsoup4 -t /home/aistudio/external-libraries')


# In[24]:


# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
