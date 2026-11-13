import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import (GroupKFold, KFold, cross_val_predict,
                                     GridSearchCV)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import (Ridge, ElasticNet, Lasso, BayesianRidge,
                                  HuberRegressor, LinearRegression)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import nnls
from scipy import stats


GROUP_MODE   = 'combo'  #按唯一特征组合分组                      

N_TUNE_FOLDS = 5         # 网格调优所用的 GroupKFold 折数
N_OUTER      = 8         # 嵌套评估外层折数
N_INNER      = 5         # 嵌套评估内层折数(学NNLS权重)
NESTED_TUNE  = True      # 外层评估的每一折内部重新调参

ALPHA        = 0.10      # 90%预测区间


N_JOBS       = int(os.environ.get('CFPA_N_JOBS', min(4, os.cpu_count() or 1)))
N_JOBS_INNER = 1         # 树模型内部并行度置 1


#数据存放在Data
_HERE = os.path.dirname(os.path.abspath(__file__))  #dirname取父目录

def _resolve_data_dir():
    """定位Data目录"""
    cands = [os.path.join(_HERE, '..', 'Data'),
             os.path.join(_HERE, 'Data'),
             os.path.join(os.getcwd(), 'Data')]
    try:
        for c in cands:
            p = os.path.abspath(c)
            if os.path.isdir(p):
                return p
    except Exception:
        raise RuntimeError('定位不到Data目录')
    # return os.path.abspath(os.path.join(_HERE, '..', 'Data'))

DATA_DIR = _resolve_data_dir()


MODEL_DIR  = os.path.join(DATA_DIR, '集成建模')                      #集成模型与训练结果
EXT_DIR    = os.path.join(DATA_DIR, '外部验证')                      #外部验证结果
SHAP_DIR   = os.path.join(DATA_DIR, 'Shap解释图')                    #SHAP 图
PI_OOF_DIR = os.path.join(DATA_DIR, 'OOF残差预测区间方法')            #共形
PI_MEM_DIR = os.path.join(DATA_DIR, '基学习器差异预测区间方法')        #成员分歧
PI_VQ_DIR  = os.path.join(DATA_DIR, '基学习器_OOF残差复合预测区间')    #方差合成
PLOT_DIR   = os.path.join(DATA_DIR, '数据分析')                      #数据特征类绘图
RING_DIR   = os.path.join(DATA_DIR, '环形热图')                      #环形热图类绘图


# 注: PI_OOF_DIR / PI_MEM_DIR 仅在运行对应旧版区间 notebook 时才按需创建，
#     这里不再预建，避免生成空目录。
for _d in (MODEL_DIR, EXT_DIR, SHAP_DIR, PI_VQ_DIR, PLOT_DIR, RING_DIR):
    os.makedirs(_d, exist_ok=True)

def _resolve_input(name):
    """输入数据路径的查找"""
    cands = [os.path.join(DATA_DIR, name),
             os.path.join(_HERE, '..', name),
             os.path.join(_HERE, name),
             os.path.join(_HERE, '..', 'Data', name)]
    for c in cands:
        p = os.path.abspath(c)
        if os.path.exists(p):
            return p
    return os.path.abspath(os.path.join(DATA_DIR, name))


#数据文件
DATA          = _resolve_input('尼龙碳纤_力学测试数据汇总.xlsx')
EXT_DATA      = _resolve_input('8组外部验证数据集.xlsx')
SUMMARY_SHEET = '汇总-均值'

SHEETS    = {'拉伸强度': 'TS', '弹性模量': 'TM', '弯曲强度': 'FS', '弯曲模量': 'FM', '冲击强度': 'IS'}
TAG2SHEET = {v: k for k, v in SHEETS.items()}
TAG_CN    = {v: k for k, v in SHEETS.items()}
TAG_EN    = {'TS': 'Tensile strength', 'TM': 'Tensile modulus',
             'FS': 'Flexural strength', 'FM': 'Flexural modulus', 'IS': 'Impact strength'}
FEATS     = ['T_noz', 'T_bed', 'speed', 'layer', 'direction']


#结果文件路径
EXT_PRED_CMP_XLSX    = '外部验证预测结果.xlsx'
COMP_XLSX            = EXT_PRED_CMP_XLSX     
TRAIN_RESULT_XLSX    = '集成模型训练结果.xlsx'
INTERVAL_RESULT_XLSX = '区间预测评估结果_复合.xlsx'

ARTIFACT_FILE    = '集成模型存档.joblib'
ARTIFACT_PATH    = os.path.join(MODEL_DIR, ARTIFACT_FILE) 
ARTIFACT_VERSION = 3       

EXT_PRED_CMP_PATH    = os.path.join(EXT_DIR, EXT_PRED_CMP_XLSX)
TRAIN_RESULT_PATH    = os.path.join(MODEL_DIR, TRAIN_RESULT_XLSX)
INTERVAL_RESULT_PATH = os.path.join(PI_VQ_DIR, INTERVAL_RESULT_XLSX)

# 每个工况的平行试样数
N_REPL_PER_COND = 3


#原始数据表预处理和自检 拆分三个平行样本为独立同工况的三行样本
def load_long(sheet, validate=False):      
    """读取单个分表, 而不是总表 validate=True 时进行结构自检
    """
    raw = pd.read_excel(DATA, sheet_name=sheet, header=None, skiprows=2)
    if raw.shape[1] < 9:
        raise ValueError(f'[{sheet}] 列数不足 9 实际 {raw.shape[1]}），分表布局已变动')
    raw = raw.iloc[:, :10]
    raw = raw.dropna(subset=[0])
    raw.columns = ['id', 'T_noz', 'T_bed', 'speed', 'layer', 'direction',
                   's1', 's2', 's3', 'mean']
    recs = []
    for _, r in raw.iterrows():
        for s in ['s1', 's2', 's3']:  #拆分三个平行样本为独立同工况的三行样本 
            recs.append(dict(cond=int(r['id']), T_noz=r['T_noz'], T_bed=r['T_bed'],
                             speed=r['speed'], layer=r['layer'],
                             direction=r['direction'], y=r[s]))  #id就是cond就是序号
    d = pd.DataFrame(recs)
    if validate:
        _validate_long(d, sheet)
    return d

def _validate_long(d, sheet):
    """长表结构自检"""
    if d['y'].isna().any():
        bad = d.loc[d['y'].isna(), 'cond'].unique().tolist()
        raise ValueError(f'[{sheet}] 存在缺失响应值的工况: {bad[:10]}')
    if d[FEATS].isna().any().any():
        raise ValueError(f'[{sheet}] 存在缺失的工艺因子')
    
    n_per = d.groupby('cond').size()  #检查所有工况组合的样本个数是不是都是3个 如果是则nunique()为1
    if n_per.nunique() != 1:
        bad = n_per[n_per != n_per.iloc[0]]
        raise ValueError(f'[{sheet}] 各工况平行试样数不相等: {bad.to_dict()}')
    
    if int(n_per.iloc[0]) != N_REPL_PER_COND:
        raise ValueError(f'[{sheet}] 每工况平行试样数 {int(n_per.iloc[0])} '
                         f'!= 代码假定的 {N_REPL_PER_COND}')
    rng = dict(T_noz=(200, 400), T_bed=(0, 150), speed=(1, 300),
               layer=(0.01, 1.0), direction=(1, 3))
    for f, (lo, hi) in rng.items():
        v = d[f].astype(float)
        if v.min() < lo or v.max() > hi:
            raise ValueError(f'[{sheet}] 因子 {f} 取值越界: [{v.min()}, {v.max()}]')

def load_condition_means(tag):
    """原始数据表的工况均值口径数据自动计算
    返回 DataFrame: cond, y(工况均值), T_noz, T_bed, speed, layer, direction
    """
    d = load_long(TAG2SHEET[tag])
    agg = {'y': ('y', 'mean')}
    agg.update({f: (f, 'first') for f in FEATS})   #自定义聚合方式 y取均值，其他因子取第一个
    return d.groupby('cond').agg(**agg).reset_index()


#=====补丁======
# 对原始数据表的子表进行数据对比检查
def check_data_integrity(rel_tol=1e-4, abs_tol=1e-3, verbose=True):
    """比对 汇总-均值 工作表 与 试样级分表推导的工况均值。
    容差用"相对 + 绝对"双判据
    """
    sm = pd.read_excel(DATA, sheet_name=SUMMARY_SHEET)
    sm.columns = [str(c).strip() for c in sm.columns]
    id_col = sm.columns[0]
    rows = []
    for tag, cname in [('TS', '拉伸强度/MPa'), ('TM', '弹性模量/MPa'),
                       ('FS', '弯曲强度/MPa'), ('FM', '弯曲模量/MPa'),
                       ('IS', '冲击强度/kJ·m-2')]:
        if cname not in sm.columns:
            continue
        ref = pd.Series(sm[cname].astype(float).values,
                        index=sm[id_col].astype(int).values)
        der = load_condition_means(tag).set_index('cond')['y']
        for c in ref.index:
            if c not in der.index:
                continue
            diff = float(der[c] - ref[c]) 
            tol = max(abs_tol, rel_tol * abs(float(ref[c])))
            if abs(diff) > tol:
                rows.append(dict(目标=tag, 工况=int(c), 汇总值=round(float(ref[c]), 4),
                                 试样均值=round(float(der[c]), 4), 差值=round(diff, 4),
                                 相对差pct=round(100 * diff / float(ref[c]), 3)))
    df = pd.DataFrame(rows)
    if verbose:
        if df.empty:
            print(f'[数据自检] 汇总-均值 与 试样级分表一致 rel_tol={rel_tol:g}, abs_tol={abs_tol:g}')
        else:
            print(f'[数据自检] 发现 {len(df)} 处超出容差的不一致:')
            print(df.to_string(index=False))
    return df

# 外部数据表的加载和自检
def load_external():
    """
    解析 8组外部验证数据集.xlsx 的 Sheet1
    返回 dict: {tag: DataFrame(含 cond, T_noz, T_bed, speed, layer, direction, y_mean, y1,y2,y3)}
    说明：本文件按行列位置解析，任何行/列增删都会整体错位，故加结构自检。
    """
    raw = pd.read_excel(EXT_DATA, sheet_name='Sheet1', header=None)

    df_b = raw.iloc[2:10, 0:14].copy()
    df_b.columns = ['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction',
                    'fs1', 'fm1', 'fs2', 'fm2', 'fs3', 'fm3', 'y_fs', 'y_fm']
    df_t = raw.iloc[14:22, 0:14].copy()
    df_t.columns = ['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction',
                    'ts1', 'tm1', 'ts2', 'tm2', 'ts3', 'tm3', 'y_ts', 'y_tm']
    df_i = raw.iloc[29:37, 0:10].copy()
    df_i.columns = ['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction',
                    'is1', 'is2', 'is3', 'y_is']

    for name, df in [('弯曲', df_b), ('拉伸', df_t), ('冲击', df_i)]:
        _validate_external(df, name)

    df_b = df_b.reset_index(drop=True)
    df_t = df_t.reset_index(drop=True)
    df_i = df_i.reset_index(drop=True)
    df_b['cond'] = df_b['cond'].astype(int)
    df_t['cond'] = df_t['cond'].astype(int)
    df_i['cond'] = df_i['cond'].astype(int)

    ext = {
        'FS': df_b[['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction', 'y_fs', 'fs1', 'fs2', 'fs3']].rename(columns={'y_fs': 'y_mean'}),
        'FM': df_b[['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction', 'y_fm', 'fm1', 'fm2', 'fm3']].rename(columns={'y_fm': 'y_mean'}),
        'TS': df_t[['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction', 'y_ts', 'ts1', 'ts2', 'ts3']].rename(columns={'y_ts': 'y_mean'}),
        'TM': df_t[['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction', 'y_tm', 'tm1', 'tm2', 'tm3']].rename(columns={'y_tm': 'y_mean'}),
        'IS': df_i[['cond', 'T_noz', 'T_bed', 'speed', 'layer', 'direction', 'y_is', 'is1', 'is2', 'is3']].rename(columns={'y_is': 'y_mean'}),
    }

    for tag in ext:
        ext[tag] = ext[tag].copy()
        ext[tag]['y_mean'] = ext[tag]['y_mean'].astype(float)
    return ext

def _validate_external(df, name):
    """外部验证集结构自检（行/列错位时会立刻报错，而不是静默给错结果）"""
    if df.isna().any().any():
        raise ValueError(f'[外部验证·{name}] 解析出缺失值，Sheet1 的行/列布局可能已变动')
    if len(df) != 8:
        raise ValueError(f'[外部验证·{name}] 工况数 {len(df)} != 8')
    if sorted(df['cond'].astype(int).tolist()) != list(range(1, 9)):
        raise ValueError(f'[外部验证·{name}] 工况编号异常: {df["cond"].tolist()}')
    if not df['direction'].astype(float).isin([1, 2]).all():
        raise ValueError(f'[外部验证·{name}] direction 取值异常')


#工况/特征组合 分组
#这个函数用于确定GroupKold用的分组标签
def group_labels(d, mode=None):
    """返回用于 GroupKFold 的分组标签。
    """
    mode = GROUP_MODE if mode is None else mode
    if mode == 'cond':
        return np.asarray(d['cond'].values)
    if mode != 'combo':
        raise ValueError("mode 只能是 'combo' 或 'cond'")

    # 按唯一因子组合分组 combo 给相同工况组合的多样本设为同个id号 确保在GroupKFold中确保在训练集或者测试集
    key = np.round(d[FEATS].values.astype(float), 6)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return np.asarray(inv).ravel()

def describe_groups(d, mode=None):
    """分组概况"""
    mode = GROUP_MODE if mode is None else mode
    g = group_labels(d, mode)
    return dict(mode=mode, n_cond=int(d['cond'].nunique()),
                n_groups=int(len(np.unique(g))), n_specimen=int(len(d)))

def fold_duplicate_report(d, n_splits=N_OUTER, mode=None):
    """量化折间重复：每个测试折中，因子组合在训练折里原样出现的工况数"""
    X = d[FEATS].values.astype(float)
    g = group_labels(d, mode)
    y = np.zeros(len(d))
    key = [tuple(r) for r in np.round(X, 6)]
    cond = d['cond'].values
    rows = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits).split(X, y, groups=g)):
        trset = {key[i] for i in tr}
        te_cond = sorted(set(cond[te].tolist()))
        dup = sorted({int(cond[i]) for i in te if key[i] in trset})
        rows.append(dict(fold=k, n_test_cond=len(te_cond), n_dup_cond=len(dup),
                         dup_conds=dup))
    return pd.DataFrame(rows)


#Bagging和GBoost集成
class MyBagging(BaseEstimator, RegressorMixin):  #用BaseEstimator继承可以覆盖初始化的超参数
    """Bagging: 自助采样 x 决策树 -> 平均（并行降方差）"""

    def __init__(self, n_estimators=300, max_depth=None, min_samples_leaf=2,
                 max_features=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        self.trees_ = []
        for b in range(self.n_estimators):
            idx = rng.integers(0, n, n)                      # 有放回自助采样
            t = DecisionTreeRegressor(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features,
                                      random_state=int(rng.integers(1e9)))
            self.trees_.append(t.fit(X[idx], y[idx]))
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.mean([t.predict(X) for t in self.trees_], axis=0)

class MyGBoost(BaseEstimator, RegressorMixin):
    """梯度提升(平方损失): 浅树逐轮拟合残差（串行降偏差）
    只用了平方损失的一阶梯度，
    没有二阶(Newton)信息、没有显式正则项、没有列子采样。
    """

    def __init__(self, n_estimators=400, learning_rate=0.05, max_depth=3,
                 min_samples_leaf=3, subsample=0.85, random_state=0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        rng = np.random.default_rng(self.random_state
                                    )
        n = len(y)
        self.F0_ = y.mean()
        F = np.full(n, self.F0_)       # F预测值向量
        self.trees_ = []
        m = max(1, int(round(self.subsample * n)))
        for _ in range(self.n_estimators):
            r = y - F                                        # 平方损失的负梯度 = 残差
            idx = rng.choice(n, m, replace=False) if m < n else np.arange(n)
            t = DecisionTreeRegressor(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf,
                                      random_state=int(rng.integers(1e9)))
            t.fit(X[idx], r[idx])      #学习的是残差
            F += self.learning_rate * t.predict(X)           # 小步累加
            self.trees_.append(t)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.F0_ + self.learning_rate * np.sum(
            [t.predict(X) for t in self.trees_], axis=0)


#GPR构造 核距离度量/交互项
def add_dir_inter(X):
    """追加方向交互特征: (direction-1.5) * 其余4因子 -> 9列"""
    X = np.asarray(X, float)
    d = (X[:, 4:5] - 1.5)
    return np.hstack([X, d * X[:, :4]])  #按列拼接

def make_gpr(ard=True, nu=2.5, noise=1e-2, dir_inter=False):
    ndim = 9 if dir_inter else 5
    ls = [1.0] * ndim if ard else 1.0
    kern = C(1.0) * Matern(length_scale=ls, nu=nu) + WhiteKernel(noise)
    steps = [StandardScaler(),
             GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                      n_restarts_optimizer=2, random_state=0)]
    if dir_inter:
        steps.insert(0, FunctionTransformer(add_dir_inter))
    return make_pipeline(*steps)

# 基学习器
def base_learners():
    """返回 {名称: 未拟合模型}"""
    return {
        'Polynomial Regression': make_pipeline(PolynomialFeatures(2, include_bias=False),
                              StandardScaler(), Ridge(alpha=1.0)),
        'SVR': make_pipeline(StandardScaler(), SVR(C=100, gamma='scale')),
        'KNN': make_pipeline(StandardScaler(),
                             KNeighborsRegressor(5, weights='distance')),
        'GPR-ARD': make_gpr(ard=True),
        'GPR-dir': make_gpr(ard=True, dir_inter=True),
        'Bagging': MyBagging(),
        'GBoost': MyGBoost(),
        'RF': make_pipeline(StandardScaler(),
                            RandomForestRegressor(n_estimators=300, random_state=0,
                                                  n_jobs=N_JOBS_INNER)),
        'ExtraTrees': make_pipeline(StandardScaler(),
                                    ExtraTreesRegressor(n_estimators=300, random_state=0,
                                                        n_jobs=N_JOBS_INNER)),
        'GBM': make_pipeline(StandardScaler(),
                             GradientBoostingRegressor(random_state=0)),
        'AdaBoost': make_pipeline(StandardScaler(),
                                  AdaBoostRegressor(random_state=0)),
        'Lasso': make_pipeline(StandardScaler(), Lasso()),
    }


#超参网格与调优
GRIDS = {
    'Polynomial Regression': {'ridge__alpha': [0.1, 1, 10, 100]},
    'SVR': {'svr__C': [10, 100, 300],
            'svr__gamma': ['scale', 0.1, 0.3]},
            
    'KNN': {'kneighborsregressor__n_neighbors': [3, 5, 7, 9]},
    
    'GPR-ARD': {'gaussianprocessregressor__kernel': [
        C(1.0) * Matern(length_scale=[1.] * 5, nu=nu) + WhiteKernel(1e-2)
        for nu in [1.5, 2.5]]},
    'GPR-dir': {'gaussianprocessregressor__kernel': [
        C(1.0) * Matern(nu=nu) + WhiteKernel(1e-2) for nu in [1.5, 2.5]]},

    'Bagging': {'max_depth': [None, 8], 'min_samples_leaf': [1, 2],
                'max_features': [None, 3],'n_estimators':[100, 200, 300]},
    'GBoost': {'n_estimators': [200, 400],
               'learning_rate': [0.03, 0.05, 0.08],
               'max_depth': [2, 3]},
    'RF': {'randomforestregressor__n_estimators': [100, 300],
           'randomforestregressor__max_depth': [None, 8, 12],
           'randomforestregressor__min_samples_leaf': [1, 2]},
    'ExtraTrees': {'extratreesregressor__n_estimators': [100, 300],
                   'extratreesregressor__max_depth': [None, 8, 12],
                   'extratreesregressor__min_samples_leaf': [1, 2]},
    'GBM': {'gradientboostingregressor__n_estimators': [100, 200],
            'gradientboostingregressor__max_depth': [2, 3],
            'gradientboostingregressor__learning_rate': [0.05, 0.1]},
    'AdaBoost': {'adaboostregressor__n_estimators': [50, 100],
                 'adaboostregressor__learning_rate': [0.5, 1.0]},
    'Lasso': {'lasso__alpha': [0.001, 0.01, 0.1, 1.0]},
}

def grid_size(grid):
    if not grid:
        return 1
    return int(np.prod([len(v) for v in grid.values()])) #prod计算乘积  计算所有参数组合


#===补丁===
def _safe_folds(g, requested):
    """折数不能超过组数（嵌套调优时训练子集可能较小）"""
    n_g = len(np.unique(g))
    return int(max(2, min(requested, n_g)))


#在训练集调优
def tune_on(X, y, g, tag='', verbose=False):
    """在给定训练子集上网格调优全部基学习器。
    返回 (tuned, cv_scores, best_params, meta)
    """
    t0 = time.time()
    tuned, cv_scores, best_params = {}, {}, {}
    n_splits = _safe_folds(g, N_TUNE_FOLDS)
    for name, est in base_learners().items():
        gs = GridSearchCV(est, GRIDS[name], scoring='r2',
                          cv=GroupKFold(n_splits), n_jobs=N_JOBS)
        gs.fit(X, y, groups=g)

        tuned[name] = clone(gs.best_estimator_)
        cv_scores[name] = gs.best_score_
        best_params[name] = dict(gs.best_params_)

        if verbose:
            print(f'    [{tag}] {name:12s} 嵌套交叉外层训练集里CV 本折R2={gs.best_score_:6.3f} | {gs.best_params_}')
    
    meta = dict(目标=tag, 调优耗时_s=round(time.time() - t0, 1),
                学习器数=len(tuned), n_splits=n_splits)
    return tuned, cv_scores, best_params, meta

#全量数据调优
def tune_all(tag, verbose=True):
    """对单个目标用全量训练数据网格调优全部基学习器
    """
    d = load_long(TAG2SHEET[tag])
    X, y = d[FEATS].values.astype(float), d['y'].values
    g = group_labels(d)
    tuned, cv_scores, best_params = {}, {}, {}
    t0 = time.time()
    n_splits = _safe_folds(g, N_TUNE_FOLDS)
    for name, est in base_learners().items():
        gs = GridSearchCV(est, GRIDS[name], scoring='r2',
                          cv=GroupKFold(n_splits), n_jobs=N_JOBS)
        gs.fit(X, y, groups=g)
        tuned[name] = clone(gs.best_estimator_)
        cv_scores[name] = gs.best_score_
        best_params[name] = dict(gs.best_params_)
        if verbose:
            print(f'  [{tag}] {name:12s} 全量数据调优CV R2={gs.best_score_:6.3f} | {gs.best_params_}')
    elapsed = time.time() - t0
    if verbose:
        print(f'  [{tag}] 调优耗时 {elapsed:.0f}s | 折数={n_splits}')

    meta = dict(目标=tag, 调优耗时_s=round(elapsed, 1), 学习器数=len(tuned),
                cv_group=GROUP_MODE, n_splits=n_splits)
    return tuned, cv_scores, best_params, meta


# OOF非负加权融合    最终集成对象
def oof_matrix(learners, X, y, g, n_splits=N_INNER, n_jobs=None):
    """各基学习器的袋外预测矩阵 (n, L)"""

    n_splits = _safe_folds(g, n_splits)
    n_jobs = N_JOBS if n_jobs is None else int(n_jobs)

    def _predict_all(nj):
        return np.column_stack([
            cross_val_predict(clone(est), X, y, cv=GroupKFold(n_splits),
                              groups=g, n_jobs=nj)
            for est in learners.values()])
    
    #===优化===
    try:
        return _predict_all(n_jobs)
    except Exception as e:
        if n_jobs == 1:
            raise
        print(f'[错误] 并行交叉验证失败'
              f'改为串行重算')
        return _predict_all(1)

def _nnls_stable(A, b):
    """ NNLS """
    A = np.asarray(A, float)
    b = np.asarray(b, float)
    n_feat = A.shape[1]

    maxiter = max(300, 50 * n_feat)   #迭代收敛次数
    try:
        w, _ = nnls(A, b, maxiter=maxiter)
    except RuntimeError:
        print(f'\033[33m[警告] NNLS迭代不收敛 在 {maxiter} 次迭代后 位置在__nnls_stable()\033[0m')
        
        #正则化NNLS
        lam = 1e-3 * float(np.sqrt(np.mean(A ** 2)))
        A2 = np.vstack([A, lam * np.eye(n_feat)])
        b2 = np.concatenate([b, np.zeros(n_feat)])
        w, _ = nnls(A2, b2, maxiter=maxiter)
    return w

def fit_weights(P, y, cond):
    """工况均值口径下的NNLS融合权重 归一化 """
    dfP = pd.DataFrame(P).assign(cond=cond, y=y).groupby('cond').mean()
    w = _nnls_stable(dfP.drop(columns='y').values, dfP['y'].values)

    #如果所有模型权重都是0
    if w.sum() == 0:                       #退化为等权
        w = np.ones(P.shape[1]) / P.shape[1]
    else:
        w = w / w.sum()                    #归一化为占比
    return w

class WeightedEnsemble(BaseEstimator, RegressorMixin):
    """交付集成模型: 全数据拟合各成员, 按w加权"""

    def __init__(self, learners, weights):
        self.learners = learners
        self.weights = np.asarray(weights, float)

    def fit(self, X, y):
        self.fitted_ = {k: clone(m).fit(X, y) for k, m in self.learners.items()}
        self.member_names_ = list(self.learners.keys())
        return self

    def member_predictions(self, X): 
        """(n, L) 成员预测矩阵 全数据拟合"""
        X = np.asarray(X, float)
        return np.column_stack([m.predict(X) for m in self.fitted_.values()])

    def predict(self, X):
        X = np.asarray(X, float)
        return self.member_predictions(X) @ self.weights


#嵌套交叉验证 评估
def nested_oof_v2(tag, tuned=None, n_outer=N_OUTER, n_inner=N_INNER,
                  nested_tune=True, per_fold=True, verbose=True, group_mode=None):
    """
    每个外层折内部：
        1. 若 nested_tune=True 在嵌套交叉该折训练子集上重新进行网格调优
        2. 内层 GroupKFold(n_inner) 学 NNLS 权重
        3. 该折训练子集上拟合成员  预测外层测试折，得成员矩阵 Pm 与集成预测

    返回dict:
        d, oof(试样级 集成折外预测), Pm(试样级 成员折外矩阵), y, cond, group,
        names, fold_w, fold_best, fold_maxw, folds(逐折逐模型R², 工况均值口径),
        nested_tune, n_outer, tune_meta, group_mode
    """
    
    d = load_long(TAG2SHEET[tag])
    X = d[FEATS].values.astype(float)
    y = d['y'].values
    cond = d['cond'].values
    grp = group_labels(d, group_mode)
    mode_used = GROUP_MODE if group_mode is None else group_mode
    n_outer = _safe_folds(grp, n_outer)
    names = list(base_learners().keys())
    n = len(y)
    oof = np.empty(n)
    Pm = np.empty((n, len(names)))
    fold_w, fold_best, fold_maxw, folds, tune_meta = [], [], [], [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(n_outer).split(X, y, groups=grp)):
        tuned_k, cv_k = tuned, None
        if nested_tune:
            tuned_k, cv_k, _, meta_k = tune_on(X[tr], y[tr], grp[tr], tag=tag)
            tune_meta.append(meta_k)
        w = fit_weights(oof_matrix(tuned_k, X[tr], y[tr], grp[tr], n_inner),
                        y[tr], cond[tr])
        Pte = np.column_stack([clone(m).fit(X[tr], y[tr]).predict(X[te])
                               for m in tuned_k.values()])
        Pm[te] = Pte
        oof[te] = Pte @ w
        fold_w.append(w)
        if cv_k is not None:
            fold_best.append(max(cv_k, key=cv_k.get))
        else:
            fold_best.append(None)
        fold_maxw.append(names[int(np.argmax(w))])
        if per_fold:
            for j, name in enumerate(names):
                yt, pj = cond_mean_pairs(y[te], Pte[:, j], cond[te])
                folds.append(dict(fold=k, model=name, R2=r2_score(yt, pj)))
            yt, pe = cond_mean_pairs(y[te], oof[te], cond[te])
            folds.append(dict(fold=k, model='Ensemble', R2=r2_score(yt, pe)))

        if verbose:
            print(f'  [{tag}] 嵌套交叉外层折 {k + 1}/{n_outer} 完成')
    if verbose:
        print(f'  [{tag}] nested_oof_v2 总耗时 {time.time() - t0:.0f}s ')
        
    return dict(d=d, oof=oof, Pm=Pm, y=y, cond=cond, group=grp, names=names,
                fold_w=np.array(fold_w), fold_best=fold_best, fold_maxw=fold_maxw,
                folds=pd.DataFrame(folds), nested_tune=nested_tune,
                n_outer=n_outer, tune_meta=tune_meta, group_mode=mode_used)

def cond_mean_pairs(y, p, cond):
    """试样级到工况均值口径聚合 返回 (y_mean, p_mean)"""
    df = pd.DataFrame(dict(y=np.asarray(y, float), p=np.asarray(p, float),
                           cond=np.asarray(cond)))
    g = df.groupby('cond').mean()
    return g['y'].values, g['p'].values

def evaluate_oof(tag, oof, y, cond):
    """工况均值口径的评估指标"""
    yt, pt = cond_mean_pairs(y, oof, cond)
    out = dict(目标=tag,
               R2=round(float(r2_score(yt, pt)), 3),
               RMSE=round(float(mean_squared_error(yt, pt) ** 0.5), 2),
               MAPE=round(float(np.mean(np.abs((yt - pt) / yt)) * 100), 2),
               n_cond=int(len(yt)))
    return out


#区间预测里面 第二部分的实验噪声方差计算
def repeatability(tag):
    """
    返回 dict:
        sd_within     : 工况内试样间 pooled SD
        sd_run        : 同一因子组合不同"次运行"的工况均值 pooled SD
                        （由中心点重复运行估计：含批间/打印间 + 试样抽样）
        sd_pure_run   : 扣除试样抽样后的纯批间 SD
        var_sampling  : sd_within^2 / N_REPL_PER_COND
        n_rep_combos / df_run : 重复组合数与自由度
    """
    d = load_long(TAG2SHEET[tag])
    agg = {'y': ('y', 'mean'), 'n': ('y', 'size')}
    agg.update({f: (f, 'first') for f in FEATS})
    cm = d.groupby('cond').agg(**agg).reset_index()
    sdw2 = float(d.groupby('cond')['y'].var(ddof=1).mean())
    sd_within = float(np.sqrt(sdw2))

    cm['key'] = [tuple(np.round(r, 6)) for r in cm[FEATS].values]
    vc = cm['key'].value_counts()
    rep_keys = list(vc[vc > 1].index)
    ss, df = 0.0, 0
    for k in rep_keys:
        v = cm.loc[cm['key'] == k, 'y'].values.astype(float)
        ss += float(((v - v.mean()) ** 2).sum())
        df += len(v) - 1
    sd_run = float(np.sqrt(ss / df)) if df > 0 else float('nan')
    var_sampling = sdw2 / float(N_REPL_PER_COND)
    sd_pure = float(np.sqrt(max(sd_run ** 2 - var_sampling, 0.0))) if df > 0 else float('nan')
    return dict(sd_within=sd_within, sd_run=sd_run, sd_pure_run=sd_pure,
                var_sampling=var_sampling,
                sd_mean_spec=float(sd_within / np.sqrt(N_REPL_PER_COND)),
                n_rep_combos=len(rep_keys), df_run=df,
                n_cond=int(len(cm)), n_specimen=int(len(d)))


#===补丁===
#辅助函数
def noise_floor(tag, mode='run'):
    """工况均值口径的实验噪声下限 σ。
    mode='run' —— 用中心点重复运行估计的批间标准差。
    """
    r = repeatability(tag)
    return r['sd_run'] if mode == 'run' else r['sd_mean_spec']

def noise_scale(tag, mode='run'):
    """方差合成用的实验噪声标准差
    """
    r = repeatability(tag)
    order = (('sd_run', 'sd_pure_run', 'sd_mean_spec') if mode == 'run'
             else ('sd_mean_spec', 'sd_pure_run', 'sd_run'))
    src, val = 'none', 0.0
    for k in order:
        v = float(r.get(k, np.nan))
        if np.isfinite(v) and v > 0:
            src, val = k, v
            break
    return dict(sigma=val, source=src,
                sd_within=r['sd_within'], sd_run=r['sd_run'],
                sd_pure_run=r['sd_pure_run'], sd_mean_spec=r['sd_mean_spec'],
                var_sampling=r['var_sampling'],
                n_rep_combos=r['n_rep_combos'], df_run=r['df_run'])


#最终全量数据构建集成模型与外部验证
def build_final_ensemble(tag, tuned_learners, cv_scores=None, verbose=True):
    """用全部训练数据构建最终部署集成：
       1. 各基学习器折外预测  工况均值口径 NNLS 权重
       2. 全数据重拟合   WeightedEnsemble
    """
    d = load_long(TAG2SHEET[tag])
    X, y = d[FEATS].values.astype(float), d['y'].values
    g = group_labels(d)
    P = oof_matrix(tuned_learners, X, y, g, N_INNER)
    w = fit_weights(P, y, d['cond'].values)
    best_single_name = max(cv_scores, key=cv_scores.get) if cv_scores else None
    maxw_single_name = list(tuned_learners.keys())[int(np.argmax(w))]

    if verbose:
        print(f"\n[{tag}] 基学习器权重分布:")
        for name, wi in zip(tuned_learners.keys(), w):
            if wi > 0.001:
                print(f"  {name:12s}: {wi:.4f}")
        print(f"  非零权重数: {(w > 0.001).sum()} / {len(w)}")
        if best_single_name:
            print(f"  指标最优单一模型: {best_single_name} (CV R2={cv_scores[best_single_name]:.3f})")
        print(f"  权重最大单一模型: {maxw_single_name} (权重={w.max():.4f})")
    
    ens = WeightedEnsemble(tuned_learners, w)
    ens.fit(X, y)
    return ens, w, best_single_name, maxw_single_name

def best_and_maxw_single(tag, cv_scores, weights, names=None):
    """由超参数调优CV得分与融合权重推出  CV最优单一模型 / 权重最大单一模型"""
    if names is None:
        names = list(base_learners().keys())
    if len(names) != len(np.asarray(weights).ravel()):
        raise ValueError(f'names 长度 {len(names)} 与权重长度 {len(weights)} 不一致')
    best = max(cv_scores, key=cv_scores.get) if cv_scores else None
    maxw = names[int(np.argmax(weights))]
    return best, maxw


#===补丁===
#辅助函数 显著性工具
def _metrics(y_t, y_p):
    y_t = np.asarray(y_t, float)
    y_p = np.asarray(y_p, float)
    nz = y_t != 0              
    mape = float(np.mean(np.abs((y_t[nz] - y_p[nz]) / y_t[nz])) * 100) if nz.any() else np.nan
    return dict(R2=float(r2_score(y_t, y_p)),
                RMSE=float(mean_squared_error(y_t, y_p) ** 0.5),
                MAPE=mape)

def paired_sign_test(y_true, p_a, p_b):
    """精确配对符号检验
    检验 H0: A 与 B 的绝对误差同分布；单侧备择 H1: A 的绝对误差更小。
    返回 dict(n_pairs, wins_a, p_one_sided, median_absdiff)
    """
    ea = np.abs(np.asarray(y_true, float) - np.asarray(p_a, float))
    eb = np.abs(np.asarray(y_true, float) - np.asarray(p_b, float))
    d = ea - eb
    nz = d[d != 0]
    n = int(nz.size)
    wins = int((nz < 0).sum())            # A 更优的次数
    if n == 0:
        p = np.nan
    else:
        p = float(stats.binomtest(wins, n, 0.5, alternative='greater').pvalue)
    return dict(n_pairs=n, wins_a=wins, p_one_sided=p,
                median_absdiff=float(np.median(d)) if len(d) else np.nan)

def predict_external_with_comparison(tag, ensemble, tuned_learners,
                                     best_single_name, maxw_single_name,
                                     ext_df, verbose=True):
    """预测外部验证集并做三模型对比（含精确配对符号检验）
    返回 (res_df, summary_row, comparison)
    """
    X_ext = ext_df[FEATS].values.astype(float)
    y_true = np.asarray(ext_df['y_mean'].values, float)

    y_pred_ens = np.asarray(ensemble.predict(X_ext), float)

    d = load_long(TAG2SHEET[tag])
    Xtr, ytr = d[FEATS].values.astype(float), d['y'].values
    y_pred_best = np.asarray(clone(tuned_learners[best_single_name]).fit(Xtr, ytr)
                             .predict(X_ext), float) if best_single_name else None
    y_pred_maxw = np.asarray(clone(tuned_learners[maxw_single_name]).fit(Xtr, ytr)
                             .predict(X_ext), float) if maxw_single_name else None

    res = {'cond': ext_df['cond'].values, 'y_true': y_true, 'y_ensemble': y_pred_ens}
    if best_single_name:
        res[f'y_best_single_{best_single_name}'] = y_pred_best
    if maxw_single_name:
        res[f'y_maxweight_single_{maxw_single_name}'] = y_pred_maxw
    res = pd.DataFrame(res)

    comparison = {'Ensemble': _metrics(y_true, y_pred_ens)}
    if y_pred_best is not None:
        comparison['BestSingle'] = _metrics(y_true, y_pred_best)
        comparison['Ens_vs_Best_signtest'] = paired_sign_test(y_true, y_pred_ens, y_pred_best)
    if y_pred_maxw is not None:
        comparison['MaxWeightSingle'] = _metrics(y_true, y_pred_maxw)
        comparison['Ens_vs_MaxW_signtest'] = paired_sign_test(y_true, y_pred_ens, y_pred_maxw)

    if verbose:
        print(f"\n{'=' * 72}")
        print(f"外部验证结果 [{tag}] {TAG_CN[tag]}  (n={len(y_true)} 工况)")
        print(f"{'=' * 72}")
        print(res.to_string(index=False))
        print("\n--- 性能对比 ---")

        for k in ['Ensemble', 'BestSingle', 'MaxWeightSingle']:
            if k not in comparison:
                continue
            m = comparison[k]
            print(f"  {k:<18} R2={m['R2']:>7.3f} "
                  f"RMSE={m['RMSE']:>9.2f} MAPE={m['MAPE']:>7.2f}%")
        for key, lab in [('Ens_vs_Best_signtest', '集成 vs 单一(CV最优)'),
                         ('Ens_vs_MaxW_signtest', '集成 vs 单一(权重最大)')]:
            if key in comparison:
                s = comparison[key]
                print(f"  [{lab}] 集成更优 {s['wins_a']}/{s['n_pairs']} 个工况 | "
                      f"精确符号检验单侧 p={s['p_one_sided']:.3f}")
        print('  注意: n=8 时 R² 的抽样不确定性很大，结论需结合外部数据解读。')

    summary_row = dict(
        tag=tag, n_ext=len(y_true),
        Ensemble_R2=round(comparison['Ensemble']['R2'], 3),
        Ensemble_RMSE=round(comparison['Ensemble']['RMSE'], 2),
        Ensemble_MAPE=round(comparison['Ensemble']['MAPE'], 2),
        BestSingle=best_single_name, MaxWeightSingle=maxw_single_name,
    )
    for k in ('BestSingle', 'MaxWeightSingle'):
        if k in comparison:
            summary_row[f'{k}_R2'] = round(comparison[k]['R2'], 3)
            summary_row[f'{k}_RMSE'] = round(comparison[k]['RMSE'], 2)
            summary_row[f'{k}_MAPE'] = round(comparison[k]['MAPE'], 2)
    if 'Ens_vs_Best_signtest' in comparison:
        summary_row['Ens_vs_Best_p'] = round(float(comparison['Ens_vs_Best_signtest']['p_one_sided']), 4)
        summary_row['Ens_vs_Best_wins'] = int(comparison['Ens_vs_Best_signtest']['wins_a'])
    if 'Ens_vs_MaxW_signtest' in comparison:
        summary_row['Ens_vs_MaxW_p'] = round(float(comparison['Ens_vs_MaxW_signtest']['p_one_sided']), 4)
        summary_row['Ens_vs_MaxW_wins'] = int(comparison['Ens_vs_MaxW_signtest']['wins_a'])
    return res, summary_row, comparison

#辅助函数 超参数组合表 / 训练参数表的构建
def _fmt_val(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return str(list(v))
    if v is None:
        return 'None'
    return str(v)

def format_params(params):
    if not params:
        return '(默认参数)'
    return '; '.join(f'{k}={_fmt_val(v)}' for k, v in params.items())

def build_hyperparam_tables(tuned, cv_scores, best_params, meta):
    names = list(tuned.keys())
    wide = pd.DataFrame({
        '学习器': names,
        '最优参数组合': [format_params(best_params[n]) for n in names],
        'CV_R2': [round(float(cv_scores[n]), 4) for n in names],
        '调优耗时_s': meta.get('调优耗时_s', np.nan),
    })
    rows = []
    for n in names:
        for k, v in best_params[n].items():
            rows.append(dict(目标=meta.get('目标', ''),
                             学习器=n, 参数名=k, 参数值=_fmt_val(v),
                             CV_R2=round(float(cv_scores[n]), 4)))
    return wide, pd.DataFrame(rows)

#辅助函数  模型存档
ARTIFACT_SEARCH_PATHS = [
    ARTIFACT_PATH,                                 
    os.path.join(DATA_DIR, ARTIFACT_FILE),       
    os.path.join(_HERE, ARTIFACT_FILE),            
    os.path.join(_HERE, '..', 'Data', ARTIFACT_FILE),
]

def save_artifacts(ART, path=ARTIFACT_PATH):
    """保存模型存档"""
    import joblib
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ART = dict(ART)
    ART['_artifact_version'] = ARTIFACT_VERSION
    ART['_group_mode'] = GROUP_MODE
    ART['_created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(path, 'wb') as f:
        joblib.dump(ART, f, compress=3)
    print(f'模型存档已保存: {path} (version={ARTIFACT_VERSION}, group_mode={GROUP_MODE})')

def load_artifacts(path=ARTIFACT_PATH):
    import joblib
    with open(path, 'rb') as f:
        return joblib.load(f)

def _find_artifact(path):
    if path and os.path.exists(path):
        return path
    for p in ARTIFACT_SEARCH_PATHS:
        if os.path.exists(p):
            return p
    return None


#加载模型存档
def load_or_build_models(path=ARTIFACT_PATH, verbose=True):
    """加载训练好保存的模型存档"""
    found = _find_artifact(path)
    if found:
        try:
            ART = load_artifacts(found)
            ver = ART.get('_artifact_version', 1)
            gm = ART.get('_group_mode', 'cond')
            if ver != ARTIFACT_VERSION:
                print(f'[警告] 存档版本 {ver} != 当前 {ARTIFACT_VERSION}，忽略该存档并重建')
            elif gm != GROUP_MODE:
                print(f'[警告] 存档分组口径 {gm} != 当前 {GROUP_MODE}，忽略该存档并重建。')
            else:
                if verbose:
                    print(f'[加载] 已从存档载入集成模型: {found} (version={ver})')
                return ART
        except Exception as e:
            print(f'[警告] 存档读取失败({e!r})')
    else:
        if verbose:
            print(f'[提示] 未找到模型存档')


    if verbose:
        print('[重建] 重训全部目标')
    TUNED, CV_SCORES, BEST_PARAMS, TUNE_META = {}, {}, {}, {}
    for tag in SHEETS.values():
        TUNED[tag], CV_SCORES[tag], BEST_PARAMS[tag], TUNE_META[tag] = \
            tune_all(tag, verbose=verbose)
    ENSEMBLES, W_raw = {}, {}
    for tag in SHEETS.values():
        d = load_long(TAG2SHEET[tag])
        X, y = d[FEATS].values.astype(float), d['y'].values
        g = group_labels(d)
        w = fit_weights(oof_matrix(TUNED[tag], X, y, g, N_INNER), y, d['cond'].values)
        ENSEMBLES[tag] = WeightedEnsemble(TUNED[tag], w).fit(X, y)
        W_raw[tag] = w

    WEIGHTS_TABLE = pd.DataFrame(
        {tag: np.round(w / w.sum(), 3) for tag, w in W_raw.items()},
        index=list(base_learners().keys()))
    WEIGHTS_TABLE.index.name = '学习器'

    return dict(ENSEMBLES=ENSEMBLES, TUNED=TUNED, W_raw=W_raw,
                WEIGHTS_TABLE=WEIGHTS_TABLE, CV_SCORES=CV_SCORES,
                BEST_PARAMS=BEST_PARAMS, TUNE_META=TUNE_META,
                _artifact_version=ARTIFACT_VERSION, _group_mode=GROUP_MODE)


#区间预测与不确定性量化
def conformal_quantile(scores, alpha):
    """有限样本校正的共形分位数：第 ⌈(n+1)(1-α)⌉ 个次序统计量。
    """
    s = np.sort(np.asarray(scores, float).ravel())
    n = s.size
    if n == 0:
        raise ValueError('校准分数为空')
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float('inf')
    return float(s[k - 1])

def variance_quadrature_scale(u, sigma_noise):
    """方差合成: σ_total(x) = sqrt(σ_model(x)² + σ_noise²)。
    两类不确定性来源 模型认知 / 实验固有噪声
    """
    u = np.asarray(u, float)
    return np.sqrt(u ** 2 + float(sigma_noise) ** 2)

def calibrate_quadrature_conformal(r_cal, u_cal, sigma_noise, alpha):
    """分数 s_i = |r_i| / sqrt(u_i² + σ_noise²)，q̂ 取 ⌈(n+1)(1-α)⌉ 次序统计量。
    返回 dict(qhat, sigma_noise, alpha, n_cal)
    """
    r = np.asarray(r_cal, float).ravel()
    u = np.asarray(u_cal, float).ravel()
    if r.size != u.size:
        raise ValueError('r_cal 与 u_cal 长度不一致')
    scale = variance_quadrature_scale(u, sigma_noise)
    q = conformal_quantile(np.abs(r) / scale, alpha)
    return dict(qhat=q, sigma_noise=float(sigma_noise), alpha=alpha, n_cal=int(r.size))

def quadrature_conformal_interval(mu, u, cal):
    """方差合成共形区间 mu ± q̂ · sqrt(u² + σ_noise²)"""
    mu = np.asarray(mu, float)
    scale = variance_quadrature_scale(u, cal['sigma_noise'])
    half = cal['qhat'] * scale
    return mu - half, mu + half, mu


#===补丁===
def variance_decomposition(r, u, sigma_noise):
    """方差分解自洽性诊断：Var(r) ≈ κ·E[u²] + σ_noise²  →  κ = (Var(r) − σ_noise²)/E[u²]

    κ ≈ 1 → 分歧与模型误差方差一致，方差合成自洽
    κ > 1 → 分歧低估模型误差方差（区间可能偏窄，PICP 略低于名义）
    κ < 1 → 分歧高估（区间偏保守）
    """
    r = np.asarray(r, float).ravel()
    u = np.asarray(u, float).ravel()
    var_r = float(np.var(r, ddof=1)) if r.size > 1 else float('nan')
    mean_u2 = float(np.mean(u ** 2)) if u.size else float('nan')
    var_n = float(sigma_noise) ** 2
    kappa = (var_r - var_n) / mean_u2 if mean_u2 > 0 else float('nan')
    return dict(var_r=var_r, mean_u2=mean_u2, var_noise=var_n, kappa=kappa,
                noise_share=(var_n / var_r) if var_r > 0 else float('nan'))

def residual_conformal_interval(mu, r_cal, alpha):
    """残差共形（常数宽度，仅用残差、不用分歧）"""
    q = conformal_quantile(np.abs(r_cal), alpha)
    mu = np.asarray(mu, float)
    return mu - q, mu + q, mu

def member_quantile_interval(Pm, alpha):
    """成员分位数区间
    """
    P = np.asarray(Pm, float)
    L = P.shape[1]
    k_hi = int(np.ceil((L + 1) * (1.0 - alpha / 2.0)))
    k_lo = int(np.ceil((L + 1) * (alpha / 2.0)))
    S = np.sort(P, axis=1)
    hi = S[:, min(k_hi, L) - 1]
    lo = S[:, max(k_lo, 1) - 1]
    return lo, hi

def cross_conformal_coverage_quadrature(r, u, sigma_noise, alpha, groups=None, n_splits=5):
    """方差合成版的交叉共形覆盖率
    返回 dict(PICP, PICP_folds, MPIW, qhats, n_cal, n_splits)
    """
    r = np.asarray(r, float).ravel()
    u = np.asarray(u, float).ravel()
    if r.size != u.size:
        raise ValueError('r 与 u 长度不一致')
    n = r.size
    n_splits = int(max(2, min(n_splits, n)))
    if groups is None:
        folds = list(KFold(n_splits, shuffle=True, random_state=0).split(r))
    else:
        folds = list(GroupKFold(n_splits).split(r, r, groups=np.asarray(groups)))
    picps, mpiws, qhats = [], [], []
    for tr, te in folds:
        q = conformal_quantile(
            np.abs(r[tr]) / variance_quadrature_scale(u[tr], sigma_noise), alpha)
        qhats.append(q)
        if not np.isfinite(q):
            picps.append(0.0)
            mpiws.append(np.inf)
            continue
        half = q * variance_quadrature_scale(u[te], sigma_noise)
        picps.append(float(np.mean(np.abs(r[te]) <= half)))
        mpiws.append(float(np.mean(2.0 * half)))
    return dict(PICP=float(np.mean(picps)), PICP_folds=[float(x) for x in picps],
                MPIW=float(np.mean(mpiws)), qhats=qhats, n_cal=int(n),
                n_splits=n_splits)

def cross_conformal_coverage_constant(r, alpha, groups=None, n_splits=5):
    """常数宽度（仅残差）共形的交叉覆盖率
    """
    r = np.asarray(r, float).ravel()
    n = r.size
    n_splits = int(max(2, min(n_splits, n)))
    if groups is None:
        folds = list(KFold(n_splits, shuffle=True, random_state=0).split(r))
    else:
        folds = list(GroupKFold(n_splits).split(r, r, groups=np.asarray(groups)))
    picps, mpivs, qhats = [], [], []
    for tr, te in folds:
        q = conformal_quantile(np.abs(r[tr]), alpha)
        qhats.append(q)
        if not np.isfinite(q):
            picps.append(0.0)
            mpivs.append(np.inf)
            continue
        picps.append(float(np.mean(np.abs(r[te]) <= q)))
        mpivs.append(float(2.0 * q))
    return dict(PICP=float(np.mean(picps)), PICP_folds=[float(x) for x in picps],
                MPIW=float(np.mean(mpivs)), qhats=qhats, n_cal=int(n),
                n_splits=n_splits)


# 评价指标 
def picp(y_true, lo, hi):
    """PICP: 真值落在区间内的比例（越接近名义水平 1-α 越好）"""
    y_true = np.asarray(y_true, float)
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def mpiw(lo, hi):
    """MPIW: 平均区间宽度（原始量纲）"""
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))

def pinaw(lo, hi, y_true):
    """PINAW: 归一化平均宽度 = MPIW / (max(y)-min(y))"""
    R = float(np.ptp(np.asarray(y_true, float)))
    if R == 0:
        return np.nan
    return mpiw(lo, hi) / R


#===补丁===
def interval_metrics(y_true, lo, hi, groups=None):
    """一次算齐 PICP / MPIW / PINAW。
    groups 给定且 n 足够时，额外给出"逐折 PICP"的均值与标准差。"""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    y_true = np.asarray(y_true, float)
    out = dict(PICP=picp(y_true, lo, hi), MPIW=mpiw(lo, hi),
               PINAW=pinaw(lo, hi, y_true), n=int(len(y_true)))
    if groups is not None:
        gs = np.asarray(groups)
        vals = [picp(y_true[gs == k], lo[gs == k], hi[gs == k]) for k in np.unique(gs)]
        if len(vals) > 1:
            out['PICP_foldmean'] = float(np.mean(vals))
            out['PICP_foldstd'] = float(np.std(vals, ddof=1))
    return out


# 区间预测对象
class IntervalEnsemble(BaseEstimator, RegressorMixin):
    """在已拟合的 WeightedEnsemble 之上提供共形区间预测
    """

    def __init__(self, ensemble, X_tr, y_tr, cond_tr):
        self.ensemble = ensemble
        self.X_tr = np.asarray(X_tr, float)
        self.y_tr = np.asarray(y_tr, float)
        self.cond_tr = np.asarray(cond_tr)
        self.qhat_ = None

    def predict(self, X):
        return self.ensemble.predict(X)

    def calibrate_conformal(self, oof_pred, y_true, alpha=ALPHA):
        """常数宽度共形：q̂ = |r| 的 ⌈(n+1)(1-α)⌉ 次序统计量（有限样本校正）。"""
        s = np.abs(np.asarray(y_true, float) - np.asarray(oof_pred, float))
        self.qhat_ = conformal_quantile(s, alpha)
        self.conformal_alpha_ = alpha
        return self.qhat_

    def calibrate_quadrature(self, oof_pred, y_true, u_cal, sigma_noise, alpha=ALPHA):
        """方差合成共形：σ_total=sqrt(σ_model²+σ_noise²)，单一标定常数 q̂。

        σ_noise 必须由**重复实验外部给定**（noise_scale(tag)），不能从本批残差回估。
        """
        r = np.asarray(y_true, float) - np.asarray(oof_pred, float)
        self.cal_quad_ = calibrate_quadrature_conformal(r, u_cal, sigma_noise, alpha)
        self.conformal_alpha_ = alpha
        return self.cal_quad_

    def conformal_interval(self, X):
        """常数宽度区间（需先 calibrate_conformal）"""
        if self.qhat_ is None:
            raise ValueError('请先调用 calibrate_conformal(oof_pred, y_true, alpha)')
        if not np.isfinite(self.qhat_):
            raise ValueError('校准集过小，有限样本共形区间为无穷宽（q̂=+inf）')
        mu = self.predict(X)
        return mu - self.qhat_, mu + self.qhat_, mu

    def quadrature_interval(self, X):
        """方差合成区间（需先 calibrate_quadrature）"""
        if not hasattr(self, 'cal_quad_'):
            raise ValueError('请先调用 calibrate_quadrature(oof_pred, y_true, u_cal, sigma_noise)')
        mu = self.predict(X)
        u = np.std(self.ensemble.member_predictions(X), axis=1)
        return quadrature_conformal_interval(mu, u, self.cal_quad_)


def lack_of_fit_test(tag):
    """经典 DoE 失拟检验"""
    cm = load_condition_means(tag)
    X = cm[FEATS].values.astype(float)
    y = cm['y'].values.astype(float)
    cm = cm.assign(key=[tuple(np.round(r, 6)) for r in X])
    vc = cm['key'].value_counts()
    rep_keys = list(vc[vc > 1].index)
    if not rep_keys:
        return dict(目标=tag, 失拟F=np.nan, p值=np.nan, 说明='无重复运行，无法做失拟检验')
    ss_pe, df_pe = 0.0, 0
    for k in rep_keys:
        v = cm.loc[cm['key'] == k, 'y'].values.astype(float)
        ss_pe += float(((v - v.mean()) ** 2).sum())
        df_pe += len(v) - 1
    A = _quad_design(X)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_e = float((resid ** 2).sum())
    df_e = len(y) - A.shape[1]
    ss_lof = ss_e - ss_pe
    df_lof = df_e - df_pe
    if df_lof <= 0 or ss_pe <= 0:
        return dict(目标=tag, 失拟F=np.nan, p值=np.nan, 说明='自由度或纯误差不足')
    F = (ss_lof / df_lof) / (ss_pe / df_pe)
    p = float(1.0 - stats.f.cdf(F, df_lof, df_pe))
    return dict(目标=tag, 全二次项数=int(A.shape[1]), SS_E=round(ss_e, 4),
                df_E=df_e, SS_PE=round(ss_pe, 4), df_PE=df_pe,
                SS_LOF=round(ss_lof, 4), df_LOF=df_lof,
                失拟F=round(float(F), 3), p值=round(p, 4),
                结论='二次模型不足(存在更高阶/交互效应)' if p < 0.05 else '二次模型充分')

def _quad_design(X):
    """全二次设计矩阵（含截距、线性、平方、交互）"""
    X = np.asarray(X, float)
    n, p = X.shape
    cols = [np.ones(n)]
    for j in range(p):
        cols.append(X[:, j])
    for j in range(p):
        for k in range(j, p):
            cols.append(X[:, j] * X[:, k])
    return np.column_stack(cols)

def _model_display_name(ens, model, best_name, maxw_name, top_k=4):
    """把 Ensemble/BestSingle/MaxWeightSingle 泛称翻译成具体模型名
      BestSingle      -> BEST_SINGLE[tag] CV 最优的基学习器
      MaxWeightSingle -> MAXW_SINGLE[tag] NNLS 权重最大的基学习器
      Ensemble        -> 'NNLS集成(' + 权重最大的 top_k 个成员及权重 + ')'，权重 0 的成员不列
    """
    if model == 'BestSingle':
        return str(best_name) if best_name else '—'
    if model == 'MaxWeightSingle':
        return str(maxw_name) if maxw_name else '—'
    if model != 'Ensemble':
        return str(model)

    names = list(getattr(ens, 'member_names_', None) or [])
    w = np.asarray(getattr(ens, 'weights', []), float).ravel()
    if not names or w.size != len(names):
        return 'NNLS集成'
    nz = [(n, wi) for n, wi in zip(names, w) if wi > 1e-6]
    if not nz:
        return 'NNLS集成'
    nz.sort(key=lambda t: -t[1])
    parts = [f'{n} {wi:.3f}' for n, wi in nz[:max(1, int(top_k))]]
    if len(nz) > len(parts):
        parts.append(f'等{len(nz)}项')
    return 'NNLS集成(' + ', '.join(parts) + ')'


# 点预测指标长表
def point_indicator_table(ART, ext=None, targets=None):
    ENS = ART['ENSEMBLES']
    NEST = ART.get('NESTED') or {}
    best_map = ART.get('BEST_SINGLE') or {}
    maxw_map = ART.get('MAXW_SINGLE') or {}
    if ext is None:
        ext = load_external()
    tags = list(targets) if targets is not None else list(SHEETS.values())

    rows = []
    for tag in tags:
        cm = load_condition_means(tag)
        Xc, yc = cm[FEATS].values.astype(float), cm['y'].values.astype(float)
        Xe = ext[tag][FEATS].values.astype(float)
        ye = ext[tag]['y_mean'].values.astype(float)
        Pmc = ENS[tag].member_predictions(Xc)     
        Pme = ENS[tag].member_predictions(Xe)

        nd = NEST.get(tag)
        if nd is not None:
            names = list(nd['names'])
            d = nd['d']
            cond = np.asarray(nd['cond'])
            Xs, ys = d[FEATS].values.astype(float), d['y'].values.astype(float)
            oof = {'Ensemble': np.asarray(nd['oof'], float),
                   'BestSingle': np.full(len(ys), np.nan),
                   'MaxWeightSingle': np.full(len(ys), np.nan)}
    
            for k, (_tr, te) in enumerate(
                    GroupKFold(int(nd['n_outer'])).split(Xs, ys, groups=nd['group'])):
                b, m = nd['fold_best'][k], nd['fold_maxw'][k]
                if b in names:
                    oof['BestSingle'][te] = nd['Pm'][te, names.index(b)]
                if m in names:
                    oof['MaxWeightSingle'][te] = nd['Pm'][te, names.index(m)]
        else:
            names, cond, ys, oof = list(base_learners().keys()), None, None, None

        def _col(P, name, _names=names):
            return P[:, _names.index(name)] if (name is not None and name in _names) else None

        pred = {
            'Ensemble': (ENS[tag].predict(Xc), ENS[tag].predict(Xe),
                         oof['Ensemble'] if oof else None),
            'BestSingle': (_col(Pmc, best_map.get(tag)), _col(Pme, best_map.get(tag)),
                           oof['BestSingle'] if oof else None),
            'MaxWeightSingle': (_col(Pmc, maxw_map.get(tag)), _col(Pme, maxw_map.get(tag)),
                                oof['MaxWeightSingle'] if oof else None),
        }

        for model, (p_in, p_ex, p_oof) in pred.items():
            _name = _model_display_name(ENS.get(tag), model, best_map.get(tag), maxw_map.get(tag))
            if p_in is not None:
                rows.append(dict(tag=tag, 数据集='原始_训练集', model=model, 模型名=_name,
                                 **_metrics(yc, p_in)))
            if p_oof is not None and np.isfinite(p_oof).all():
                y_cm, p_cm = cond_mean_pairs(ys, p_oof, cond)
                rows.append(dict(tag=tag, 数据集='原始_测试集', model=model, 模型名=_name,
                                 **_metrics(y_cm, p_cm)))
            if p_ex is not None:
                rows.append(dict(tag=tag, 数据集='外部_测试集', model=model, 模型名=_name,
                                 **_metrics(ye, p_ex)))

    return pd.DataFrame(rows, columns=['tag', '数据集', 'model', '模型名', 'R2', 'RMSE', 'MAPE'])
