
import time
import psutil
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef, precision_recall_fscore_support,
    average_precision_score  # 新增：导入AUPR计算函数
)
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import torch
import torch.nn as nn  # 确保导入nn，用于Focal Loss定义
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
# 导入双分支模型（修改：适配新模型类名）
from model import NcrnaDrugDiseaseDualBranchModel as NcrnaDrugDiseaseModel, parameters_set
# 导入负样本生成函数
from Utils.negative_sample_generate import neg_data_generate
import os
# 启用CUDA同步执行，准确定位断言错误
os.environ["CUDA_LAUNCH_BLOCKING"] = "3"
os.environ["TORCH_USE_CUDA_DSA"] = "3"  # 启用设备端断言调试

sorted_pred_dict = {}
EXPORT_OOF_PREDICTIONS = os.environ.get("SPHLCMGA_EXPORT_OOF", "0") == "1"
RESULT_DIR_OVERRIDE = os.environ.get("SPHLCMGA_RESULT_DIR")
BEST_MODEL_DIR_OVERRIDE = os.environ.get("SPHLCMGA_BEST_MODEL_DIR")


# 设置中文字体（避免绘图中文乱码）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 英文无乱码，如需中文：['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------------- 新增：多分类Focal Loss（解决类别不平衡） --------------------------
class FocalLoss(nn.Module):
    """
    多分类Focal Loss实现
    参数：
    - alpha: 类别权重（与CrossEntropyLoss的weight参数一致，形状[num_classes]）
    - gamma: 聚焦参数（默认2，越大对易分类样本权重衰减越明显）
    - reduction: 损失聚合方式（默认mean，返回均值；sum返回总和；none返回逐样本损失）
    """

    def __init__(self, alpha=None, gamma=0.3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重（可传入之前计算的类别不平衡权重）
        self.gamma = gamma  # 聚焦参数，核心是降低易分类样本的权重
        self.reduction = reduction  # 损失聚合方式
        self.eps = 1e-8  # 防止数值下溢（log(0)问题）

    def forward(self, logits, labels):
        """
        前向传播：
        - logits: 模型输出的原始预测值（未经过softmax，形状[B, num_classes]）
        - labels: 真实标签（形状[B]，整数类型）
        """
        # 1. 计算softmax概率，获取每个样本对应真实类别的概率p_t
        log_softmax = F.log_softmax(logits, dim=1)  # 数值稳定的log(softmax)
        softmax_probs = torch.exp(log_softmax)  # 转换为softmax概率

        # 提取每个样本对应真实标签的概率（p_t）和log概率（log_p_t）
        p_t = softmax_probs.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
        log_p_t = log_softmax.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)

        # 2. 应用Focal Loss调制因子：(1 - p_t)^gamma
        modulating_factor = (1.0 - p_t) ** self.gamma

        # 3. 计算带权重的Focal Loss（如果有alpha类别权重）
        if self.alpha is not None:
            # 提取每个样本对应真实标签的alpha权重
            alpha_t = self.alpha.gather(dim=0, index=labels)
            # 组合alpha和调制因子，计算最终损失
            focal_loss = -alpha_t * modulating_factor * log_p_t
        else:
            # 无类别权重的基础Focal Loss
            focal_loss = -modulating_factor * log_p_t

        # 4. 损失聚合（与CrossEntropyLoss保持一致的返回格式）
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:  # none
            return focal_loss


# -------------------------- 修改：关系超图构建函数（保持不变，已兼容模型） --------------------------
def build_relational_hypergraph(train_data, ncrna_num, drug_num, disease_num, device):
    """
    构建关系超图（修改后）：
    - 耐药超边（标签1）：权重-1；敏感超边（标签2）：权重+1
    - 输出合并后的hyperedge_index和对应的sign_weights，符合SignedHypergraphConv输入要求
    参数：
    - train_data: 训练数据 [N, 4]，列：ncRNA_id, drug_id, disease_id, label
    - ncrna_num/drug_num/disease_num: 各模态实体数量（用于ID偏移）
    - device: 张量设备（cuda/cpu）
    返回：
    - hyperedge_index: 合并后的超边索引 [2, E*3]（E为总超边数）
    - sign_weights: 超边符号权重 [E,]（耐药=-1，敏感=+1）
    """
    # 1. 筛选关系样本
    resistant_data = train_data[train_data[:, 3] == 1]  # 耐药性（标签1，权重-1）
    sensitive_data = train_data[train_data[:, 3] == 2]  # 敏感性（标签2，权重+1）

    # 2. 构建单类超边索引和权重（内部辅助函数）
    def build_single_hyperedge(data, ncrna_num, drug_num, sign):
        if len(data) == 0:
            return np.array([[], []]), np.array([])
        # ID偏移：drug_id = drug_id + ncrna_num; disease_id = disease_id + ncrna_num + drug_num
        ncrna_ids = data[:, 0].astype(int)
        drug_ids = data[:, 1].astype(int) + ncrna_num
        disease_ids = data[:, 2].astype(int) + ncrna_num + drug_num

        # 构建超边：每个三元组对应一个超边，包含3个节点
        hyperedge_nodes = np.vstack([ncrna_ids, drug_ids, disease_ids]).T  # [E, 3]
        hyperedge_index = []
        for e_idx, nodes in enumerate(hyperedge_nodes):
            for node in nodes:
                hyperedge_index.append([node, e_idx])  # [节点ID, 超边ID]
        hyperedge_index = np.array(hyperedge_index).T  # [2, E*3]
        # 构建该类超边的符号权重
        sign_weights = np.ones(len(hyperedge_nodes)) * sign  # [E,]
        return hyperedge_index, sign_weights

    # 3. 生成两类关系的超边和权重
    res_edge, res_sign = build_single_hyperedge(resistant_data, ncrna_num, drug_num, sign=-1)
    sen_edge, sen_sign = build_single_hyperedge(sensitive_data, ncrna_num, drug_num, sign=+1)

    # 4. 合并两类超边（处理空数据情况）
    if res_edge.size == 0 and sen_edge.size == 0:
        hyperedge_index = torch.tensor([[], []], dtype=torch.long).to(device)
        sign_weights = torch.tensor([], dtype=torch.float).to(device)
    elif res_edge.size == 0:
        # 仅敏感超边
        hyperedge_index = torch.tensor(sen_edge, dtype=torch.long).to(device)
        sign_weights = torch.tensor(sen_sign, dtype=torch.float).to(device)
    elif sen_edge.size == 0:
        # 仅耐药超边
        hyperedge_index = torch.tensor(res_edge, dtype=torch.long).to(device)
        sign_weights = torch.tensor(res_sign, dtype=torch.float).to(device)
    else:
        # 合并超边（修正超边ID，避免冲突）
        res_edge[1, :] += len(sen_sign)  # 耐药超边ID偏移（在敏感超边之后）
        hyperedge_index = np.hstack([sen_edge, res_edge])
        sign_weights = np.hstack([sen_sign, res_sign])
        # 转换为tensor
        hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.long).to(device)
        sign_weights = torch.tensor(sign_weights, dtype=torch.float).to(device)

    return hyperedge_index, sign_weights


# -------------------------- 原有工具函数（保留，无修改） --------------------------
def series_num(data):
    """修正ID偏移逻辑：仅用于可视化/日志，特征提取时使用原始ID"""
    data = data.astype(int)
    # 保存原始ID（用于特征提取）
    data_original = data.copy()
    # 偏移仅用于展示，不影响特征索引
    for line in data:
        line[1] += 867  # 药物ID偏移（ncRNA数量：721）
        line[2] = line[2] + 867+ 131 # 疾病ID偏移（ncRNA+药物：721+92）
    return data, data_original  # 返回偏移后+原始ID


def random_seed(seed):
    """设置全局随机种子，保证可复现"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_cpu_memory_usage():
    """获取当前进程的CPU内存使用量（GB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)


def calculate_specificity(y_true, y_pred, class_idx):
    """计算指定类别的特异度（Specificity）"""
    # 特异度 = 真阴性 / (真阴性 + 假阳性)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    tn = np.sum(cm) - (np.sum(cm[class_idx, :]) + np.sum(cm[:, class_idx]) - cm[class_idx, class_idx])
    fp = np.sum(cm[:, class_idx]) - cm[class_idx, class_idx]
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


# -------------------------- 修改：训练函数（移除扰动损失） --------------------------
def train(ncrna_feat, drug_feat, disease_feat, train_data, model, optimizer, loss_fn, device, hyperedge_index,
          sign_weights):
    """训练函数（修改后）：移除扰动损失，仅保留分类损失"""
    model.train()
    start_time = time.time()
    optimizer.zero_grad()

    # 关键修复：使用原始ID（未偏移）提取特征，避免索引越界
    ncrna_ids = torch.tensor(train_data[:, 0], dtype=torch.long).to(device)
    drug_ids = torch.tensor(train_data[:, 1], dtype=torch.long).to(device)
    disease_ids = torch.tensor(train_data[:, 2], dtype=torch.long).to(device)
    labels = torch.tensor(train_data[:, 3], dtype=torch.long).to(device)

    # 模型前向传播（修改：仅接收logits和pred_probs，不再传递labels给超图编码器）
    logits, pred_probs, pred_list = model(
        ncrna_feat, drug_feat, disease_feat,
        ncrna_ids, drug_ids, disease_ids,
        hyperedge_index=hyperedge_index,
        sign_weights=sign_weights
    )

    # 仅保留分类损失（Focal Loss）
    total_loss = loss_fn(logits, labels)

    # 反向传播与优化
    total_loss.backward()
    optimizer.step()
    end_time = time.time()
    # 返回总损失（仅分类损失）、耗时、内存占用
    return total_loss.item(), end_time - start_time, get_cpu_memory_usage()


# -------------------------- 修改：测试函数（新增AUPR计算） --------------------------
def test(ncrna_feat, drug_feat, disease_feat, val_data, model, device, fold_num, result_dir, hyperedge_index,
         sign_weights):
    """测试函数（修改后）：新增AUPR指标计算，移除扰动损失相关逻辑"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # 保存预测概率（用于计算AUC/AUPR）
    start_time = time.time()

    with torch.no_grad():
        # 关键修复：使用原始ID提取特征
        ncrna_ids = torch.tensor(val_data[:, 0], dtype=torch.long).to(device)
        drug_ids = torch.tensor(val_data[:, 1], dtype=torch.long).to(device)
        disease_ids = torch.tensor(val_data[:, 2], dtype=torch.long).to(device)
        labels = val_data[:, 3].astype(int)

        # 模型前向传播（修改：仅接收logits和pred_probs）
        logits, pred_probs, pred_list = model(
            ncrna_feat, drug_feat, disease_feat,
            ncrna_ids, drug_ids, disease_ids,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights
        )
        pred_classes = torch.argmax(pred_probs, dim=1).cpu().numpy()
        pred_probs_np = pred_probs.cpu().numpy()  # 概率值（用于AUC/AUPR）

        # 收集预测和真实标签
        all_preds.extend(pred_classes)
        all_labels.extend(labels)
        all_probs.extend(pred_probs_np)

    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 1. 基础指标（原有）
    accuracy = np.mean(all_preds == all_labels)
    precision_macro = precision_score(all_labels, all_preds, labels=[0, 1, 2], average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, labels=[0, 1, 2], average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, labels=[0, 1, 2], average='macro', zero_division=0)
    precision_micro = precision_score(all_labels, all_preds, labels=[0, 1, 2], average='micro', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, labels=[0, 1, 2], average='micro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, labels=[0, 1, 2], average='micro', zero_division=0)

    # 2. 每个类别的精准率/召回率/F1
    class_metrics = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1, 2], average=None, zero_division=0
    )
    class_precision = {0: class_metrics[0][0], 1: class_metrics[0][1], 2: class_metrics[0][2]}
    class_recall = {0: class_metrics[1][0], 1: class_metrics[1][1], 2: class_metrics[1][2]}
    class_f1 = {0: class_metrics[2][0], 1: class_metrics[2][1], 2: class_metrics[2][2]}

    # 3. 特异度（Specificity）
    specificity_0 = calculate_specificity(all_labels, all_preds, 0)  # 无关联类别的特异度
    specificity_1 = calculate_specificity(all_labels, all_preds, 1)  # 耐药性类别的特异度
    specificity_2 = calculate_specificity(all_labels, all_preds, 2)  # 敏感性类别的特异度

    # 4. Kappa系数（评估分类一致性）
    kappa = cohen_kappa_score(all_labels, all_preds)

    # 5. MCC系数（马修斯相关系数，适合不平衡数据）
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except:
        mcc = 0.0  # 处理极端情况

    # 6. 多分类AUC-ROC（One-vs-Rest）
    try:
        # 标签二值化（3类→3列）
        y_true_bin = label_binarize(all_labels, classes=[0, 1, 2])
        auc_ovo = roc_auc_score(y_true_bin, all_probs, multi_class='ovr', average='micro')  # 修改：AUC用微平均（匹配论文）
    except:
        auc_ovo = 0.0  # 处理只有单个类别时的异常

    # 7. 新增：多分类AUPR（One-vs-Rest + 微平均，匹配论文要求）
    try:
        # 计算每个类别的平均精确率（AP），再按微平均汇总
        aupr_ovr = average_precision_score(
            y_true_bin, all_probs, average='micro'  # 微平均（论文要求）
        )
    except:
        aupr_ovr = 0.0  # 处理极端情况（如类别缺失）

    # 返回扩展指标字典（新增aupr_ovr）
    return {
        # 原有指标
        'preds': all_preds, 'labels': all_labels, 'accuracy': accuracy,
        'precision_macro': precision_macro, 'recall_macro': recall_macro, 'f1_macro': f1_macro,
        'precision_micro': precision_micro, 'recall_micro': recall_micro, 'f1_micro': f1_micro,
        # 论文表格专用分类召回率
        '0_recall': class_recall[0],
        '1_recall': class_recall[1],
        '2_recall': class_recall[2],
        # 其他扩展指标
        'class_precision': class_precision, 'class_recall': class_recall, 'class_f1': class_f1,
        'specificity_0': specificity_0, 'specificity_1': specificity_1, 'specificity_2': specificity_2,
        'kappa': kappa, 'mcc': mcc, 'auc_ovr': auc_ovo, 'aupr_ovr': aupr_ovr,  # 新增：AUPR指标
        'probabilities': all_probs, 'triplets': val_data[:, :3].astype(int),
        'test_time': time.time() - start_time, 'memory': get_cpu_memory_usage()
    }


# -------------------------- 数据处理函数（保留，无修改） --------------------------
def build_oof_prediction_frame(test_results, fold_num):
    probs = np.asarray(test_results['probabilities'])
    triplets = np.asarray(test_results['triplets'])
    return pd.DataFrame({
        'fold': int(fold_num),
        'ncrna_id': triplets[:, 0].astype(int),
        'drug_id': triplets[:, 1].astype(int),
        'disease_id': triplets[:, 2].astype(int),
        'label': np.asarray(test_results['labels']).astype(int),
        'pred_label': np.asarray(test_results['preds']).astype(int),
        'prob_non_association': probs[:, 0],
        'prob_resistance': probs[:, 1],
        'prob_sensitivity': probs[:, 2],
        'confidence': probs.max(axis=1),
    })


def get_train_val_data(all_data, train_ind, val_ind, adj, seed):
    """
    生成训练/验证集的正负样本，返回格式：
    train_pos: 训练集正样本（原始ID）
    train_neg: 训练集负样本（原始ID）
    val_pos: 验证集正样本（原始ID）
    val_neg: 验证集负样本（原始ID）
    """
    # 1. 拆分原始正负样本（正样本：从all_data中提取）
    train_data_pos = all_data[train_ind].copy().astype(int)  # 训练集正样本
    val_data_pos = all_data[val_ind].copy().astype(int)  # 验证集正样本

    # 2. 调用负样本生成函数
    train_data_all, tr_neg_1_ls, te_neg_1_ls = neg_data_generate(adj, train_data_pos, val_data_pos, seed)

    # 3. 过滤负样本（仅保留标签为0的负样本，对齐get_indep_data逻辑）
    train_neg_data = []
    for i in tr_neg_1_ls:
        if list(i)[-1] == 0:  # 确保负样本标签为0
            train_neg_data.append(i)
    val_neg_data = []
    for i in te_neg_1_ls:
        if list(i)[-1] == 0:  # 确保负样本标签为0
            val_neg_data.append(i)

    # 4. 类型转换+标签清洗（确保标签是0/1/2）
    train_data_pos[:, 3] = np.clip(train_data_pos[:, 3], 0, 2)
    val_data_pos[:, 3] = np.clip(val_data_pos[:, 3], 0, 2)
    train_neg_data = np.array(train_neg_data).astype(int) if train_neg_data else np.empty((0, 4), int)
    val_neg_data = np.array(val_neg_data).astype(int) if val_neg_data else np.empty((0, 4), int)
    if len(train_neg_data) > 0:
        train_neg_data[:, 3] = np.clip(train_neg_data[:, 3], 0, 2)
    if len(val_neg_data) > 0:
        val_neg_data[:, 3] = np.clip(val_neg_data[:, 3], 0, 2)

    # 5. ID偏移仅用于展示，特征提取用原始ID
    # 6. 返回格式：训练正、训练负、验证正、验证负（原始ID，无偏移）
    return train_data_pos, train_neg_data, val_data_pos, val_neg_data


# -------------------------- 新增：早停核心函数 --------------------------
def early_stop(current_metric, model, optimizer, fold_num, epoch, best_metric, patience_counter, patience, save_path):
    """
    早停逻辑处理函数：
    - 监控核心指标，更新最佳模型，判断是否触发早停
    参数：
    - current_metric: 当前epoch的监控指标值（此处为MCC）
    - model: 当前模型实例
    - optimizer: 优化器实例（用于保存训练状态）
    - fold_num: 当前交叉验证折数
    - epoch: 当前训练轮数
    - best_metric: 历史最佳指标值
    - patience_counter: 当前耐心累计计数器
    - patience: 预设耐心值（最大累计数）
    - save_path: 最佳模型保存路径
    返回：
    - updated_best_metric: 更新后的最佳指标值
    - updated_patience_counter: 更新后的耐心计数器
    - stop_flag: 是否触发早停（True/False）
    """
    stop_flag = False
    updated_best_metric = best_metric
    updated_patience_counter = patience_counter

    # 1. 当前指标优于历史最佳（MCC初始为-1，数值越大性能越好）
    if current_metric > updated_best_metric:
        updated_best_metric = current_metric
        updated_patience_counter = 0  # 重置耐心计数器
        # 保存最佳模型（包含模型参数、优化器参数、最佳指标、epoch）
        torch.save({
            'fold_num': fold_num,
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': updated_best_metric,
            'metric_name': 'MCC'
        }, save_path)
        print(f"  ✅ 最佳模型更新：Epoch {epoch + 1}，最佳MCC={updated_best_metric:.4f}（已保存至{save_path}）")
    # 2. 当前指标未提升，累计耐心计数器
    else:
        updated_patience_counter += 1
        print(f"  ⚠️  指标未提升，耐心计数器累计：{updated_patience_counter}/{patience}")
        # 3. 计数器达到耐心值，触发早停
        if updated_patience_counter >= patience:
            stop_flag = True
            print(f"  🛑 早停触发！连续{patience}轮指标未提升，终止当前折训练")

    return updated_best_metric, updated_patience_counter, stop_flag


# -------------------------- 修正：主函数（添加AUPR指标记录/打印/保存） --------------------------
if __name__ == '__main__':
    # 1. 参数初始化
    args = parameters_set()
    random_seed(args.seed)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 手动指定使用 cuda:1，替换数字即可切换GPU
    device_name = os.environ.get("SPHLCMGA_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    print(f"===== 实验配置 =====")
    print(f"使用设备: {device}")
    print(f"交叉验证折数: {args.k_fold}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"单模态隐藏维度: {args.modal_hidden_dim}")
    print(f"超图卷积第一层维度: {args.hgnn_dim_1}")  # 修正：单独打印超图参数，更清晰
    print(f"Dropout概率: {args.dropout}")  # 新增：打印dropout参数

    # -------------------------- 新增：早停超参数配置 --------------------------
    PATIENCE = 50  # 早停耐心值（连续50轮指标未提升则终止）
    MONITOR_METRIC = 'f1_macro'  # 监控核心指标（可改为aupr_ovr）
    BEST_MODEL_DIR = BEST_MODEL_DIR_OVERRIDE or os.path.join(RESULT_DIR_OVERRIDE or 'Data', 'best_models')  # 最佳模型保存目录
    if not os.path.exists(BEST_MODEL_DIR):
        os.makedirs(BEST_MODEL_DIR)
    print(f"早停配置: 耐心值={PATIENCE}，监控指标={MONITOR_METRIC}，最佳模型保存至={BEST_MODEL_DIR}")

    # 关键校验：确保modal_out_dim是4的整数倍（满足多头注意力要求）
    assert args.modal_out_dim % 4 == 0, f"modal_out_dim({args.modal_out_dim})必须是4的整数倍，以适配多头注意力"
    print(f"单模态输出维度: {args.modal_out_dim}（符合4的整数倍要求）")

    # 2. 加载数据（确保Data目录下有对应文件）
    try:
        # 加载特征数据（选择其中一种特征组合即可，注释部分为备用）
        # disease_sim1 = np.loadtxt('Data/LLM_disease_sim.txt')
        # drug_sim1 = np.loadtxt('Data/LLM_drug_sim.txt')
        # ncrna_sim1 = np.loadtxt('Data/LLM_rna_sim.txt')
        #
        # disease_sim2 = np.loadtxt('Data/disease.txt')
        # drug_sim2 = np.loadtxt('Data/drug.txt')
        # ncrna_sim2 = np.loadtxt('Data/ncRNA.txt')
        #
        # disease_sim = (disease_sim1 + disease_sim2)/2
        # drug_sim = (drug_sim1 + drug_sim2)/2
        # ncrna_sim = (ncrna_sim1 + ncrna_sim2)/2

        #**************消融*************
        # disease_sim = np.loadtxt('Data/disease.txt')
        # drug_sim = np.loadtxt('Data/drug.txt')
        # ncrna_sim = np.loadtxt('Data/ncRNA.txt')

        # disease_sim = np.loadtxt('Data/LLM_disease_sim.txt')
        # drug_sim = np.loadtxt('Data/LLM_drug_sim.txt')
        # ncrna_sim = np.loadtxt('Data/LLM_rna_sim.txt')

        # **************数据集2*************
        disease_sim1 = np.loadtxt('Data2/disease.txt')
        drug_sim1 = np.loadtxt('Data2/drug.txt')
        ncrna_sim1 = np.loadtxt('Data2/RNA.txt')

        disease_sim2 = np.loadtxt('Data2/LLM_disease_sim.txt')
        drug_sim2 = np.loadtxt('Data2/LLM_drug_sim.txt')
        ncrna_sim2 = np.loadtxt('Data2/LLM_rna_sim.txt')

        disease_sim = (disease_sim1 + disease_sim2)/2
        drug_sim = (drug_sim1 + drug_sim2)/2
        ncrna_sim = (ncrna_sim1 + ncrna_sim2)/2
        #**************消融*************
        # disease_sim = np.loadtxt('Data2/disease.txt')
        # drug_sim = np.loadtxt('Data2/drug.txt')
        # ncrna_sim = np.loadtxt('Data2/RNA.txt')

        # disease_sim = np.loadtxt('Data2/LLM_disease_sim.txt')
        # drug_sim = np.loadtxt('Data2/LLM_drug_sim.txt')
        # ncrna_sim = np.loadtxt('Data2/LLM_rna_sim.txt')
        adj_data = np.loadtxt('Data2/association.txt')
        # 提取第四列（numpy数组索引从0开始，所以第四列是索引3）
        fourth_column = adj_data[:, 3]

        # 统计1的数量：利用布尔数组求和（True=1，False=0）
        count_1 = np.sum(fourth_column == 1)
        # 统计2的数量
        count_2 = np.sum(fourth_column == 2)

        # 输出结果
        print(f"第四列中数值1的数量：{count_1}")
        print(f"第四列中数值2的数量：{count_2}")
        print(f"\n数据加载成功！")
        print(f"ncRNA特征维度: {ncrna_sim.shape} (数量: {ncrna_sim.shape[0]})")
        print(f"药物特征维度: {drug_sim.shape} (数量: {drug_sim.shape[0]})")
        print(f"疾病特征维度: {disease_sim.shape} (数量: {disease_sim.shape[0]})")
        print(f"关联数据条数: {len(adj_data)}")
        # 记录实体数量（用于超图构建和模型配置）
        args.ncrna_num = ncrna_sim.shape[0]
        args.drug_num = drug_sim.shape[0]
        args.disease_num = disease_sim.shape[0]
    except FileNotFoundError as e:
        print(f"数据文件缺失: {e}")
        exit(1)

    # 验证标签有效性（必须是0/1/2）
    assert np.all(np.isin(adj_data[:, 3], [0, 1, 2])), "标签必须是0（无关联）、1（耐药）、2（敏感）"

    # 数据打乱并拆分（80%用于交叉验证，20%预留）
    np.random.shuffle(adj_data)
    # cv_data = adj_data[int(0.2 * len(adj_data)):, :]
    cv_data = adj_data
    # 转换特征为tensor并移到指定设备
    ncrna_feat = torch.from_numpy(ncrna_sim).type(torch.FloatTensor).to(device)
    drug_feat = torch.from_numpy(drug_sim).type(torch.FloatTensor).to(device)
    disease_feat = torch.from_numpy(disease_sim).type(torch.FloatTensor).to(device)

    # 3. 五折交叉验证初始化（扩展指标存储，新增AUPR）
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    fold_final_metrics = {
        # 原有基础指标
        'accuracy': [], 'precision_macro': [], 'recall_macro': [],
        'f1_macro': [], 'precision_micro': [], 'recall_micro': [], 'f1_micro': [],
        # 论文核心指标（新增aupr_ovr）
        'mcc': [], 'auc_ovr': [], 'aupr_ovr': [],
        # 论文表格分类召回率
        'class0_recall': [], 'class1_recall': [], 'class2_recall': [],
        # 其他扩展指标
        'kappa': [], 'specificity_0': [], 'specificity_1': [], 'specificity_2': [],
        'class0_precision': [], 'class0_recall': [], 'class0_f1': [],
        'class1_precision': [], 'class1_recall': [], 'class1_f1': [],
        'class2_precision': [], 'class2_recall': [], 'class2_f1': []
    }
    fold_final_reports = []
    oof_prediction_frames = []
    # 结果保存路径
    result_dir = RESULT_DIR_OVERRIDE or 'Data'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file = os.path.join(result_dir, 'fold_final_results_paper.txt')
    report_file = os.path.join(result_dir, 'fold_final_classification_reports.txt')

    # 清空结果文件
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('双分支模型（交叉注意力+超图卷积+Focal Loss+早停） - 论文指标输出结果（百分比，均值±标准差）\n')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('双分支模型（交叉注意力+超图卷积+Focal Loss+早停） - 五折分类报告\n')

    # 4. 遍历每一折
    fold_num = 0
    for train_index, val_index in kf.split(cv_data):
        fold_num += 1
        print(f"\n==================== 第{fold_num}折交叉验证 ====================")

        # 加载数据（返回训练正、训练负、验证正、验证负）
        train_pos, train_neg, val_pos, val_neg = get_train_val_data(
            cv_data, train_index, val_index, adj_data, args.seed
        )

        # 拼接训练集（正+负）、验证集（正+负）并打乱
        train_data = np.vstack([train_pos, train_neg]) if len(train_neg) > 0 else train_pos
        val_data = np.vstack([val_pos, val_neg]) if len(val_neg) > 0 else val_pos
        np.random.shuffle(train_data)
        np.random.shuffle(val_data)
        print("Train labels unique:", np.unique(train_data[:, 3]))
        print("Val labels unique:", np.unique(val_data[:, 3]))
        # 打印数据维度和ID范围
        print(f"第{fold_num}折数据规模：")
        print(f"训练集正样本: {len(train_pos)}, 负样本: {len(train_neg)}, 总计: {len(train_data)}")
        print(f"验证集正样本: {len(val_pos)}, 负样本: {len(val_neg)}, 总计: {len(val_data)}")
        print(
            f"训练集ID范围 - ncRNA: [{train_data[:, 0].min()}, {train_data[:, 0].max()}], 药物: [{train_data[:, 1].min()}, {train_data[:, 1].max()}], 疾病: [{train_data[:, 2].min()}, {train_data[:, 2].max()}]")
        print(f"特征矩阵维度 - ncRNA: {ncrna_feat.shape[0]}, 药物: {drug_feat.shape[0]}, 疾病: {disease_feat.shape[0]}")

        # 修改：构建关系超图（返回合并后的hyperedge_index和sign_weights）
        hyperedge_index, sign_weights = build_relational_hypergraph(
            train_data, args.ncrna_num, args.drug_num, args.disease_num, device
        )
        total_hyperedges = len(sign_weights)
        resistant_edges = (sign_weights == -1).sum().item()
        sensitive_edges = (sign_weights == +1).sum().item()
        print(f"关系超图构建完成：总超边数={total_hyperedges}, 耐药超边数={resistant_edges}, 敏感超边数={sensitive_edges}")

        # 核心修正1：模型初始化（适配双分支模型，补充缺失的dropout参数）
        model = NcrnaDrugDiseaseModel(
            ncrna_dim=ncrna_sim.shape[1],
            drug_dim=drug_sim.shape[1],
            disease_dim=disease_sim.shape[1],
            modal_hidden_dim=args.modal_hidden_dim,
            modal_out_dim=args.modal_out_dim,
            classifier_hidden_dim=args.classifier_hidden_dim,
            hgnn_dim_1=args.hgnn_dim_1,
            dropout=args.dropout  # 补充：传入dropout参数，匹配双分支模型__init__
        ).to(device)

        # 移除set_entity_nums调用（不再需要实体数量计算扰动损失）
        print("模型初始化完成，已移除扰动对比损失相关逻辑")
        # 使用带alpha和更大gamma的Focal Loss
        loss_fn = FocalLoss(gamma=1, reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # -------------------------- 新增：当前折早停变量初始化 --------------------------
        best_metric = -1.0  # 初始化最佳指标（MCC最小值为-1）
        patience_counter = 0  # 初始化耐心计数器
        best_model_path = os.path.join(BEST_MODEL_DIR, f"best_model_fold_{fold_num}.pth")  # 当前折最佳模型路径
        final_test_results = None  # 最终结果（最佳模型评估结果）

        # 训练过程
        fold_epoch_losses = []
        fold_epoch_accuracies = []
        fold_epoch_mcc = []  # 新增：记录每轮MCC，用于可视化

        for e in range(args.epochs):
            print(f"\n--- 第{fold_num}折 | Epoch {e + 1}/{args.epochs} ---")
            # 训练（修改：移除扰动损失相关返回值）
            try:
                total_loss, train_time, train_mem = train(
                    ncrna_feat, drug_feat, disease_feat, train_data,
                    model, optimizer, loss_fn, device,
                    hyperedge_index=hyperedge_index,  # 传入合并后的超边索引
                    sign_weights=sign_weights  # 传入符号权重
                )
            except Exception as e_train:
                print(f"训练出错: {e_train}")
                break
            # 测试（修改：传入正确的超图参数）
            test_results = test(
                ncrna_feat, drug_feat, disease_feat, val_data,
                model, device, fold_num, result_dir,
                hyperedge_index=hyperedge_index,  # 传入合并后的超边索引
                sign_weights=sign_weights  # 传入符号权重
            )

            # 记录指标
            fold_epoch_losses.append(total_loss)
            fold_epoch_accuracies.append(test_results['accuracy'])
            fold_epoch_mcc.append(test_results['mcc'])  # 记录MCC

            # 打印日志（补充AUPR信息，移除扰动损失）
            print(f"Total Loss: {total_loss:.6f}")
            print(f"耗时: {train_time:.2f}s | 内存: {train_mem:.2f}GB")
            print(f"基础指标 - Accuracy: {test_results['accuracy']:.4f} | Macro F1: {test_results['f1_macro']:.4f}")
            print(
                f"论文核心指标 - MCC: {test_results['mcc']:.4f} | AUC-ROC: {test_results['auc_ovr']:.4f} | AUPR: {test_results['aupr_ovr']:.4f}")  # 新增AUPR
            print(
                f"分类召回率 - 无关联: {test_results['0_recall']:.4f} | 耐药性: {test_results['1_recall']:.4f} | 敏感性: {test_results['2_recall']:.4f}")

            # -------------------------- 新增：调用早停函数，判断是否终止训练 --------------------------
            current_metric = test_results[MONITOR_METRIC]
            best_metric, patience_counter, stop_flag = early_stop(
                current_metric=current_metric,
                model=model,
                optimizer=optimizer,
                fold_num=fold_num,
                epoch=e,
                best_metric=best_metric,
                patience_counter=patience_counter,
                patience=PATIENCE,
                save_path=best_model_path
            )

            # 触发早停，跳出epoch循环
            if stop_flag:
                break

        # -------------------------- 新增：加载最佳模型，评估最终结果 --------------------------
        print(f"\n=== 第{fold_num}折：加载最佳模型进行最终评估 ===")
        # 重新初始化模型（保证加载权重的兼容性）
        best_model = NcrnaDrugDiseaseModel(
            ncrna_dim=ncrna_sim.shape[1],
            drug_dim=drug_sim.shape[1],
            disease_dim=disease_sim.shape[1],
            modal_hidden_dim=args.modal_hidden_dim,
            modal_out_dim=args.modal_out_dim,
            classifier_hidden_dim=args.classifier_hidden_dim,
            hgnn_dim_1=args.hgnn_dim_1,
            dropout=args.dropout
        ).to(device)

        # 加载最佳模型权重
        # 修正后（添加weights_only=True）
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        best_epoch = checkpoint['epoch']
        best_final_metric = checkpoint['best_metric']
        print(f"最佳模型信息：Epoch {best_epoch}，最佳{MONITOR_METRIC.upper()}={best_final_metric:.4f}")

        # 使用最佳模型评估验证集
        final_test_results = test(
            ncrna_feat, drug_feat, disease_feat, val_data,
            best_model, device, fold_num, result_dir,
            hyperedge_index=hyperedge_index,
            sign_weights=sign_weights
        )

        # 处理该折结果（新增AUPR记录）
        if final_test_results is not None:
            # 记录原有基础指标
            fold_final_metrics['accuracy'].append(final_test_results['accuracy'])
            fold_final_metrics['precision_macro'].append(final_test_results['precision_macro'])
            fold_final_metrics['recall_macro'].append(final_test_results['recall_macro'])
            fold_final_metrics['f1_macro'].append(final_test_results['f1_macro'])
            fold_final_metrics['precision_micro'].append(final_test_results['precision_micro'])
            fold_final_metrics['recall_micro'].append(final_test_results['recall_micro'])
            fold_final_metrics['f1_micro'].append(final_test_results['f1_micro'])

            # 记录论文核心指标（新增aupr_ovr）
            fold_final_metrics['mcc'].append(final_test_results['mcc'])
            fold_final_metrics['auc_ovr'].append(final_test_results['auc_ovr'])
            fold_final_metrics['aupr_ovr'].append(final_test_results['aupr_ovr'])  # 新增
            fold_final_metrics['class0_recall'].append(final_test_results['0_recall'])
            fold_final_metrics['class1_recall'].append(final_test_results['1_recall'])
            fold_final_metrics['class2_recall'].append(final_test_results['2_recall'])

            # 记录其他扩展指标
            fold_final_metrics['kappa'].append(final_test_results['kappa'])
            fold_final_metrics['specificity_0'].append(final_test_results['specificity_0'])
            fold_final_metrics['specificity_1'].append(final_test_results['specificity_1'])
            fold_final_metrics['specificity_2'].append(final_test_results['specificity_2'])
            fold_final_metrics['class0_precision'].append(final_test_results['class_precision'][0])
            fold_final_metrics['class0_f1'].append(final_test_results['class_f1'][0])
            fold_final_metrics['class1_precision'].append(final_test_results['class_precision'][1])
            fold_final_metrics['class1_f1'].append(final_test_results['class_f1'][1])
            fold_final_metrics['class2_precision'].append(final_test_results['class_precision'][2])
            fold_final_metrics['class2_f1'].append(final_test_results['class_f1'][2])

            # 生成分类报告
            final_report = classification_report(
                final_test_results['labels'], final_test_results['preds'],
                labels=[0, 1, 2], target_names=['无关联', '耐药性', '敏感性'], zero_division=0
            )
            fold_final_reports.append(final_report)

            if EXPORT_OOF_PREDICTIONS:
                fold_oof = build_oof_prediction_frame(final_test_results, fold_num)
                oof_prediction_frames.append(fold_oof)
                fold_oof_path = os.path.join(result_dir, f"oof_predictions_fold_{fold_num}.csv")
                fold_oof.to_csv(fold_oof_path, index=False)

            # 打印该折最终结果（新增AUPR）
            print(f"\n=== 第{fold_num}折最佳模型评估结果（论文格式，百分比）===")
            print(f"Accuracy: {final_test_results['accuracy'] * 100:.2f}%")
            print(f"Precision: {final_test_results['precision_macro'] * 100:.2f}%")
            print(f"Recall: {final_test_results['recall_macro'] * 100:.2f}%")
            print(f"F1: {final_test_results['f1_macro'] * 100:.2f}%")
            print(f"MCC: {final_test_results['mcc'] * 100:.2f}%")
            print(f"ROC_AUC: {final_test_results['auc_ovr'] * 100:.2f}%")
            print(f"AUPR: {final_test_results['aupr_ovr'] * 100:.2f}%")  # 新增
            print(f"0_recall: {final_test_results['0_recall'] * 100:.2f}%")
            print(f"1_recall: {final_test_results['1_recall'] * 100:.2f}%")
            print(f"2_recall: {final_test_results['2_recall'] * 100:.2f}%")
            print(f"\n分类报告:\n{final_report}")

            # 保存分类报告到文件
            with open(report_file, 'a', encoding='utf-8') as f:
                f.write(f"\n===== 第{fold_num}折最佳模型分类报告（Epoch {best_epoch}）=====\n{final_report}")

    # 5. 统计五折结果（新增AUPR）
    print("\n==================== 论文表格指标（五折平均） ====================")
    # 论文表格列与代码指标的映射（转百分比，新增AUPR）
    paper_metrics = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision_macro"),
        ("Recall", "recall_macro"),
        ("F1", "f1_macro"),
        ("MCC", "mcc"),
        ("ROC_AUC", "auc_ovr"),
        ("AUPR", "aupr_ovr"),  # 新增
        ("0_recall", "class0_recall"),
        ("1_recall", "class1_recall"),
        ("2_recall", "class2_recall")
    ]

    # 计算均值和标准差（转百分比）
    paper_stats = {}
    for metric_name, metric_key in paper_metrics:
        values = np.array(fold_final_metrics[metric_key]) * 100  # 转百分比
        mean_val = np.mean(values)
        std_val = np.std(values)
        paper_stats[metric_name] = (mean_val, std_val)

    # 打印论文表格格式结果
    print(f"{'指标名称':<10} {'均值(%)':<10} {'标准差(%)':<10}")
    print("-" * 30)
    for metric_name, (mean_val, std_val) in paper_stats.items():
        print(f"{metric_name:<10} {mean_val:.2f}      {std_val:.2f}")

    # 打印紧凑的论文表格单行复制版（新增AUPR）
    print("\n===== 论文表格单行复制版 =====")
    table_row = "DualBranch(CrossAttention+Hypergraph+FocalLoss+EarlyStop)"
    for metric_name in paper_metrics:
        mean_val = paper_stats[metric_name[0]][0]
        table_row += f" & {mean_val:.2f}"
    table_row += " \\\\"
    print(table_row)

    # 保存论文指标到文件（新增AUPR）
    with open(result_file, 'a', encoding='utf-8') as f:
        f.write("\n===== 论文表格指标（五折平均，百分比） =====\n")
        f.write(f"{'指标名称':<10} {'均值(%)':<10} {'标准差(%)':<10}\n")
        f.write("-" * 30 + "\n")
        for metric_name, (mean_val, std_val) in paper_stats.items():
            f.write(f"{metric_name:<10} {mean_val:.2f}      {std_val:.2f}\n")
        f.write(f"\n论文表格单行复制版:\n{table_row}\n")

    # 6. 输出关键结论（新增AUPR）
    print(f"\n===== 关键结论 =====")
    print(f"五折平均准确率: {paper_stats['Accuracy'][0]:.2f}% ± {paper_stats['Accuracy'][1]:.2f}%")
    print(f"五折平均F1值: {paper_stats['F1'][0]:.2f}% ± {paper_stats['F1'][1]:.2f}%")
    print(f"五折平均AUC-ROC: {paper_stats['ROC_AUC'][0]:.2f}% ± {paper_stats['ROC_AUC'][1]:.2f}%")
    print(f"五折平均AUPR: {paper_stats['AUPR'][0]:.2f}% ± {paper_stats['AUPR'][1]:.2f}%")  # 新增
    print(f"五折平均MCC: {paper_stats['MCC'][0]:.2f}% ± {paper_stats['MCC'][1]:.2f}%")