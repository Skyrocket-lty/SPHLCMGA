# import random
# import numpy as np
# neg_train_num = 1
# neg_test_num = 1
# def neg_data_generate(adj_data_all,train_data_fix,val_data_fix,seed):
#     random.seed(seed)
#     train_neg_ls_all = []
#     val_neg_1_ls = []
#     # arr_true = np.zeros((190,219,163))
#     arr_true = np.zeros((721, 92, 27))
#     for line in adj_data_all:
#         arr_true[int(line[0]), int(line[1]), int(line[2])] = 1
#     arr_false_train = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])),
#                                 len(set(adj_data_all[:,2]))))
#
#     L1 = 0
#     for i in train_data_fix:
#         L1 += 1
#         k1 = 0
#         tr_diet_ls = [j for j in range(0, arr_true.shape[0])]
#         tr_mic_ls = [j for j in range(0, arr_true.shape[1])]
#         tr_dis_ls = [j for j in range(0, arr_true.shape[2])]
#         while k1 < neg_train_num:
#             a = int(i[0])
#             b = random.randint(0, arr_true.shape[1] - 1)
#             c = int(i[2])
#             if arr_true[a, b, c] != 1 and arr_false_train[a, b, c] != 1:
#                 arr_false_train[a, b, c] = 1
#                 k1 += 1
#                 train_neg_ls_all.append((a, b, c, 0))
#             else:
#                 distance_t2 = neg_train_num - k1
#                 # print('triplet:', i, 'tr_4:', distance_t4)
#                 last_ind = len(train_neg_ls_all) - 1
#                 for k in range(distance_t2):
#                     train_neg_ls_all.append(train_neg_ls_all[last_ind])
#                 break
#
#     train_neg_all = np.array(train_neg_ls_all)
#     train_data_all = np.vstack((np.array(train_neg_ls_all), train_data_fix))
#     np.random.shuffle(train_neg_all)
#     np.random.shuffle(train_data_all)
#     L2 = 0
#     for i in val_data_fix:
#         t1 = 0
#         neg_1_i = []
#         # Because it is too easy to repeat, it is only guaranteed that multiple negative samples generated for a
#         # certain positive sample are not repeated, and negative samples of different positive samples may be repeated.
#         arr_false_val_1 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
#         neg_1_i.append(i)
#         L2 += 1
#         arr_false_val_2 = np.zeros((len(set(adj_data_all[:,0])), len(set(adj_data_all[:,1])), len(set(adj_data_all[:,2]))))
#         diet_ls = [j for j in range(0, arr_true.shape[0])]
#         mic_ls = [j for j in range(0, arr_true.shape[1])]
#         dis_ls = [j for j in range(0, arr_true.shape[2])]
#         while t1 < neg_test_num:
#             a_3 = int(i[0])
#             c_3 = int(i[2])
#             if mic_ls != []:
#                 b_3 = random.choice(mic_ls)
#                 mic_ls.remove(b_3)
#                 if arr_true[a_3, b_3, c_3] != 1 and arr_false_train[a_3, b_3, c_3] != 1 and arr_false_val_1[
#                     a_3, b_3, c_3] != 1:
#                     arr_false_val_1[a_3, b_3, c_3] = 1
#                     t1 += 1
#                     neg_1_i.append((a_3, b_3, c_3, 0))
#             else:
#                 distance_3 = neg_test_num - t1
#                 # print('triplet:', i, 'val_3:', distance_3)
#                 last_ind = len(neg_1_i) - 1
#                 for k in range(distance_3):
#                     neg_1_i.append(neg_1_i[last_ind])
#                 break
#         np.random.shuffle(neg_1_i)
#         val_neg_1_ls.extend(neg_1_i)
#         # print('fold_num:', fold_num, 'neg_2:', neg_2, 'neg_3:', neg_3, 'neg_4:', neg_4)
#     return train_data_all, train_neg_all, val_neg_1_ls
import numpy as np

neg_num_train = 1
neg_num_test = 1


def _infer_entity_shape(*arrays):
    """Infer tensor shape from the observed triplet ids instead of hardcoding dataset sizes."""
    max_ids = np.array([0, 0, 0], dtype=int)
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        max_ids = np.maximum(max_ids, arr[:, :3].astype(int).max(axis=0))
    return tuple((max_ids + 1).tolist())

def neg_data_generate(adj_data_all, train_data_fix, val_data_fix, seed):
    """
    Three-class negative sample generation
    label:
        1 -> resistance
        2 -> sensitivity
        0 -> no association
    """
    np.random.seed(seed)
    # =========================
    # containers
    # =========================
    train_neg_1_ls, train_neg_2_ls = [], []
    train_neg_3_ls, train_neg_4_ls = [], []

    val_neg_1_ls, val_neg_2_ls = [], []
    val_neg_3_ls, val_neg_4_ls = [], []

    # =========================
    # true association tensor
    # =========================
    # 0: no association
    # 1: resistance
    # 2: sensitivity
    arr_true = np.zeros(_infer_entity_shape(adj_data_all, train_data_fix, val_data_fix))
    for line in adj_data_all:
        a, b, c, label = int(line[0]), int(line[1]), int(line[2]), int(line[3])
        arr_true[a, b, c] = label

    # mark generated train negatives (avoid duplication)
    arr_false_train = np.zeros(arr_true.shape)

    # =========================
    # -------- TRAIN ----------
    # =========================
    for i in train_data_fix:
        a_pos, b_pos, c_pos = int(i[0]), int(i[1]), int(i[2])

        ctn_1 = ctn_2 = ctn_3 = ctn_4 = 0

        gene_ls = list(range(arr_true.shape[0]))
        drug_ls = list(range(arr_true.shape[1]))
        dis_ls = list(range(arr_true.shape[2]))

        # ---- Type 1: fully random ----
        while ctn_1 < neg_num_train:
            a = np.random.randint(0, arr_true.shape[0])
            b = np.random.randint(0, arr_true.shape[1])
            c = np.random.randint(0, arr_true.shape[2])

            if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1:
                arr_false_train[a, b, c] = 1
                train_neg_1_ls.append((a, b, c, 0))
                ctn_1 += 1

        # ---- Type 2: fix drug & disease, change gene ----
        while ctn_2 < neg_num_train:
            if gene_ls:
                a = np.random.choice(gene_ls)
                gene_ls.remove(a)
                b, c = b_pos, c_pos
                if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    train_neg_2_ls.append((a, b, c, 0))
                    ctn_2 += 1
            else:
                break

        # ---- Type 3: fix gene & disease, change drug ----
        while ctn_3 < neg_num_train:
            if drug_ls:
                b = np.random.choice(drug_ls)
                drug_ls.remove(b)
                a, c = a_pos, c_pos
                if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    train_neg_3_ls.append((a, b, c, 0))
                    ctn_3 += 1
            else:
                break

        # ---- Type 4: fix gene & drug, change disease ----
        while ctn_4 < neg_num_train:
            if dis_ls:
                c = np.random.choice(dis_ls)
                dis_ls.remove(c)
                a, b = a_pos, b_pos
                if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1:
                    arr_false_train[a, b, c] = 1
                    train_neg_4_ls.append((a, b, c, 0))
                    ctn_4 += 1
            else:
                break

    # merge train data
    train_neg_all = np.vstack((
        np.array(train_neg_1_ls),
        np.array(train_neg_2_ls),
        np.array(train_neg_3_ls),
        np.array(train_neg_4_ls),
        train_data_fix      # labels are 1 or 2
    ))
    np.random.shuffle(train_neg_all)

    # =========================
    # -------- VALID ----------
    # =========================
    for i in val_data_fix:
        a_pos, b_pos, c_pos = int(i[0]), int(i[1]), int(i[2])

        gene_ls = list(range(arr_true.shape[0]))
        drug_ls = list(range(arr_true.shape[1]))
        dis_ls = list(range(arr_true.shape[2]))

        arr_false_val_1 = np.zeros(arr_true.shape)
        arr_false_val_2 = np.zeros(arr_true.shape)
        arr_false_val_3 = np.zeros(arr_true.shape)
        arr_false_val_4 = np.zeros(arr_true.shape)

        # keep positive sample
        neg_1_i, neg_2_i = [i], [i]
        neg_3_i, neg_4_i = [i], [i]

        cva_1 = cva_2 = cva_3 = cva_4 = 0

        # ---- Type 1 ----
        while cva_1 < neg_num_test:
            a = np.random.randint(0, arr_true.shape[0])
            b = np.random.randint(0, arr_true.shape[1])
            c = np.random.randint(0, arr_true.shape[2])
            if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1 \
               and arr_false_val_1[a, b, c] != 1:
                arr_false_val_1[a, b, c] = 1
                neg_1_i.append((a, b, c, 0))
                cva_1 += 1
        np.random.shuffle(neg_1_i)
        val_neg_1_ls.extend(neg_1_i)

        # ---- Type 2 ----
        while cva_2 < neg_num_test and gene_ls:
            a = np.random.choice(gene_ls)
            gene_ls.remove(a)
            b, c = b_pos, c_pos
            if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1 \
               and arr_false_val_2[a, b, c] != 1:
                arr_false_val_2[a, b, c] = 1
                neg_2_i.append((a, b, c, 0))
                cva_2 += 1
        np.random.shuffle(neg_2_i)
        val_neg_2_ls.extend(neg_2_i)

        # ---- Type 3 ----
        while cva_3 < neg_num_test and drug_ls:
            b = np.random.choice(drug_ls)
            drug_ls.remove(b)
            a, c = a_pos, c_pos
            if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1 \
               and arr_false_val_3[a, b, c] != 1:
                arr_false_val_3[a, b, c] = 1
                neg_3_i.append((a, b, c, 0))
                cva_3 += 1
        np.random.shuffle(neg_3_i)
        val_neg_3_ls.extend(neg_3_i)

        # ---- Type 4 ----
        while cva_4 < neg_num_test and dis_ls:
            c = np.random.choice(dis_ls)
            dis_ls.remove(c)
            a, b = a_pos, b_pos
            if arr_true[a, b, c] == 0 and arr_false_train[a, b, c] != 1 \
               and arr_false_val_4[a, b, c] != 1:
                arr_false_val_4[a, b, c] = 1
                neg_4_i.append((a, b, c, 0))
                cva_4 += 1
        np.random.shuffle(neg_4_i)
        val_neg_4_ls.extend(neg_4_i)

    return train_neg_all, train_neg_1_ls, val_neg_1_ls


#  train_neg_all     代表训练集的正+负样本（四种）

#  train_neg_1_ls    代表纯负样本（第一种）

#  val_neg_1_ls    代表验证集的正+负样本（第一种）

