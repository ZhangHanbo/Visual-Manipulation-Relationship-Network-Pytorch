import torch
import itertools
import torch.nn.functional as F

def gen_adj_matrix(MaxRel, obj_num_img):
    idx = 0
    adj_matrix = MaxRel.new(obj_num_img, obj_num_img).zero_()
    # print 'sb', obj_num_img
    for i in range(obj_num_img):
        for j in range(i + 1, obj_num_img):
            adj_matrix[i, j] = MaxRel[idx]
            idx += 1
            adj_matrix[j, i] = MaxRel[idx]
            idx += 1
    return adj_matrix

def decode(idx_target, obj_num_img):
    idx = 0
    for i in range(obj_num_img):
        for j in range(i + 1, obj_num_img):
            if idx == idx_target:
                return i, j
            idx += 1
            if idx == idx_target:
                return j, i
            idx += 1

def decode_all(obj_num_img):
    decode_list = []
    for i in range(obj_num_img):
        for j in range(i + 1, obj_num_img):
            decode_list.append((i, j))
            decode_list.append((j, i))
    return decode_list

def check_con(adj_matrix, src, des, rel_c):
    if src == des:
        return True
    visited_vexes = []
    visiting_vexes = [src]
    while (len(visiting_vexes)):
        new_vexes = [vex.item() for vex in (adj_matrix[visiting_vexes[0]] == rel_c).nonzero()
                     if vex.item() not in (visited_vexes + visiting_vexes)]
        if des in new_vexes:
            return True
        visiting_vexes += new_vexes
        visited_vexes.append(visiting_vexes.pop(0))
    return False

def GetConArray(x, obj_num_img):
    if x.shape[0] == 0:
        return x
    _, MaxRel = torch.max(x, dim=1)
    adj_matrix = gen_adj_matrix(MaxRel, obj_num_img)
    ConArray = MaxRel.new(obj_num_img, obj_num_img, 2).zero_().byte()
    for i in range(obj_num_img):
        for j in range(obj_num_img):
            ConArray[i, j, 0] = int(check_con(adj_matrix, i, j, 0))
            ConArray[i, j, 1] = int(check_con(adj_matrix, i, j, 1))
    return ConArray

def RelaTransform(Array):
    '''Transform the number of relationship categories from two to four'''
    ArrayTran = Array.clone()
    for i in range(Array.shape[0]):
        shape = torch.nonzero(Array[i]).max().item() + 1
        Label = range(shape)
        Label_perm = list(itertools.permutations(Label, 2))
        for Label_pair in Label_perm:
            temp = ArrayTran[i, Label_pair[0], Label_pair[1]]
            if temp != 1 and temp != 2:
                if check_con(Array[i], Label_pair[0], Label_pair[1], 1):
                    ArrayTran[i, Label_pair[0], Label_pair[1]] = 4
                    ArrayTran[i, Label_pair[1], Label_pair[0]] = 5
    return ArrayTran

def crf(x, obj_num, iter_num, Add_online):
    x_s = x.clone()
    pair_num = obj_num * (obj_num - 1)
    start_img = pair_num.size(0) / 2 * int(not Add_online)
    for idx in range(start_img, pair_num.size(0)):
        start_po = torch.sum(pair_num[:idx])
        end_po = torch.sum(pair_num[:idx + 1])
        obj_num_img = obj_num[idx].item()
        decode_list = decode_all(obj_num_img)
        ConArray = GetConArray(x[start_po:end_po], obj_num_img)
        for i in range(iter_num):
            x[start_po:end_po] = crf_single_img(decode_list, ConArray, x_s[start_po:end_po], x[start_po:end_po],
                                                 obj_num_img)
    return x

def crf_single_img(decode_list, ConArray, x_s, x, obj_num_img):
    '''
    x: The unary function of all relationships in an image. shape: [num_relationship, 5]
    '''
    # Q = F.softmax(Q)
    # print(x.size(0), obj_num_img)
    ra = 0.5
    if x.shape[0] == 0:
        return x
    Q = x.detach()
    # Q = x
    # Q.requires_grad = False
    Q = F.softmax(Q, dim=1)
    # print Q.shape
    # raise NameError
    _, MaxRel = torch.max(x, dim=1)
    # adj_matrix = gen_adj_matrix(MaxRel, obj_num_img)

    E_p = torch.zeros_like(Q)
    for r_idx in range(x.size(0)):  # r_idx: idx of relationship
        src1, des1 = decode_list[r_idx]
        # src1, des1 = decode(r_idx, obj_num_img)
        for c_idx in range(x.size(1)):  # c_idx: idx of relationship class
            if r_idx % 2 == 0:  # This code block adds pair function according symmetry.
                if c_idx == 2:
                    E_p[r_idx, c_idx] += 2.5 * ra * Q[r_idx + 1, c_idx]  # For symmetry of no relationship
                else:
                    E_p[r_idx, c_idx] += 2.5 * ra * Q[r_idx + 1, abs(c_idx / 3 * 4 - (1 - c_idx % 2))]
                    # For symmetry of father-child and far father- far child
            else:
                if c_idx == 2:
                    E_p[r_idx, c_idx] += 2.5 * ra * Q[r_idx - 1, c_idx]  # For symmetry of no relationship
                else:
                    E_p[r_idx, c_idx] += 2.5 * ra * Q[r_idx - 1, abs(c_idx / 3 * 4 - (1 - c_idx % 2))]
                    # For symmetry of father-child and far father- far child
            if c_idx != 2:
                for rr_idx in range(x.size(0)):
                    if rr_idx != r_idx:
                        src2, des2 = decode_list[rr_idx]
                        # src2, des2 = decode(rr_idx, obj_num_img)
                        # tmp = adj_matrix[src2, des2]
                        if c_idx == 0:
                            # adj_matrix[src2, des2] = 0
                            # Penalty for redundant edges
                            if ConArray[src1, src2, 0] and ConArray[des2, des1, 0]:
                                E_p[r_idx, c_idx] -= 1.2 * ra * Q[rr_idx, 0]
                            # Penalty for circles
                            if ConArray[des1, src2, 0] and ConArray[des2, src1, 0]:
                                E_p[r_idx, c_idx] -= 1 * ra * Q[rr_idx, 0]

                        if c_idx == 1:
                            # adj_matrix[src2, des2] = 1
                            # Penalty for redundant edges
                            if ConArray[src1, src2, 1] and ConArray[des2, des1, 1]:
                                E_p[r_idx, c_idx] -= 1.2 * ra * Q[rr_idx, 1]
                            # Penalty for circles
                            if ConArray[des1, src2, 1] and ConArray[des2, src1, 1]:
                                E_p[r_idx, c_idx] -= 1 * ra * Q[rr_idx, 1]

                        if c_idx == 3:
                            # adj_matrix[src2, des2] = 0
                            # Awards for near-far match
                            if ConArray[src1, src2, 0] and ConArray[des2, des1, 0]:
                                E_p[r_idx, c_idx] += 4 * ra * Q[rr_idx, 0]

                        if c_idx == 4:
                            # adj_matrix[src2, des2] = 1
                            # Awards for near-far match
                            if ConArray[src1, src2, 1] and ConArray[des2, des1, 1]:
                                E_p[r_idx, c_idx] += 4 * ra * Q[rr_idx, 1]
                                # adj_matrix[src2, des2] = tmp
    return x_s + E_p

