import torch

def adaptive_mps_inverse(matrix):
    if hasattr(matrix, 'is_mps') and matrix.is_mps:
        device = matrix.device
        matrix = matrix.to('cpu')
        inv = torch.linalg.inv(matrix)
        inv = inv.to(device)
    else:
        inv = torch.linalg.inv(matrix)
    return inv

def adaptive_mps_matmul(a, b):
    if hasattr(a, 'is_mps') and a.is_mps:
        device = a.device
        a_ = a.to('cpu')
        b_ = b.to('cpu')
        c_ = a_ @ b_
        c = c_.to(device)
    else:
        c = a @ b
    return c

def adaptive_mps_unique(val):
    if hasattr(val, 'is_mps') and val.is_mps:
        device = val.device
        val_ = val.to('cpu')
        unique = val_.unique()
        unique = unique.to(device)
    else:
        unique = val.unique()
    return unique

def adaptive_mps_argsort(vals):
    if hasattr(vals, 'is_mps') and vals.is_mps:
        device = vals.device
        vals_ = vals.to('cpu')
        sorts = vals_.argsort()
        return sorts.to(device)
    else:
        return vals.argsort()

def adaptive_mps_cumsum(vals, dim):
    if hasattr(vals, 'is_mps') and vals.is_mps:
        device = vals.device
        vals_ = vals.to('cpu')
        sum = vals_.cumsum(dim)
        return sum.to(device)
    else:
        return vals.cumsum(dim)