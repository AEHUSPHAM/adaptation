import torch 

def momentum_matrix(n, m1, m2, s, opt):   
    mask_matrix = torch.zeros([n,n], dtype=torch.float32, requires_grad=False)
    if opt == "origin" or opt == "test":
        for i in range(n):
            mask_matrix[i,i] = 1
            for j in range(1,i+1):
                mask_matrix[i,i-j] = mask_matrix[i,i-j+1] * m1
    elif opt == "bidirectional":
        for i in range(n):
            mask_matrix[i,i] = 1
            for j in range(1,i+1):
                mask_matrix[i,i-j] = mask_matrix[i,i-j+1] * m1
            for j in range(i+1,n):
                mask_matrix[i,j] = mask_matrix[i,j-1] * m2
            
    elif opt== "threshold" or opt == "adam":
        for i in range(n):
            mask_matrix[i,i] = 1
            for j in range(1,min(10,i+1)):
                mask_matrix[i,i-j] = mask_matrix[i,i-j+1] * m1
    elif opt== "none":
        for i in range(n):
            mask_matrix[i,i] = 1
    elif opt== "nesterov":
        for i in range(n):
            m=(i%5)/((i%5)+3)
            mask_matrix[i,i] = 1
            for j in range(1,i+1):
                mask_matrix[i,i-j] = mask_matrix[i,i-j+1] * m
            for j in range(i+1,n):
                mask_matrix[i,j] = mask_matrix[i,j-1] * m
    else: return None

    if opt != "adam":
        mask_matrix = s*mask_matrix

    return mask_matrix
  