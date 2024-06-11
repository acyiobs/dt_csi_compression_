import torch
from einops import rearrange, reduce, repeat
def cal_nmse(A, B):
    """
    Compute the Normalized Mean Squared Error (NMSE) between matrices A and B using PyTorch.

    Args:
    - A (torch.Tensor): The original matrix.
    - B (torch.Tensor): The approximated matrix.

    Returns:
    - float: The NMSE value between matrices A and B.
    """
    
    # Calculate the Frobenius norm difference between A and B

    A = rearrange(A, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()
    B = rearrange(B, 'b RealImag Nt Nc -> b Nt Nc RealImag').contiguous()

    A = torch.view_as_complex(A)
    B = torch.view_as_complex(B)

    error_norm = torch.norm(A - B, p='fro', dim=(-1, -2))
    
    # Calculate the Frobenius norm of A
    A_norm = torch.norm(A, p='fro', dim=(-1, -2))
    
    # Return NMSE
    return (error_norm**2) / (A_norm**2)

if __name__ == "__main__":
    A = torch.ones((64,2,32,32)).float()
    B = torch.ones((64,2,32,32)).float()
    nmse = cal_nmse(A, B)
    print(nmse)