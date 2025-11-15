import math
import torch

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pi = math.pi
sqrt2 = math.sqrt(2.0)

T_MIN, T_MAX = 1e-1, 10.0
S_MIN, S_MAX = math.log(T_MIN), math.log(T_MAX)