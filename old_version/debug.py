from data import PrecompDataset
from torch.utils.data import DataLoader, Dataset


d = PrecompDataset("/workspace/precomp_data")
t_d = DataLoader(d, batch_size=32, num_workers=16, pin_memory=True)

dd = next(iter(t_d))
print(dd['extent'].shape)
print(dd['extent'][0:4])