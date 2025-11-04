import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloaders.flash_dataset import FlashMRIDataset

ds = FlashMRIDataset(["sub-04570","sub-04800"], "data/processed")
x, sid = ds[0]
print(x.shape, sid)
print("Dataset size:", len(ds))