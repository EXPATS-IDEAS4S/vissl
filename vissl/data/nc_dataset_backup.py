import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import glob
import os
import logging

class NetCDFDataset(Dataset):
    def __init__(self, cfg, path, split, dataset_name, data_source, variables=None):
        """
        Args:
            cfg: VISSL config
            path: directory of .nc files or list of file paths
            split: 'train', 'val', etc.
            dataset_name: name in config
            data_source: always 'netcdf'
            variables: list of variable names to use. If None, include all (time, lat, lon) variables
        """
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source

        # Load file paths
        self.files = sorted(glob.glob(os.path.join(path, "*/*.nc"))) if isinstance(path, str) else path
        self.samples = []  # list of (file_path, time_index) tuples
        logging.info(f"Found {len(self.files)} NetCDF files in {path}")
        
        # Determine valid variables (once)
        first_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        #print(first_ds)
        if variables is None:
            self.variables = [
                var for var in first_ds.data_vars
                if first_ds[var].dims == ("time", "lat", "lon")
            ]
        else:
            self.variables = variables
        first_ds.close()

        # Build full index: one sample per (file, time)
        for file in self.files:
            ds = xr.open_dataset(file, engine="h5netcdf")
            n_times = ds.dims.get("time", 1)
            for t in range(n_times):
                self.samples.append((file, t))
            ds.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, time_idx = self.samples[idx]
        ds = xr.open_dataset(file_path, engine="h5netcdf")

        channels = []
        for var in self.variables:
            data = ds[var].isel(time=time_idx).values#.astype(np.float32)  # or keep dtype?
            channels.append(data)

        tensor = torch.tensor(np.stack(channels, axis=0))  # [C, H, W]
        ds.close()
        return tensor, True  # second output = is_valid for VISSL
