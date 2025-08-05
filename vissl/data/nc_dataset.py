import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import glob
import os
import logging

class NetCDFDataset(Dataset):
    def __init__(self, cfg, path, split, dataset_name, data_source, variables=None):
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source

        self.use_full_time_series = getattr(cfg.DATA, "USE_FULL_TIME_SERIES", False)
        self.config_vars = getattr(cfg.DATA, "VARIABLES", None)

        self.files = sorted(glob.glob(os.path.join(path, "*/*.nc"))) if isinstance(path, str) else path
        self.samples = []  # If use_full_time_series: just files, else: (file, time_index)
        logging.info(f"Found {len(self.files)} NetCDF files in {path}")

        # Determine variables to load
        first_ds = xr.open_dataset(self.files[0])
        all_vars = list(first_ds.data_vars)

        if variables:
            self.variables = variables
        elif self.config_vars:
            self.variables = [var for var in self.config_vars if var in all_vars]
        else:
            # Default to all 3D spatiotemporal variables
            self.variables = [
                var for var in all_vars
                if first_ds[var].dims == ("time", "lat", "lon")
            ]
        first_ds.close()

        if self.config_vars:
            missing = [var for var in self.config_vars if var not in self.variables]
            if missing:
                logging.warning(f"Some config variables not found in dataset: {missing}")

        # Index samples
        for file in self.files:
            if self.use_full_time_series:
                self.samples.append(file)
            else:
                ds = xr.open_dataset(file)
                n_times = ds.dims.get("time", 1)
                for t in range(n_times):
                    self.samples.append((file, t))
                ds.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_full_time_series:
            file_path = self.samples[idx]
            ds = xr.open_dataset(file_path)

            channels = []
            for var in self.variables:
                if var not in ds:
                    logging.warning(f"Variable {var} not found in {file_path}, skipping.")
                    continue
                data = ds[var].values  # (T, H, W)
                channels.append(data)

            tensor = torch.tensor(np.stack(channels, axis=0))  # (C, T, H, W)
            ds.close()
            return tensor, True

        else:
            file_path, time_idx = self.samples[idx]
            ds = xr.open_dataset(file_path)

            channels = []
            for var in self.variables:
                if var not in ds:
                    logging.warning(f"Variable {var} not found in {file_path}, skipping.")
                    continue
                data = ds[var].isel(time=time_idx).values
                channels.append(data)

            tensor = torch.tensor(np.stack(channels, axis=0))  # (C, H, W)
            ds.close()
            return tensor, True
