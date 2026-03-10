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
        logging.info(f"Found {len(self.files)} NetCDF files in {path}")

        # ---- NEW: infer classes from folder structure ----
        class_names = sorted({os.path.basename(os.path.dirname(f)) for f in self.files})
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

        self.samples = []  # (file, label) or (file, t, label)

        # Determine variables
        first_ds = xr.open_dataset(self.files[0], engine="h5netcdf")
        all_vars = list(first_ds.data_vars)

        if variables:
            self.variables = variables
        elif self.config_vars:
            self.variables = [var for var in self.config_vars if var in all_vars]
        else:
            self.variables = [var for var in all_vars if first_ds[var].dims == ("time", "lat", "lon")]
        first_ds.close()

        if self.config_vars:
            missing = [var for var in self.config_vars if var not in self.variables]
            if missing:
                logging.warning(f"Some config variables not found in dataset: {missing}")

        # Index samples with labels
        for file in self.files:
            class_name = os.path.basename(os.path.dirname(file))
            class_idx = self.class_to_idx[class_name]

            if self.use_full_time_series:
                self.samples.append((file, class_idx))
            else:
                ds = xr.open_dataset(file, engine="h5netcdf")
                n_times = ds.dims.get("time", 1)
                for t in range(n_times):
                    self.samples.append((file, t, class_idx))
                ds.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_full_time_series:
            file_path, label = self.samples[idx]
            ds = xr.open_dataset(file_path, engine="h5netcdf")

            channels = []
            for var in self.variables:
                if var not in ds:
                    logging.warning(f"Variable {var} not found in {file_path}, skipping.")
                    continue
                data = ds[var].values  # (T, H, W)
                channels.append(data)

            tensor = torch.tensor(np.stack(channels, axis=0))  # (C, T, H, W)
            ds.close()
            return tensor, label

        else:
            file_path, time_idx, label = self.samples[idx]
            ds = xr.open_dataset(file_path, engine="h5netcdf")
            #print(ds)
            channels = []
            for var in self.variables:
                if var not in ds:
                    logging.warning(f"Variable {var} not found in {file_path}, skipping.")
                    continue
                #check if time dimension exists
                if "time" not in ds[var].dims:
                    data = ds[var].values
                else:
                    data = ds[var].isel(time=time_idx).values
                #print(data.shape)
                channels.append(data)
            #print(channels)
            tensor = torch.tensor(np.stack(channels, axis=0))  # (C, H, W)
            ds.close()

            return tensor, label

    # ---- helper for VISSL ----
    def get_labels(self):
        return [lbl for *_, lbl in self.samples]
