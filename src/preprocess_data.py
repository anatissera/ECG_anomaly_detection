from typing import Dict, List
import numpy as np
tqdm = __import__('tqdm').tqdm
import copy


def trim_to_full_batches(arrays: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """Trim arrays so that the sample count is a multiple of batch_size."""

    n_samples = arrays[0].shape[0]
    n_trimmed = (n_samples // batch_size) * batch_size
    trimmed = []
    for arr in arrays:
        assert arr.shape[0] == n_samples, "All inputs must have same sample count"
        trimmed.append(arr[:n_trimmed])
    return trimmed

def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Ensure shape (samples, time, leads)."""

    if x.ndim == 2:
        return x[..., np.newaxis]
    return x

def count_per_class(labels: np.ndarray, num_classes: int) -> List[int]:
    """Count samples per class."""

    return [int(np.sum(labels == i)) for i in range(num_classes)]

def preprocess_data_fmm(input_data: Dict[str, np.ndarray], dataset_params: Dict[str, List[int]], fs: int, batch_size: int, split_ecg: bool = False, **kwargs) -> Dict[str, np.ndarray]:
    """
    Preprocess FMM model data for training:
    - Converts signals to float32 and deep-copies them to avoid side effects.
    - Ensures a 3D shape (samples, time, leads).
    - Trims every array so the sample count is a multiple of batch_size.
    - Computes and prints the per-class sample count using dataset_params['classes'].

    Returns a dict with:
      'data', 'labels', 'sizes', 'coefficients', 'coefficients_ang', 'records'
    """

    data = copy.deepcopy(input_data['data'].astype(np.float32))
    labels = copy.deepcopy(input_data['labels'])
    sizes = input_data['sizes']
    coeffs = input_data['coefficients']
    coeffs_ang = input_data['coefficients_ang']
    records = input_data['records']

    data = ensure_3d(data)

    data, labels, sizes, coeffs, coeffs_ang, records = trim_to_full_batches(
        [data, labels, sizes, coeffs, coeffs_ang, records], batch_size
    )

    classes = dataset_params.get('classes', []) if isinstance(dataset_params, dict) else []
    counts = count_per_class(labels, num_classes=len(classes))
    print(f"Number of samples per class: {counts}")

    return {
        'data': data,
        'labels': labels,
        'sizes': sizes,
        'coefficients': coeffs,
        'coefficients_ang': coeffs_ang,
        'records': records
    }
