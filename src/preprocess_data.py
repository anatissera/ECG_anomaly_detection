from typing import Dict, List
import numpy as np
tqdm = __import__('tqdm').tqdm
import copy


def trim_to_full_batches(arrays: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
    """Recorta arrays para que muestras sean múltiplo de batch_size."""
    
    n_samples = arrays[0].shape[0]
    n_trimmed = (n_samples // batch_size) * batch_size
    trimmed = []
    for arr in arrays:
        assert arr.shape[0] == n_samples, "All inputs must have same sample count"
        trimmed.append(arr[:n_trimmed])
    return trimmed

def ensure_3d(x: np.ndarray) -> np.ndarray:
    """Asegura forma (samples, time, leads)."""
    
    if x.ndim == 2:
        return x[..., np.newaxis]
    return x

def count_per_class(labels: np.ndarray, num_classes: int) -> List[int]:
    """Cuenta muestras por clase."""
    
    return [int(np.sum(labels == i)) for i in range(num_classes)]

def preprocess_data_fmm(input_data: Dict[str, np.ndarray], dataset_params: Dict[str, List[int]], fs: int, batch_size: int, split_ecg: bool = False, **kwargs) -> Dict[str, np.ndarray]:
    """
    Preprocesa datos del modelo FMM para entrenamiento:
    - Convierte señales a float32 y copia profunda para evitar efectos secundarios.
    - Asegura forma 3D (muestras, tiempo, derivaciones).
    - Recorta todos los arrays para que su número de muestras sea múltiplo de batch_size.
    - Calcula e imprime el conteo de muestras por clase usando dataset_params['classes'].

    Retorna un dict con:
      'data', 'labels', 'sizes', 'coefficients', 'coefficients_ang'
    """

    data = copy.deepcopy(input_data['data'].astype(np.float32))
    labels = copy.deepcopy(input_data['labels'])
    sizes = input_data['sizes']
    coeffs = input_data['coefficients']
    coeffs_ang = input_data['coefficients_ang']

    data = ensure_3d(data)

    data, labels, sizes, coeffs, coeffs_ang = trim_to_full_batches(
        [data, labels, sizes, coeffs, coeffs_ang], batch_size
    )

    classes = dataset_params.get('classes', []) if isinstance(dataset_params, dict) else []
    counts = count_per_class(labels, num_classes=len(classes))
    print(f"Number of samples per class: {counts}")

    return {
        'data': data,
        'labels': labels,
        'sizes': sizes,
        'coefficients': coeffs,
        'coefficients_ang': coeffs_ang
    }