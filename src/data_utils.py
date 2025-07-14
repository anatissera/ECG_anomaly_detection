
from typing import Dict, Tuple, List
import numpy as np

DEFAULT_WAVES = ["P", "Q", "R", "S", "T"]


# Indexes

def get_fmm_num_parameters(n_leads: int, n_waves: int = 5) -> Tuple[int, int]:
    """Devuelve el total de parámetros y el conteo por onda (A, alpha, beta, omega)."""
    
    per_wave = 2 * n_leads + 2
    total = per_wave * n_waves + n_leads
    return total, per_wave

# offset_map: coeff -> (base_offset, size_fn)
offset_map = {
    'A':     (0,                 lambda L: L),
    'alpha': (lambda L: L,        lambda L: 1),
    'beta':  (lambda L: L + 1,    lambda L: L),
    'omega': (lambda L: 2*L + 1,  lambda L: 1)
}

def get_coeff_indexes(coeff: str, wave_idx: int, n_leads: int, n_waves: int = 5) -> Tuple[int, int]:
    """Calcula índices inicio y fin para un coeficiente dado."""
    
    _, per_wave = get_fmm_num_parameters(n_leads, n_waves)
    if coeff not in offset_map:
        raise ValueError(f"Unknown coeff '{coeff}'")
    base_fn, size_fn = offset_map[coeff]
    base = base_fn(n_leads) if callable(base_fn) else base_fn
    size = size_fn(n_leads) if callable(size_fn) else size_fn
    start = wave_idx * per_wave + base
    return start, start + size


def get_M_indexes(n_leads: int, n_waves: int = 5) -> Tuple[int, int]:
    """Obtiene índices de M al final del array."""
    
    total, _ = get_fmm_num_parameters(n_leads, n_waves)
    return total - n_leads, total


def get_circular_mask(n_leads: int, n_waves: int = 5) -> np.ndarray:
    """Máscara booleana para los índices de alpha y beta."""
    
    total, _ = get_fmm_num_parameters(n_leads, n_waves)
    mask = np.zeros(total, dtype=bool)
    for i in range(n_waves):
        for coeff in ('alpha', 'beta'):
            s, e = get_coeff_indexes(coeff, i, n_leads, n_waves)
            mask[s:e] = True
    return mask


# Conversion

def convert_fmm_dict_to_array(fmm: Dict[str, Dict[str, np.ndarray]],
                              waves: List[str] = DEFAULT_WAVES) -> np.ndarray:
    """Convierte dict FMM anidado a un array plano."""
    
    n_waves = len(waves)
    n_leads = fmm[waves[0]]['A'].shape[0]
    total, _ = get_fmm_num_parameters(n_leads, n_waves)
    arr = np.zeros(total)
    for idx, w in enumerate(waves):
        for coeff in ('A', 'alpha', 'beta', 'omega'):
            s, e = get_coeff_indexes(coeff, idx, n_leads, n_waves)
            vals = np.squeeze(fmm[w][coeff])
            arr[s:e] = vals.flat[0] if coeff in ('alpha', 'omega') else vals
    m_vals = np.squeeze(fmm[waves[-1]]['M'])
    ms, me = get_M_indexes(n_leads, n_waves)
    arr[ms:me] = m_vals
    return arr


def array_to_fmm_dict(coeffs: np.ndarray, num_leads: int, num_waves: int = 5) -> Dict[str, Dict[str, np.ndarray]]:
    """Convierte un array plano de coeficientes en un dict FMM anidado."""
    
    total, _ = get_fmm_num_parameters(num_leads, num_waves)
    m_start, m_end = get_M_indexes(num_leads, num_waves)

    fmm = {}
    for i, wave in enumerate(DEFAULT_WAVES[:num_waves]):
        wave_data: Dict[str, np.ndarray] = {}
        for field in ('A', 'alpha', 'beta', 'omega'):
            start, end = get_coeff_indexes(field, i, num_leads, num_waves)
            vals = coeffs[start:end]
            if field in ('alpha', 'omega'):
                # replicar valor escalar per-lead
                wave_data[field] = np.full((num_leads,), vals[0])
            else:
                wave_data[field] = vals.copy()
        wave_data['M'] = coeffs[m_start:m_end].copy()

        fmm[wave] = wave_data

    return fmm


def extract_lead_coeffs(arr: np.ndarray, lead: int, n_leads: int, n_waves: int = 5) -> np.ndarray:
    """Extrae parámetros FMM de un solo electrodo."""
    
    single_total, _ = get_fmm_num_parameters(1, n_waves)
    out = np.zeros(single_total)
    for idx, coeff in enumerate(('A', 'alpha', 'beta', 'omega')):
        s_all, _ = get_coeff_indexes(coeff, idx, n_leads, n_waves)
        s_one, _ = get_coeff_indexes(coeff, idx, 1, n_waves)
        src = s_all + lead if coeff in ('A', 'beta') else s_all
        out[s_one] = arr[src]
    ms, _ = get_M_indexes(n_leads, n_waves)
    m_one_start, _ = get_M_indexes(1, n_waves)
    out[m_one_start] = arr[ms + lead]
    return out


def angle_to_cos_sin(X: np.ndarray, ang_mask: np.ndarray, zero_one: bool = False) -> np.ndarray:
    """Agrega coseno y seno de ángulos al array de coeficientes."""
    
    vals = X[:, ang_mask]
    cos = np.cos(vals)
    sin = np.sin(vals)
    if zero_one:
        cos = (cos + 1) / 2
        sin = (sin + 1) / 2
    cs = np.stack([cos, sin], axis=-1).reshape(X.shape[0], -1)
    return np.concatenate([X, cs], axis=1)


# Sorting

def sort_fmm_coeffs_array(fmm_array: np.ndarray, n_leads: int, n_waves: int = 5) -> np.ndarray:
    """Ordena coeficientes FMM por ángulo alpha creciente."""
    
    # lineariza ángulos alpha
    alpha_idxs = [get_coeff_indexes('alpha', i, n_leads, n_waves)[0]
                  for i in range(n_waves)]
    def lin(x: float) -> float:
        if 0 <= x < np.pi: return x + np.pi
        if np.pi <= x < 2*np.pi: return x - np.pi
        raise ValueError(f"Angle out of range: {x}")
    vec_lin = np.vectorize(lin)
    alpha_mat = vec_lin(fmm_array[:, alpha_idxs])
    orders = np.argsort(alpha_mat, axis=1)

    blocks = np.concatenate([
        np.arange(*get_coeff_indexes(coeff, i, n_leads, n_waves))
        for coeff in ('A','alpha','beta','omega')
        for i in range(n_waves)
    ]).reshape(n_waves, -1)

    sorted_arr = np.zeros_like(fmm_array)
    for j in range(fmm_array.shape[0]):
        perm = orders[j]
        src = blocks[perm].flatten()
        dst = blocks.flatten()
        sorted_arr[j, dst] = fmm_array[j, src]
    ms, me = get_M_indexes(n_leads, n_waves)
    sorted_arr[:, ms:me] = fmm_array[:, ms:me]
    return sorted_arr