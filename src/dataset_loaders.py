import os
import pickle
import zipfile
from typing import Dict
from sklearn.model_selection import train_test_split
import gdown
import numpy as np
tqdm = __import__('tqdm').tqdm

from src.data_utils import (get_fmm_num_parameters, get_coeff_indexes, get_circular_mask, convert_fmm_dict_to_array, extract_lead_coeffs, angle_to_cos_sin, sort_fmm_coeffs_array)

PTB_URL_ID = '1nYRvbVYJJXPbCwKEqOIeCLnMXdIDAka7'
SHAOXING_URL_ID = '1FjvmVb8-PnpDdoBwv-Eqb89tYFOCx9KW'


def _load_fmm_split(base: str, split: str, sequence_length: int, n_leads: int, n_waves: int, lead: int, delete_high_A: bool, url_id: str) -> Dict:
    """
    Carga interna de un split de FMM.

    Args:
        base: Ruta base del dataset.
        split: Nombre del subdirectorio ("train", "test" o "all").
        sequence_length: Longitud fija de la secuencia de datos.
        n_leads: Cantidad de derivaciones a usar.
        n_waves: Número de ondas en el modelo FMM.
        lead: Índice de derivación a extraer si se reduce de multi-lead.
        delete_high_A: Si True, filtra muestras con amplitudes altas.
        url_id: ID de Google Drive para descarga.
    Returns:
        Diccionario con claves: 'data', 'labels', 'coefficients', 'sizes'.
    """
    
    folder = os.path.join(base, split)
    files = [f for f in os.listdir(folder)
             if not f.startswith(('params', 'elapsed_times'))]

    sample0 = pickle.load(open(os.path.join(folder, files[0]), 'rb'))
    keys = list(sample0.keys())
    out = {k: -np.ones(len(files),
                       dtype=(int if k == 'label' else float))
           for k in keys}
    out['data'] = np.zeros((len(files), sequence_length, n_leads))
    total_params, _ = get_fmm_num_parameters(n_leads, n_waves)
    out['coefficients'] = np.zeros((len(files), total_params))

    # por cada sample
    for i, fname in tqdm(enumerate(files), total=len(files),
                         desc=f"Loading '{split}'"):
        sample = pickle.load(open(os.path.join(folder, fname), 'rb'))
        raw_full = sample['data']                
        orig_leads = raw_full.shape[1]        

        seq = raw_full[:, [0]] if n_leads == 1 else raw_full
        L = sample['len']
  
        if L > sequence_length:
            out['data'][i] = seq[:sequence_length]
        else:
            out['data'][i, :L-1] = seq[:L-1]
        out['label'][i] = sample['label']

        arr = convert_fmm_dict_to_array(sample['coefficients'])
        
        total_len = arr.shape[0]
        # per_wave = 2*L +2, total_len = per_wave*n_waves + L => L = (total_len - 2*n_waves)/(2*n_waves+1)
        L = int((total_len - 2*n_waves) / (2*n_waves + 1))
        if n_leads != L:
            arr = extract_lead_coeffs(arr, lead, L, n_waves)
        out['coefficients'][i] = arr

    # opcional: filtrar por A alta
    if delete_high_A:
        a_idxs = [get_coeff_indexes('A', w, n_leads)[0]
                  for w in range(n_waves)]
        mask = np.all(out['coefficients'][:, a_idxs] <= 5, axis=1)
        for k in list(out):
            out[k] = out[k][mask]

    out['labels'] = out.pop('label')
    out['sizes'] = out.pop('len')
    return out


def get_ptb_xl_fmm_dataset(datapath: str = './data', frequency: int = 100, lead: int = 0, delete_high_A: bool = True, num_leads: int = 12, num_waves: int = 5, sequence_length: int = 100, **kwargs) -> Dict:
    """
    Carga y preprocesa el dataset PTB-XL con modelos FMM.

    Args:
        datapath: Carpeta raíz de datos.
        frequency: Frecuencia de muestreo (no usada internamente).
        lead: Índice de derivación a usar si num_leads=1.
        delete_high_A: Filtrar amplitudes altas si True.
        num_leads: Número de derivaciones a conservar.
        num_waves: Cantidad de ondas FMM.
        sequence_length: Longitud de secuencia uniforme.
    Returns:
        Diccionario con llaves 'train', 'test' y 'params'.
    """
    
    base = os.path.join(datapath, 'ptb_xl_fmm')
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, 'ptb_xl_fmm.zip')
    if not os.listdir(base):
        gdown.download(f'https://drive.google.com/uc?id={PTB_URL_ID}', zip_path,
                       quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(base)

    train = _load_fmm_split(base, 'train', sequence_length,
                             num_leads, num_waves, lead,
                             delete_high_A, PTB_URL_ID)
    test = _load_fmm_split(base, 'test', sequence_length,
                            num_leads, num_waves, lead,
                            delete_high_A, PTB_URL_ID)

    train['sorted'] = sort_fmm_coeffs_array(train['coefficients'],
                                            num_leads, num_waves)
    test['sorted'] = sort_fmm_coeffs_array(test['coefficients'],
                                           num_leads, num_waves)
    mask = get_circular_mask(num_leads, num_waves)
    train['ang'] = angle_to_cos_sin(train['sorted'], mask,
                                     zero_one=True)
    test['ang'] = angle_to_cos_sin(test['sorted'], mask,
                                    zero_one=True)

    params = pickle.load(open(os.path.join(base, 'train', 'params'), 'rb'))
    return {'train': train, 'test': test, 'params': params}


def get_shaoxing_fmm_dataset(datapath: str = './data', frequency: int = 500, lead: int = 0, test_size: float = 0.2, split_seed: int = None, delete_high_A: bool = False, num_leads: int = 12, num_waves: int = 5, sequence_length: int = 100, **kwargs) -> Dict:
    """
    Carga y particiona el dataset Chapman-Shaoxing con FMM.

    Args:
        datapath: Carpeta raíz de datos.
        frequency: Frecuencia de muestreo original.
        lead: Derivación a extraer si num_leads=1.
        test_size: Fracción de prueba para split.
        split_seed: Semilla aleatoria para reproducibilidad.
        delete_high_A: Filtrar coeficientes con A altos si True.
        num_leads: Número de derivaciones conservadas.
        num_waves: Cantidad de ondas FMM.
        sequence_length: Longitud de secuencias.
    Returns:
        Diccionario con 'train', 'test' y 'params'.
    """
    
    base = os.path.join(datapath, 'ChapmanShaoxing_fmm')
    os.makedirs(base, exist_ok=True)
    zip_path = os.path.join(base, 'ChapmanShaoxing_fmm.zip')
    if not os.listdir(base):
        gdown.download(f'https://drive.google.com/uc?id={SHAOXING_URL_ID}', zip_path,
                       quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(base)

    ds_all = _load_fmm_split(base, 'all', sequence_length,
                              num_leads, num_waves, lead,
                              delete_high_A, SHAOXING_URL_ID)
    X, y = ds_all['data'], ds_all['labels']
    C, S = ds_all['coefficients'], ds_all['sizes']
    X_tr, X_te, y_tr, y_te, C_tr, C_te, S_tr, S_te = train_test_split(
        X, y, C, S, test_size=test_size, random_state=split_seed
    )
    # coefficients post proceso
    C_tr_s = sort_fmm_coeffs_array(C_tr, num_leads, num_waves)
    C_te_s = sort_fmm_coeffs_array(C_te, num_leads, num_waves)
    mask = get_circular_mask(num_leads, num_waves)
    C_tr_ang = angle_to_cos_sin(C_tr_s, mask, zero_one=True)
    C_te_ang = angle_to_cos_sin(C_te_s, mask, zero_one=True)

    params = pickle.load(open(os.path.join(base, 'all', 'params'), 'rb'))
    return {
        'train': {'data': X_tr, 'labels': y_tr,
                  'coefficients': C_tr_s, 'coefficients_ang': C_tr_ang,
                  'sizes': S_tr},
        'test':  {'data': X_te, 'labels': y_te,
                  'coefficients': C_te_s, 'coefficients_ang': C_te_ang,
                  'sizes': S_te},
        'params': params
    }
