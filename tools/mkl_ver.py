import re

import torch


def compatible_mkl_ver():
    torch_config = torch.__config__.show()
    with_mkl_blas = 'BLAS_INFO=mkl' in torch_config
    if torch.backends.mkl.is_available() and with_mkl_blas:
        product_version = '2023.1.0'
        pattern = r'oneAPI Math Kernel Library Version [0-9]{4}\.[0-9]+'
        match = re.search(pattern, torch_config)
        if match:
            product_version = match.group(0).split(' ')[-1]

        return product_version

    return None
