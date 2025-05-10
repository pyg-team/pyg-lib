# # Creates a PEP 503 compliant index for the given release type.
# # https://peps.python.org/pep-0503/
# # just like what PyTorch does at download.pytorch.org.
# #
# # In PyTorch, it is like:
# # https://download.pytorch.org/whl/cu121
# # https://download.pytorch.org/whl/cu128
# # and PyTorch users install it by:
# # >>> pip install torch --index-url https://download.pytorch.org/whl/cu128
# # but in our pyg-lib case, it is like:
# # https://data.pyg.org/whl/torch-2.4.0+cu121/
# # https://data.pyg.org/whl/torch-2.4.0+cu128/
# # and our users install it by:
# # >>> pip install pyg-lib --index-url https://data.pyg.org/whl/torch-2.4.0+cu128
# # and pip will look for wheels in:
# # https://data.pyg.org/whl/torch-2.4.0+cu128/pyg-lib/index.html


# import argparse
# import boto3

# from packaging import version

# MIN_TORCH_VERSION = '1.13.0'
# INDEX_URL_STABLE = 'https://data.pyg.org/whl'
# INDEX_URL_NIGHTLY = 'https://data.pyg.org/whl/nightly'
# PYPI_TORCH_URL = 'https://pypi.org/pypi/torch/json'


# def get_list_of_torch_versions() -> list[str]:
#     import requests
#     response = requests.get(PYPI_TORCH_URL)
#     response.raise_for_status()
#     data = response.json()
#     torch_versions = sorted(data['releases'].keys(),
#                             key=lambda x: version.parse(x), reverse=True)
#     torch_versions = [
#         torch_version for torch_version in torch_versions
#         if version.parse(torch_version) >= version.parse(MIN_TORCH_VERSION)
#     ]
#     print(torch_versions)
#     return torch_versions


# def generate_root_index(torch_versions: list[str]) -> None:
#     r"""Creates a PEP 503 compliant index for the given release type.
#     """

#     # Need to index all combinations of torch and cuda:
#     content = f"""
# <!DOCTYPE html>
# <html>
#   <body>
#     <a href="/torch-2.4.0+cu121/">torch-2.4.0+cu121</a>
#     <a href="/torch-2.4.0+cu128/">torch-2.4.0+cu128</a>
#   </body>
# </html>
#     """
#     print(content)

#     bucket = boto3.resource('s3').Bucket(name='data.pyg.org')
#     for obj in bucket.objects.filter(Prefix='whl'):
#         if obj.key[-3:] != 'whl':
#             continue
#         torch_version, wheel = obj.key.split('/')[-2:]
#         wheel = f'{torch_version}/{wheel}'
#         wheels_dict[torch_version].append(wheel)

#     html = ''
#     for torch_version in torch_versions:
#         html += f'<a href="{torch_version}/">{torch_version}</a>\n'

#     print(html)


# # TODO
# # def generate_torch_index(torch_version: str) -> None:
# #     bucket = boto3.resource('s3').Bucket(name='data.pyg.org')
# #     for obj in bucket.objects.filter(Prefix='whl/nightly'):
# #         if obj.key[-3:] != 'whl':
# #             continue
# #         torch_version, wheel = obj.key.split('/')[-2:]
# #         wheel = f'{torch_version}/{wheel}'
# #         wheels_dict[torch_version].append(wheel)


# def main():
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--release-type', type=str, required=True, choices=['stable', 'nightly'])
#     parser.add_argument('--index-type', type=str, required=True,
#                         choices=['root', 'torch'])
#     parser.parse_args()
#     torch_versions = get_list_of_torch_versions()
#     generate_root_index(torch_versions)


# if __name__ == '__main__':
#     main()
