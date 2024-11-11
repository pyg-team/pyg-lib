from collections import defaultdict

import boto3

ROOT_URL = 'https://data.pyg.org/whl/nightly'
html = '<!DOCTYPE html>\n<html>\n<body>\n{}\n</body>\n</html>'
href = '  <a href="{}">{}</a><br/>'
args = {
    'ContentType': 'text/html',
    'CacheControl': 'max-age=300',
    'ACL': 'public-read',
}

bucket = boto3.resource('s3').Bucket(name='data.pyg.org')

wheels_dict = defaultdict(list)
for obj in bucket.objects.filter(Prefix='whl/nightly'):
    if obj.key[-3:] != 'whl':
        continue
    torch_version, wheel = obj.key.split('/')[-2:]
    wheel = f'{torch_version}/{wheel}'
    wheels_dict[torch_version].append(wheel)

    if '1.12.0' in torch_version:
        wheels_dict[torch_version.replace('1.12.0', '1.12.1')].append(wheel)
    if '1.13.0' in torch_version:
        wheels_dict[torch_version.replace('1.13.0', '1.13.1')].append(wheel)
    if '2.0.0' in torch_version:
        wheels_dict[torch_version.replace('2.0.0', '2.0.1')].append(wheel)
    if '2.1.0' in torch_version:
        wheels_dict[torch_version.replace('2.1.0', '2.1.1')].append(wheel)
        wheels_dict[torch_version.replace('2.1.0', '2.1.2')].append(wheel)
    if '2.2.0' in torch_version:
        wheels_dict[torch_version.replace('2.2.0', '2.2.1')].append(wheel)
        wheels_dict[torch_version.replace('2.2.0', '2.2.2')].append(wheel)
    if '2.3.0' in torch_version:
        wheels_dict[torch_version.replace('2.3.0', '2.3.1')].append(wheel)
    if '2.4.0' in torch_version:
        wheels_dict[torch_version.replace('2.4.0', '2.4.1')].append(wheel)
    if '2.5.0' in torch_version:
        wheels_dict[torch_version.replace('2.5.0', '2.5.1')].append(wheel)

index_html = html.format('\n'.join([
    href.format(f'{version}.html'.replace('+', '%2B'), version)
    for version in wheels_dict
]))

with open('index.html', 'w') as f:
    f.write(index_html)
bucket.Object('whl/nightly/index.html').upload_file('index.html', args)

for torch_version, wheel_names in wheels_dict.items():
    torch_version_html = html.format('\n'.join([
        href.format(f'{ROOT_URL}/{wheel_name}'.replace('+', '%2B'),
                    wheel_name.split('/')[-1]) for wheel_name in wheel_names
    ]))

    with open(f'{torch_version}.html', 'w') as f:
        f.write(torch_version_html)
    bucket.Object(f'whl/nightly/{torch_version}.html').upload_file(
        f'{torch_version}.html', args)
