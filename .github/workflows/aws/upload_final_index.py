from collections import defaultdict

import boto3

bucket = boto3.resource('s3').Bucket(name='data.pyg.org')

wheels = [obj.key for obj in bucket.objects.all() if obj.key[-3:] == 'whl']

# wheels_dict = { torch_version: wheel, ...], ... }
wheels_dict = defaultdict(list)
for wheel in wheels:
    if 'wheels' in wheel:
        continue
    if 'nightly' in wheel:
        continue
    _, torch_version, wheel = wheel.split('/')
    wheel = (torch_version, wheel)
    wheels_dict[torch_version].append(wheel)

    if '1.7.0' in torch_version:
        wheels_dict[torch_version.replace('1.7.0', '1.7.1')].append(wheel)
    if '1.8.0' in torch_version:
        wheels_dict[torch_version.replace('1.8.0', '1.8.1')].append(wheel)
    if '1.9.0' in torch_version:
        wheels_dict[torch_version.replace('1.9.0', '1.9.1')].append(wheel)
    if '1.10.0' in torch_version:
        wheels_dict[torch_version.replace('1.10.0', '1.10.1')].append(wheel)
        wheels_dict[torch_version.replace('1.10.0', '1.10.2')].append(wheel)
    if '1.10.0+cu113' in torch_version:
        wheels_dict['torch-1.10.0+cu111'].append(wheel)
        wheels_dict['torch-1.10.1+cu111'].append(wheel)
        wheels_dict['torch-1.10.2+cu111'].append(wheel)
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
    if '2.7.0' in torch_version:
        wheels_dict[torch_version.replace('2.7.0', '2.7.1')].append(wheel)

html = '<!DOCTYPE html>\n<html>\n<body>\n{}\n</body>\n</html>'
href = '<a href="{}">{}</a><br/>'

args = {
    'ContentType': 'text/html',
    'CacheControl': 'max-age=300',
    'ACL': 'public-read',
}

index_html = html.format('\n'.join([href.format('whl/index.html', 'whl')]))

with open('index.html', 'w') as f:
    f.write(index_html)

bucket.Object('index.html').upload_file('index.html', ExtraArgs=args)

index_html = html.format('\n'.join([
    href.format(f'{torch_version}.html'.replace('+', '%2B'), torch_version)
    for torch_version in wheels_dict
]))

with open('index.html', 'w') as f:
    f.write(index_html)

bucket.Object('whl/index.html').upload_file('index.html', ExtraArgs=args)

root = 'https://data.pyg.org'

for torch_version, wheels in wheels_dict.items():
    torch_version_html = html.format('\n'.join([
        href.format(
            f'{root}/whl/{orig_torch_version}/{wheel}'.replace('+', '%2B'),
            wheel) for orig_torch_version, wheel in wheels
    ]))

    with open(f'{torch_version}.html', 'w') as f:
        f.write(torch_version_html)

    bucket.Object(f'whl/{torch_version}.html').upload_file(
        f'{torch_version}.html', ExtraArgs=args)
