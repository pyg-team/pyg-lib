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
for obj in bucket.objects.filter(Prefix='whl/nightly/torch'):
    torch_version, wheel_name = obj.key.split('/')[-2:]
    wheels_dict[torch_version].append(wheel_name)

index_html = html.format('\n'.join([
    href.format(f'{torch_version}.html', version) for version in wheels_dict
]))

with open('index.html', 'w') as f:
    f.write(index_html)
bucket.Object('whl/nightly/index.html').upload_file('index.html', args)

for torch_version, wheel_names in wheels_dict.items():
    torch_version_html = html.format('\n'.join([
        href.format(
            f'{ROOT_URL}/{torch_version}/{wheel_name}'.replace('+', '%2B'),
            wheel_name) for wheel_name in wheel_names
    ]))

    with open(f'{torch_version}.html', 'w') as f:
        f.write(torch_version_html)
    bucket.Object(f'whl/nightly/{torch_version}.html').upload_file(
        f'{torch_version}.html', args)
