import re
from netaddr import IPAddress, AddrFormatError

def is_ip_address_or_domain(input_string):
    domain_pattern = r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

    if str(input_string).isspace() or input_string is None:
        return "[UNK]"

    # Remove square brackets if present (common in URLs with IPv6 addresses)
    input_string = input_string.strip('[]')
    parts = input_string.split(':')
    if len(parts) > 2:
        ip_string = input_string
    else:
        ip_string = parts[0]

    try:
        ip = IPAddress(ip_string)
        if ip.version == 4:
            return "[IP]"
        elif ip.version == 6:
            return "[IPv6]"
    except AddrFormatError:
        pass

    if re.match(domain_pattern, input_string):
        return "[DOMAIN]"

    return "[DOMAIN]"

from urllib.parse import urlparse

def split_url(url):
    edited_url = False
    original_url = url
    if 'http' not in url:
        url = 'http://' + url
    if '/../' in url:
        url = url.replace('/../', '/')
        edited_url = True
    try:
        url_obj = urlparse(url)
        domain_or_ip = url_obj.netloc

        if edited_url:
            path = original_url.partition(domain_or_ip)[2]
        else:
            path = url_obj.path + ('?' + url_obj.query if url_obj.query else '')

        ip_or_domain_token = is_ip_address_or_domain(domain_or_ip)

        if path:
            return f'{ip_or_domain_token} {domain_or_ip} [PATH] {path}'
        else:
            return f'{ip_or_domain_token} {domain_or_ip}'
    except:
        #print(original_url)
        return f'[DOMAIN] {original_url}'
