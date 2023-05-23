import re

def remove_p_tags(x):
    pattern = re.compile(r'<p.*?>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</p>', '')

    return x

def remove_href_tags(x):
    pattern = re.compile(r'<a .*?>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</a>', '')

    return x

def remove_span_tags(x):
    pattern = re.compile(r'<span .*?>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</span>', ':</span>')
    x = x.replace('</span>', '')

    return x

def remove_li_tags(x):
    pattern = re.compile(r'<li>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</li>', '')

    return x

def remove_font_tags(x):
    pattern = re.compile(r'<code>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</code>', '')

    attern = re.compile(r'<strong>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</strong>', '')

    return x

def remove_table_tags(x):
    pattern = re.compile(r'<table>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</table>', '')

    pattern = re.compile(r'<thead>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</thead>', '')

    pattern = re.compile(r'<tbody>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</tbody>', '')

    pattern = re.compile(r'<th.*?>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</th>', '')

    pattern = re.compile(r'<td.*?>', re.S)
    x = pattern.sub('', x)
    x = x.replace('</td>', '')

    return x

def replace_tags(x):
    pattern = re.compile(r'<[^>]+>', re.S)
    x = pattern.sub('\n', x)

    return x