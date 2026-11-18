"""Fetch audited subsets of official GOOSE ZIPs using HTTP byte ranges.

No archive is extracted wholesale. Member CRCs and local SHA256s are recorded;
selection is explicit in a plan so adjacent frames cannot silently cross splits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import threading
import time
import zipfile
import zlib

import requests

HTTP = threading.local()

URLS = {
    'goose_train': 'https://goose-dataset.de/storage/goose_3d_train.zip',
    'goose_val': 'https://goose-dataset.de/storage/goose_3d_val.zip',
    'gooseex_train': 'https://goose-dataset.de/storage/gooseEx_3d_train.zip',
    'gooseex_val': 'https://goose-dataset.de/storage/gooseEx_3d_val.zip',
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url, self.position, self.cache = url, 0, []
        _, self.size = self.fetch(0, 1)

    def fetch(self, offset, length):
        for attempt in range(4):
            try:
                if not hasattr(HTTP, 'session'):
                    HTTP.session = requests.Session()
                with HTTP.session.get(self.url, headers={'Range': f'bytes={offset}-{offset+length-1}',
                                                    'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(20, 120)) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if (response.status_code != 206 or not match or
                            int(match[1]) != offset or int(match[2]) != offset+length-1):
                        raise ValueError(f'Exact byte range unavailable: {response.status_code} {response.headers.get("Content-Range")}')
                    content = response.raw.read(length+1)
                    if len(content) != length:
                        raise ValueError('Truncated or oversized range')
                    return content, int(match[3])
            except (requests.RequestException, ValueError, OSError):
                if attempt == 3:
                    raise
                time.sleep(1+attempt)

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        if self.position < 0:
            raise ValueError('Negative ZIP seek')
        return self.position

    def tell(self):
        return self.position

    def read(self, length=-1):
        if length < 0:
            length = self.size-self.position
        length = min(length, self.size-self.position)
        if length <= 0:
            return b''
        start = self.position
        self.position += length
        for offset, content in self.cache:
            if offset <= start and start+length <= offset+len(content):
                return content[start-offset:start-offset+length]
        content, size = self.fetch(start, length)
        if size != self.size:
            raise ValueError('Remote archive size changed')
        self.cache.append((start, content))
        return content


def index_archive(name, root):
    path = Path(root)/'indices'/f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    remote = RemoteZip(URLS[name])
    with zipfile.ZipFile(remote) as archive:
        entries = [dict(name=x.filename, size=x.file_size, compressed=x.compress_size,
                        offset=x.header_offset, crc=x.CRC, compression=x.compress_type)
                   for x in archive.infolist() if not x.is_dir()]
    result = dict(archive=name, url=URLS[name], size=remote.size, entries=entries)
    save_json(path, result)
    return result


def read_member(index, entry):
    remote = RemoteZip.__new__(RemoteZip)
    remote.url = index['url']
    # One request normally contains header + payload. The generous extra-field
    # allowance avoids a separate small TLS request for every member.
    length = min(30+len(entry['name'].encode())+1024+entry['compressed'], index['size']-entry['offset'])
    payload, size = remote.fetch(entry['offset'], length)
    header = payload[:30]
    if size != index['size'] or header[:4] != b'PK\x03\x04':
        raise ValueError('Archive/header changed')
    fields = struct.unpack('<4s5H3I2H', header)
    if fields[2] & 1 or fields[3] != entry['compression']:
        raise ValueError('Unsupported or mismatched ZIP member')
    variable = fields[-2]+fields[-1]
    required = 30+variable+entry['compressed']
    if required > len(payload):
        remaining, size = remote.fetch(entry['offset']+len(payload), required-len(payload))
        payload += remaining
    data = payload[30:required]
    if size != index['size'] or data[:fields[-2]].decode('utf-8') != entry['name']:
        raise ValueError('ZIP member identity mismatch')
    compressed = data[variable:]
    if entry['compression'] == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif entry['compression'] == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(content) != entry['size'] or zlib.crc32(content) != entry['crc']:
        raise ValueError('ZIP member integrity check failed')
    return content


def pairs(index):
    """Match native names, including GOOSE-Ex sensor suffixes."""
    entries = {x['name']: x for x in index['entries']}
    result = []
    for name, cloud in sorted(entries.items()):
        path = Path(name)
        if path.suffix != '.bin' or not {'velodyne', 'lidar'}.intersection(path.parts):
            continue
        prefix = str(path.parent).replace('velodyne/', 'labels/').replace('lidar/', 'labels/')
        stem = re.sub(r'_(vls128|ouster|lidar|pcl).*$', '', path.stem)
        label = entries.get(f'{prefix}/{stem}_goose.label')
        if label is None:
            raise ValueError(f'Missing label for {name}')
        result.append(dict(scenario=path.parent.name, frame=stem, cloud=cloud, label=label))
    return result


def download(plan_path, root, workers=4):
    root = Path(root).resolve()
    plan = json.loads(Path(plan_path).read_text())
    jobs = []
    indices = {}
    for selection in plan['selections']:
        name = selection['archive']
        if name not in indices:
            indices[name] = index_archive(name, root)
        index = indices[name]
        candidates = [x for x in pairs(index) if x['scenario'] == selection['scenario']]
        count = min(selection['frames'], len(candidates))
        if count < 1:
            raise ValueError(f'Empty selection: {selection}')
        # Uniform index spacing, fixed before inspecting semantic labels.
        selected = sorted({round(i*(len(candidates)-1)/max(1, count-1)) for i in range(count)})
        jobs.extend((index, candidates[i], selection) for i in selected)
    for name, index in indices.items():
        mapping = [x for x in index['entries'] if Path(x['name']).name == 'goose_label_mapping.csv']
        if len(mapping) != 1:
            raise ValueError('Expected one official ontology per archive')
        content = read_member(index, mapping[0])
        destination = root/name/'goose_label_mapping.csv'
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError('Ontology changed')
        destination.write_bytes(content)

    def task(job):
        index, pair, selection = job
        record = {k: selection[k] for k in ('archive', 'split', 'group', 'scenario')}
        record['frame'] = pair['frame']
        for key in ('cloud', 'label'):
            entry = pair[key]
            destination = root/index['archive']/entry['name']
            if not destination.resolve().is_relative_to(root):
                raise ValueError('Unsafe ZIP member path')
            if destination.exists():
                content = destination.read_bytes()
                if len(content) != entry['size'] or zlib.crc32(content) != entry['crc']:
                    raise ValueError(f'Existing file is corrupt: {destination}')
            else:
                content = read_member(index, entry)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix+'.part')
                temporary.write_bytes(content)
                temporary.replace(destination)
            record[key] = str(destination.relative_to(root))
            record[key+'_sha256'] = digest(content)
            record[key+'_crc32'] = entry['crc']
        return record

    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(task, job): job[1]['frame'] for job in jobs}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                failures.append(dict(frame=futures[future], error=repr(exc)))
            if (len(records)+len(failures)) % 10 == 0 or len(records)+len(failures) == len(jobs):
                status = dict(stage='downloading', complete=len(records), total=len(jobs), failed=len(failures))
                save_json(root/'download_status.json', status)
                print(json.dumps(status), flush=True)
    save_json(root/'download_failures.json', failures)
    if failures:
        raise RuntimeError(f'{len(failures)} downloads failed; rerun resumes verified members')
    save_json(root/'raw_manifest.json', dict(format='goose-raw-subset-v1', source='official GOOSE and GOOSE-Ex',
        plan_sha256=digest(Path(plan_path).read_bytes()), split_protocol=plan['split_protocol'],
        simulation_used=False, archives=[dict(archive=k, url=v['url'], size=v['size']) for k, v in indices.items()],
        records=sorted(records, key=lambda x:(x['split'], x['archive'], x['frame']))))
    save_json(root/'download_status.json', dict(stage='complete', complete=len(records), failed=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', nargs='+', choices=tuple(URLS))
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.index:
        for name in args.index:
            index = index_archive(name, args.root)
            counts = {}
            for pair in pairs(index):
                counts[pair['scenario']] = counts.get(pair['scenario'], 0)+1
            print(json.dumps(dict(archive=name, bytes=index['size'], scenarios=counts)), flush=True)
    if args.plan:
        download(args.plan, args.root, args.workers)


if __name__ == '__main__':
    main()
