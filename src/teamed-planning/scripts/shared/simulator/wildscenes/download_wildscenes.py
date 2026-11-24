#!/usr/bin/env python3
"""Download an official WildScenes sequence's metric geometry, with provenance.

Uses the CSIRO public API to obtain temporary read-only S3 connection details.
No credentials are written to the output manifest or log. Dataset CC BY-NC-SA 4.0.
Requires requests and boto3. Network proxies are bypassed for the CSIRO S3 host.
"""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import time

import boto3
from botocore.config import Config
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True, help='Output directory containing raw/ and provenance manifests')
    parser.add_argument('--sequence', default='V-01', choices=('V-01', 'V-02', 'V-03', 'K-01', 'K-03'))
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    api = 'https://data.csiro.au/dap/'
    r = requests.get(api + 'ws/v2/collections/csiro:61541', timeout=45)
    r.raise_for_status()
    collection = r.json()
    (root / 'collection_metadata.json').write_text(json.dumps(collection, indent=2))
    cid = collection['dataCollectionId']
    r = requests.get(api + f'api/v2/collections/{cid}/files/s3', timeout=45)
    r.raise_for_status()
    info = r.json()
    client = boto3.client('s3', endpoint_url=info['endPointUrl'],
        aws_access_key_id=info['accessKey'], aws_secret_access_key=info['secretAccessKey'],
        region_name='us-east-1', config=Config(proxies={}, connect_timeout=20,
        read_timeout=90, max_pool_connections=args.workers, retries={'max_attempts':5}))
    prefix = info['remoteDirectory'].split('/', 1)[1] + 'data/WildScenes/'
    bucket = info['bucket']
    entries = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix+f'WildScenes3d/{args.sequence}/'):
        for obj in page.get('Contents', []):
            rel = obj['Key'][len(prefix):]
            if obj['Size'] and ('/Clouds/' in rel or '/Labels/' in rel or rel.endswith('/poses3d.csv')):
                entries.append({'key':obj['Key'], 'path':rel, 'bytes':obj['Size'],
                    'etag':obj['ETag'].strip('"')})
    # Include the 2D calibration/poses and six evenly spaced real photos as provenance.
    images = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix+f'WildScenes2d/{args.sequence}/'):
        for obj in page.get('Contents', []):
            rel = obj['Key'][len(prefix):]
            if obj['Size'] and '/image/' in rel:
                images.append(obj)
            elif obj['Size'] and rel.endswith(('.csv','.yaml','.yml','.npy')):
                entries.append({'key':obj['Key'], 'path':rel, 'bytes':obj['Size'], 'etag':obj['ETag'].strip('"')})
    for i in sorted(set(round(j*(len(images)-1)/5) for j in range(6))) if images else []:
        obj = images[i]
        entries.append({'key':obj['Key'], 'path':obj['Key'][len(prefix):], 'bytes':obj['Size'], 'etag':obj['ETag'].strip('"')})
    # Fetch metadata/photos first, then interleave each point cloud and its labels.
    entries.sort(key=lambda e: (0 if '/Clouds/' not in e['path'] and '/Labels/' not in e['path'] else 1, pathlib.Path(e['path']).stem, e['path']))
    manifest = {'dataset':'WildScenes v3', 'sequence':args.sequence,
        'doi':collection['doi'], 'licence':collection['licence'],
        'attribution':collection['attributionStatement'], 'collection_id':cid,
        'endpoint':info['endPointUrl'], 'bucket':bucket, 'files':entries}
    manifest_path = root/f'download_manifest_{args.sequence}.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Downloading {len(entries)} files / {sum(e['bytes'] for e in entries)/1e9:.3f} GB", flush=True)
    start = time.time()
    def download(entry):
        path = root/'raw'/entry['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        data = None
        if path.exists() and path.stat().st_size == entry['bytes']:
            candidate = path.read_bytes()
            if '-' in entry['etag'] or hashlib.md5(candidate).hexdigest() == entry['etag']:
                data = candidate
        if data is None:
            for attempt in range(5):
                try:
                    body = client.get_object(Bucket=bucket, Key=entry['key'])['Body']
                    try:
                        data = body.read()
                    finally:
                        body.close()
                    if len(data) != entry['bytes']:
                        raise ValueError('Size mismatch')
                    if '-' not in entry['etag'] and hashlib.md5(data).hexdigest() != entry['etag']:
                        raise ValueError('MD5 mismatch')
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError('Download failed: '+entry['path']) from exc
                    time.sleep(min(20, 2**attempt))
            tmp = path.with_suffix(path.suffix+'.part')
            tmp.write_bytes(data)
            tmp.replace(path)
        md5 = hashlib.md5(data).hexdigest()
        entry['md5'] = md5
        entry['sha256'] = hashlib.sha256(data).hexdigest()
        return entry
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(download,e) for e in entries]
        for f in concurrent.futures.as_completed(futures):
            done.append(f.result())
            if len(done)%100==0 or len(done)==len(entries):
                print(f'{len(done)}/{len(entries)}, {time.time()-start:.1f}s',flush=True)
    manifest['files']=sorted(done,key=lambda e:e['path'])
    manifest['verified_files']=len(done)
    manifest['verified_bytes']=sum(e['bytes'] for e in done)
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print('Complete; MD5 checked against S3 ETags; SHA256 stored.',flush=True)

if __name__=='__main__':main()
