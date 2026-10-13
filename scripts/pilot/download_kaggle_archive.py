"""Use official Kaggle streaming/resume transport; never print signed URLs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def download(kernel, manifest, output, prefix=None):
    owner,slug,version=kernel.split('/')
    if owner!='rayyannmerchant' or version!='1':raise ValueError('Only assigned account saved version1 is supported')
    reference=json.loads(manifest.read_text(encoding='utf-8'));name=reference['run_id'].split('_seed42_')[0]+'_seed42_pilot.zip'
    api=KaggleApi();api.authenticate()
    assert api.config_values[api.CONFIG_NAME_USER]==owner
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(manifest,output/'artifact_manifest.json')
    item=None;token=None;began=time.perf_counter()
    with api.build_kaggle_client() as client:
        while item is None:
            request=ApiListKernelSessionOutputRequest();request.user_name=owner;request.kernel_slug=slug
            api._set_paging(request,20,token)
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            item=next((f for f in response.files or [] if f.file_name==name),None)
            token=response.next_page_token
            if item is None and not token:raise FileNotFoundError('Published archive missing')
    partial=output/(name+'.part')
    if prefix is not None:
        offset=prefix.stat().st_size
        assert 0<offset<reference['archive_bytes']
        pieces=output/'pieces';pieces.mkdir()
        ranges=[(i,min(i+8*2**20,reference['archive_bytes'])-1) for i in range(offset,reference['archive_bytes'],8*2**20)]
        def piece(pair):
            start,end=pair
            for attempt in range(6):
                path=pieces/f'{start:012d}_{end:012d}_attempt{attempt}.part'
                try:
                    with requests.get(item.url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,60)) as response:
                        if response.status_code==429:
                            delay=float(response.headers.get('Retry-After','30'));time.sleep(max(1,delay));continue
                        response.raise_for_status()
                        assert response.status_code==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{reference["archive_bytes"]}'
                        with path.open('xb') as f:
                            for data in response.iter_content(2**20):f.write(data)
                    assert path.stat().st_size==end-start+1
                    return path
                except (requests.RequestException,OSError):
                    if attempt==5:raise
                    time.sleep(min(2**(attempt+1),30))
        with ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(piece,ranges))
        with partial.open('xb') as f:
            with prefix.open('rb') as old:
                remaining=offset
                while remaining:
                    block=old.read(min(2**20,remaining));assert block;f.write(block);remaining-=len(block)
            for path in paths:
                with path.open('rb') as part:shutil.copyfileobj(part,f,2**20)
        transport='Official Kaggle API output URL; four bounded byte-range streams with exact206/Content-Range/size validation; retained45MiB prefix; full object SHA256 verified'
    else:
        with requests.get(item.url,stream=True,timeout=(30,60)) as response:
            response.raise_for_status()
            assert int(response.headers['Content-Length'])==reference['archive_bytes']
            api.download_file(response,str(partial),None,quiet=True,resume=True,max_retries=20,timeout=60)
        transport='Unmodified official KaggleApi.download_file with identity-checked HTTP Range retry'
    checksum=hashlib.sha256()
    with partial.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):checksum.update(block)
    assert partial.stat().st_size==reference['archive_bytes'] and checksum.hexdigest()==reference['archive_sha256']
    partial.rename(output/name)
    receipt={'status':'authorized_download_size_SHA256_verified','kernel':kernel,'archive_bytes':reference['archive_bytes'],'archive_sha256':checksum.hexdigest(),'seconds':time.perf_counter()-began,'transport':transport,'credentials':'No signed URL or credentials recorded'}
    (output/'download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kernel',required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prefix',type=Path)
    a=p.parse_args()
    try:download(a.kernel,a.manifest,a.output,a.prefix)
    except Exception as error:
        if a.output.exists():(a.output/'download_failure.json').write_text(json.dumps({'error_type':type(error).__name__,'scope':'Independent artifact transfer, not training; partial evidence retained'}))
        print('DOWNLOAD FAILED:',type(error).__name__,flush=True);raise SystemExit(1)
