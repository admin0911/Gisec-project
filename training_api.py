"""Local preparation and asynchronous training API, separate from feature extraction."""
import json
from pathlib import Path
import re
import threading
import uuid

from training.preparation import ARTIFACTS, prepare_dataset, load_preparation, preparation_summary
from training.web_comparison import train_comparison, validate_epochs
# Leila: ground truth is exposed only through a separate evaluation endpoint.
from training.demo_evaluation import evaluate_prepared

RUN_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
OUTPUT = ARTIFACTS/'web_training'


def _update(job_id,**changes):
    with JOBS_LOCK:
        JOBS[job_id].update(changes)
        snapshot = dict(JOBS[job_id])
        folder = OUTPUT/job_id
        folder.mkdir(parents=True,exist_ok=True)
        temporary = folder/'job.tmp'
        temporary.write_text(json.dumps(snapshot,allow_nan=False),encoding='utf-8')
        temporary.replace(folder/'job.json')


def get_job(job_id):
    if not isinstance(job_id,str) or not re.fullmatch('[0-9a-f]{32}',job_id):
        raise ValueError('Invalid training job ID.')
    with JOBS_LOCK:
        if job_id in JOBS: return dict(JOBS[job_id])
    path = (OUTPUT/job_id/'job.json').resolve()
    if path.parent.parent != OUTPUT.resolve() or not path.is_file():
        raise ValueError('Training job was not found.')
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved['status'] not in ('complete','error'):
        saved.update(status='error',message='Training was interrupted by a server restart. Start a new comparison.')
    return saved


def start_training(version,epochs):
    validate_epochs(epochs)
    load_preparation(version)
    if not RUN_LOCK.acquire(blocking=False):
        raise ValueError('A training comparison is already running. Wait for it to finish.')
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = dict(job_id=job_id,version=version,status='queued',progress=0,message='Training queued')
    def worker():
        try:
            import torch
            torch.set_num_threads(4)
            result = train_comparison(version,epochs,OUTPUT/job_id,
                lambda value,message:_update(job_id,status='running',progress=value,message=message))
            _update(job_id,status='complete',progress=100,message='Training comparison complete',result=result)
        except Exception as exc:
            _update(job_id,status='error',message=str(exc))
        finally:
            RUN_LOCK.release()
    threading.Thread(target=worker,daemon=True).start()
    return dict(job_id=job_id,status='queued')


def handle_training_request(handler):
    routes = ('/api/training/prepare','/api/training/prepared','/api/training/start','/api/training/job','/api/training/evaluation')
    if handler.path not in routes: return False
    try:
        length = int(handler.headers.get('Content-Length','0'))
        if not 0 < length <= 8192: raise ValueError('Invalid training request size.')
        request = json.loads(handler.rfile.read(length))
        if not isinstance(request,dict): raise ValueError('Invalid training request.')
        if handler.path.endswith('/prepare'):
            result = prepare_dataset(request.get('scan_id'))
        elif handler.path.endswith('/prepared'):
            result = preparation_summary(load_preparation(request.get('version')))
        elif handler.path.endswith('/evaluation'):
            result = evaluate_prepared(request.get('version'))
        elif handler.path.endswith('/start'):
            result = start_training(request.get('version'),request.get('epochs',5))
            handler._json(result,status=202)
            return True
        else:
            result = get_job(request.get('job_id'))
        handler._json(result)
    except (ValueError,OSError,KeyError,TypeError) as exc:
        handler._json({'error':str(exc)},status=400)
    return True
