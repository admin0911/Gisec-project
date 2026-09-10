"""Small adapter between the local feature server and label-flip scan service."""
import json
import threading
import uuid

from detectors.label_flip.web_scan import ARTIFACTS, availability, feature_pair, run_scan, saved_feature_pairs
from detectors.label_flip.scan_cache import find_cached_scan
from cleaning.human_review import restored_scan_job

SCAN_LOCK = threading.Lock()


def handle_scan_request(handler, jobs, jobs_lock, update_job):
    """Return False for unrelated routes so the teammate's extraction API continues."""
    if handler.path not in ('/api/label-flip/saved', '/api/label-flip/ready', '/api/label-flip/scan'):
        return False
    try:
        size = int(handler.headers.get('Content-Length', '0'))
        if not 0 < size <= 8192:
            raise ValueError('Invalid scan request size.')
        request = json.loads(handler.rfile.read(size))
        if handler.path.endswith('/saved') and isinstance(request, dict):
            handler._json({'pairs': saved_feature_pairs()})
            return True
        if not isinstance(request, dict) or not isinstance(request.get('feature_file'), str):
            raise ValueError('Select a completed feature extraction first.')
        feature_file = request['feature_file']
        if handler.path.endswith('/ready'):
            handler._json(availability(feature_file))
            return True
        feature_pair(feature_file)
        # Leila: reopen the original scan ID so saved human review remains attached.
        cached_id = find_cached_scan(feature_file)
        if cached_id:
            try:
                cached = restored_scan_job(cached_id)
            except (ValueError,OSError,KeyError,TypeError):
                cached = None
            if cached is not None:
                with jobs_lock:
                    jobs[cached_id] = cached
                handler._json(dict(job_id=cached_id,status='complete',reused=True))
                return True
    except (ValueError, TypeError, OSError) as exc:
        handler._json({'error': str(exc)}, status=400)
        return True
    if not SCAN_LOCK.acquire(blocking=False):
        handler._json({'error': 'A label-flip scan is already running. Wait for it to finish.'}, status=409)
        return True
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = dict(job_id=job_id, status='queued', progress=0, message='Scan queued')

    def worker():
        try:
            def progress(step, message):
                update_job(job_id, status='running', progress=round(step / 7 * 100), message=message)
            result = run_scan(feature_file, ARTIFACTS / 'label_flip_scans' / job_id, progress)
            update_job(job_id, status='complete', progress=100, message='Label-flip scan complete', result=result)
        except Exception as exc:
            update_job(job_id, status='error', progress=0, message=str(exc))
        finally:
            SCAN_LOCK.release()

    threading.Thread(target=worker, daemon=True).start()
    handler._json(dict(job_id=job_id, status='queued'), status=202)
    return True
