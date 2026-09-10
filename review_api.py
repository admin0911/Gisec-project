"""Local HTTP adapter for paginated human review."""
import json
# Leila: evaluate saved scans independently of training and human decisions.
from cleaning.scan_evaluation import evaluate_scan
from cleaning.human_review import review_page, save_review, review_summary, ReviewConflict


def handle_review_request(handler):
    if handler.path not in ('/api/review/page', '/api/review/save', '/api/review/summary', '/api/review/evaluation'):
        return False
    try:
        length = int(handler.headers.get('Content-Length', '0'))
        if not 0 < length <= 65536:
            raise ValueError('Invalid review request size.')
        request = json.loads(handler.rfile.read(length))
        if not isinstance(request, dict):
            raise ValueError('Expected a review request.')
        job = request.get('job_id')
        if handler.path.endswith('/evaluation'):
            result = evaluate_scan(job)
        elif handler.path.endswith('/summary'):
            result = review_summary(job)
        elif handler.path.endswith('/page'):
            result = review_page(job,request.get('group','uncertain'),request.get('page',0),request.get('page_size',20))
        else:
            result = save_review(job,request.get('changes'),request.get('revision'))
        handler._json(result)
    except ReviewConflict as exc:
        handler._json({'error':str(exc)},status=409)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        handler._json({'error':str(exc)},status=400)
    return True
