"""Offline HTML presentation of detector evidence and separate evaluation."""
import base64
import csv
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


STYLE = """
body{margin:0;background:#101725;color:#e8edf6;font:16px/1.6 system-ui,sans-serif}
main{max-width:1160px;margin:auto;padding:36px 24px}h1{font-size:32px;line-height:1.2}
h2{margin-top:32px}a{color:#6ee7cb}.muted{color:#abb7cd}.notice{border-left:4px solid
#f7c86a;padding:12px 18px;background:#202838}table{border-collapse:collapse;width:100%;
margin:20px 0}th,td{text-align:left;padding:10px;border-bottom:1px solid #344055}
.scroll{overflow-x:auto}.cards{display:grid;grid-template-columns:repeat(auto-fit,
minmax(230px,1fr));gap:14px}.card{background:#1b2537;padding:18px;border-radius:12px;
overflow-wrap:anywhere}.card strong{font-size:26px}.grid{display:grid;
grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:16px}
figure{margin:0;background:#1b2537;padding:12px;border-radius:10px}figure img{
width:128px;height:128px;object-fit:contain;image-rendering:pixelated;display:block;
margin:0 auto 12px}figcaption{font-size:13px;overflow-wrap:anywhere}code{
overflow-wrap:anywhere}.pattern{font-size:14px}footer{margin-top:32px;color:#abb7cd}
"""


def page(title, body):
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{STYLE}</style><main>{body}</main></html>")


def rate(value):
    return "N/A" if value is None else f"{100 * value:.2f}%"


def report_html(result, pixels):
    candidates = np.flatnonzero(result["candidate_flags"])
    evaluation = result.get("evaluation", {})
    title = "Backdoor scan review"
    body = f"<h1>{title}</h1><p class='muted'>{escape(result['input']['feature_file'] or result['input']['image_file'])}</p>"
    body += (f"<div class='cards'><div class='card'>Images evaluated<br><strong>{len(pixels.sample_ids):,}</strong></div>"
             f"<div class='card'>Review candidates<br><strong>{len(candidates):,}</strong></div>"
             f"<div class='card'>Detector time<br><strong>{result['runtime_seconds']:.2f}s</strong></div></div>"
             "<p class='notice'>Candidates require review. No samples have been removed or relabelled. "
             "Feature detectors can miss a patch that the pixel detector finds. Zero flags does not prove a dataset is clean.</p>"
             "<p><a href='results.json'>Complete JSON</a> · <a href='samples.csv'>All sample scores (CSV)</a></p>"
             "<h2>Detector results</h2><div class='scroll'><table><tr><th>Detector</th><th>Flagged</th>"
             "<th>Poison caught</th><th>Clean flagged</th><th>Recall</th><th>Precision</th><th>Clean FPR</th></tr>")
    rows = [(name, int(np.sum(item["flags"]))) for name, item in result["detectors"].items()]
    rows.append(("candidates", len(candidates)))
    for name, flagged in rows:
        metric = evaluation.get(name)
        label = name + (" (comparison only)" if name in result["settings"].get("comparison_only", []) else "")
        details = ([f"{metric['tp']}/{metric['tp'] + metric['fn']}", str(metric['fp']),
                    rate(metric["recall"]), rate(metric["precision"]), rate(metric["false_positive_rate"])]
                   if metric else ["N/A"] * 5)
        body += "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in [label, flagged, *details]) + "</tr>"
    body += ("</table></div><p class='muted'>" + escape(result["evaluation_note"]) +
             " Precision is undefined when nothing is flagged; recall is undefined when there are no known poisoned samples. "
             "Detection recall is not model attack success rate (ASR).</p>")
    active = result["settings"].get("active_detectors", list(result["detectors"]))
    body += "<p>Review candidates use: " + escape(", ".join(active)) + ". Comparison-only flags are retained in JSON/CSV but do not add candidates.</p>"
    pixel_name = "contrast_patch" if "contrast_patch" in active else "repeated_patch"
    pixel_result = result["detectors"][pixel_name]
    patterns = pixel_result["evidence"]["patterns"]
    body += "<h2>Pixel evidence</h2><p>Coordinates start at zero. Patterns are inferred from the submitted pixels and current labels.</p>"
    if patterns:
        body += "<div class='scroll'><table class='pattern'><tr><th>Position (row, column)</th><th>Size</th><th>Matches</th><th>Dominant label</th><th>Label purity</th></tr>"
        for pattern in patterns[:100]:
            values = [f"({pattern['row']}, {pattern['column']})", f"{pattern['size']} × {pattern['size']}",
                      pattern["support"], pattern["dominant_label"], rate(pattern["purity"])]
            body += "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>"
        body += f"</table></div><p>Showing {min(100, len(patterns))} of {len(patterns)} accepted patterns. All are in JSON.</p>"
    else:
        body += "<p>No repeated patch passed the selected profile.</p>"
    diagnostic_counts = pixel_result["evidence"].get("diagnostic_counts", {})
    if diagnostic_counts:
        body += "<h2>Candidate diagnostics</h2><p>Counts describe candidate patterns, not individual images. Full details are in JSON.</p><table><tr><th>Decision</th><th>Pattern groups</th></tr>"
        for name, count in diagnostic_counts.items():
            body += f"<tr><td>{escape(name)}</td><td>{count}</td></tr>"
        body += "</table>"
    body += f"<h2>Flagged images</h2><p>Showing the first {min(24, len(candidates))} of {len(candidates)} candidates in input order. All sample IDs are in CSV.</p><div class='grid'>"
    for index in candidates[:24]:
        values = np.rint(pixels.images[index] * 255).astype(np.uint8)
        values = values[0] if values.shape[0] == 1 else values.transpose(1, 2, 0)
        buffer = BytesIO(); Image.fromarray(values).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        methods = ", ".join(name for name, item in result["detectors"].items() if name in active and item["flags"][index])
        body += (f"<figure><img alt='Submitted image' src='data:image/png;base64,{encoded}'>"
                 f"<figcaption><b>{escape(str(pixels.sample_ids[index]))}</b><br>Current label: "
                 f"{escape(str(pixels.labels[index]))}<br>Flagged by: {escape(methods)}</figcaption></figure>")
    body += "</div><footer>Selected profile: " + escape(pixel_result["settings"].get("profile", pixel_name)) + ". Feature detectors remain experimental. No classifier training or automatic cleaning was performed.</footer>"
    return page(title, body)


def write_sample_csv(path, result):
    names = list(result["detectors"])
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_id", "review_candidate", *[f"{n}_{s}" for n in names for s in ("flag", "score")]])
        for i, sid in enumerate(result["sample_ids"]):
            # Avoid formula execution when arbitrary string IDs are opened in Excel.
            sid = str(sid)
            if sid.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
                sid = "'" + sid
            row = [sid, bool(result["candidate_flags"][i])]
            for name in names:
                row.extend([bool(result["detectors"][name]["flags"][i]), float(result["detectors"][name]["scores"][i])])
            writer.writerow(row)


def index_html(entries):
    body = "<h1>Backdoor scan reports</h1><p>Pixels and embeddings evaluated separately. Active detector flags create review candidates; comparison-only results remain separate.</p><div class='cards'>"
    for entry in entries:
        body += (f"<div class='card'><a href='{escape(entry['link'], quote=True)}'>{escape(entry['name'])}</a>"
                 f"<p>{entry['rows']:,} images · <strong>{entry['flagged']}</strong> candidates</p></div>")
    return page("Backdoor scan reports", body + "</div><p class='notice'>No automatic removal. A clean verdict cannot be inferred from zero flags.</p>")
