# Optional image review

After a scan completes, click **Human review**, **Needs review**, or **Poisoned
data** on the results page. The latter group remains suspected, not confirmed.
The review page defaults to 20 images, with 50 and 100 also available.
Images come from the saved post-attack image bundle and are aligned by sample
ID with the scan. Their original CIFAR-10 resolution is 32 × 32.

Each tile shows the supplied class, sample ID, encoder vote counts and one
optional decision: **Keep**, **Quarantine**, or **Unsure**. No choice is selected
for an unreviewed sample. Suggested labels are unavailable in the current
saved scan schema; the UI does not invent them or use original-label truth.

**Save review** saves explicit pending choices across all visited pages.
**Next page**, Previous, changing group and changing page size do not save;
pending choices are preserved in the open page. Finish review and Back ask
Save, Don't save or Cancel if changes are pending. Don't save discards only
pending browser edits and returns to results; previously saved decisions remain.
Cancel stays on the page with pending edits intact. Unsure remains unresolved.
Leaving or refreshing with unsaved choices triggers
the browser's standard prompt. Concurrent saves from another tab return a
conflict rather than overwriting them silently.

Reviews are stored in `artifacts/label_flip_scans/<job_id>/human_review.json`,
with sample IDs, decisions, timestamps, revisions, edit history and the hash
of the scan evidence. Detector results and image/feature files are unchanged.
Opening review never creates Keep decisions. Use **Prepare training dataset**
on Scan results to include saved choices in a new training selection version.
Already prepared versions retain their original review snapshot.

The results and review pages can reopen completed scans after restarting the
server. No detector rerun is required. In-progress jobs still need the server
to remain running.
