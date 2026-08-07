# Experiment — FPN-Fusion (abandoned)

Dual-backbone architecture: two-stage SE (squeeze-excitation) channel-attention fusion — first a
per-task SE block, then a cross-task fusion SE block — feeding into a shared PANet neck, with
dual task-specific heads on top.

**Status: abandoned — failed due to negative transfer.** Described in the paper as a design
that was explored and explicitly not adopted, in favour of Hook-Tap.

## Files needed here

No code for this attempt has been uploaded to any repo-building session yet, even partial or
broken versions. **Please send whatever exists** (however incomplete) — it's useful for the
repo's `experiments/`/`archive/` record even though it wasn't the adopted approach, and having
it available strengthens the paper's claim that FPN-Fusion was tried and rejected rather than
just asserted.
