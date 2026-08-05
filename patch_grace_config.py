"""
Patch crowd_nav/configs/config.py in place for GRACE E1's Stage B / Stage C
curriculum, without committing actual config values to git.

config.py is a shared, live "dial" file -- it currently reflects whatever
experiment last touched it (e.g. the SF fine-tuning work), and it's read
fresh every time train.py runs. Baking specific stage values into a git
commit risks silently clobbering unrelated in-progress work on the next
pull. This script edits only the named attributes, by name, regardless of
their current value, so it's safe to run no matter what state the file is
currently in -- and it's auditable before running, since STAGE_B/STAGE_C
below are the exact values documented in GRACE_E1_TRAINING_COMMANDS.txt.

Usage:
  python patch_grace_config.py B     # apply Stage B values
  python patch_grace_config.py C     # apply Stage C values
  python patch_grace_config.py B --dry-run   # print the diff, change nothing

Every attribute must already exist in config.py with exactly one assignment
line, or this refuses to proceed rather than silently doing nothing.
"""

import argparse
import re
import sys

CONFIG_PATH = "crowd_nav/configs/config.py"

STAGE_B = {
    "grace.freeze_backbone": "True",
    "grace.freeze_nav":      "False",
    "grace.use_aux_loss":    "False",
    "group.num_groups":      "2",
    "group.types":           "['static_f', 'dynamic_lf']",
    "group.num_on_path":     "1",
    "realistic.enabled":     "False",
    "sim.human_num":         "15",
    "sim.circle_radius":     "6",
    "sim.arena_size":        "6",
}

STAGE_C = {
    "grace.freeze_backbone": "False",
    "grace.freeze_nav":      "False",
    "grace.use_aux_loss":    "True",
    "group.num_groups":      "3",
    "group.types":           "['static_f', 'dynamic_lf', 'dynamic_free']",
    "group.num_on_path":     "2",
    "realistic.enabled":     "True",
    "sim.human_num":         "20",
    "sim.circle_radius":     "8.5",
    "sim.arena_size":        "8.5",
}


def patch(text: str, values: dict) -> tuple:
    """Returns (new_text, list of (attr, old_value, new_value) changes)."""
    changes = []
    for attr, new_val in values.items():
        pattern = re.compile(
            r"^(?P<indent>[ \t]*)" + re.escape(attr) +
            r"(?P<eq>[ \t]*=[ \t]*)(?P<value>[^#\n]*?)(?P<tail>[ \t]*(#.*)?)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if len(matches) == 0:
            print(f"ERROR: attribute '{attr}' not found in {CONFIG_PATH} -- "
                  f"refusing to proceed (it may have been renamed).", file=sys.stderr)
            sys.exit(1)
        if len(matches) > 1:
            print(f"ERROR: attribute '{attr}' matched {len(matches)} lines -- "
                  f"ambiguous, refusing to proceed.", file=sys.stderr)
            sys.exit(1)
        m = matches[0]
        old_val = m.group("value").strip()
        if old_val == new_val:
            changes.append((attr, old_val, new_val, False))
            continue
        replacement = f"{m.group('indent')}{attr}{m.group('eq')}{new_val}{m.group('tail')}"
        text = text[:m.start()] + replacement + text[m.end():]
        changes.append((attr, old_val, new_val, True))
    return text, changes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["B", "C"])
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change, write nothing")
    args = ap.parse_args()

    values = STAGE_B if args.stage == "B" else STAGE_C
    with open(CONFIG_PATH) as f:
        original = f.read()

    new_text, changes = patch(original, values)

    print(f"GRACE E1 Stage {args.stage} config patch:")
    for attr, old, new, did_change in changes:
        marker = "->" if did_change else "== (already correct)"
        print(f"  {attr:24s} {old:35s} {marker} {new}")

    if args.dry_run:
        print("\n--dry-run: no changes written.")
        return

    if new_text == original:
        print("\nNo changes needed -- config.py already matches this stage.")
        return

    with open(CONFIG_PATH, "w") as f:
        f.write(new_text)
    print(f"\nWrote {CONFIG_PATH}.")


if __name__ == "__main__":
    main()
