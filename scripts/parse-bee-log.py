#!/usr/bin/env python3
"""Parse bee.log and extract condensed IME-relevant lines.

Usage: scripts/parse-bee-log.py [--tail N] [--since TIME]
  --tail N     Show last N relevant lines (default: 50)
  --since TIME Only show lines after TIME (e.g. "11:28" or "2026-04-10T11:28")
"""

import re, sys, os, argparse
from collections import deque

LOG = os.path.expanduser(
    "~/Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee/bee.log"
)

parser = argparse.ArgumentParser()
parser.add_argument("--tail", type=int, default=50)
parser.add_argument("--since", default="")
args = parser.parse_args()

LINE_RE = re.compile(r"^(\S+)\s+INFO\s+(IME|APP):\s+(.*)")

SKIP = re.compile(
    r"^(init!|setDelegate|hidePalettes|compositionAttributes|selectionRange|"
    r"replacementRange|mark\(forStyle|delegate|server\b|menu\b|inputControllerWillClose|"
    r"annotationSelected|candidateSelectionChanged|candidateSelected|"
    r"composedString|originalString|updateComposition|recognizedEvents|doCommand|"
    r"AUDIO|TIS SELECT|VOXIPC: (claimSession|prepareSession))"
)

UUID_PFX = re.compile(r"id=[A-F0-9-]+-\d+-[A-F0-9]+ ")
PROXY_TYPE = re.compile(r"type=_IPMDServerClientWrapperLegacy ?")
NSNOTFOUND_PAIR = re.compile(r"\{9223372036854775807, 9223372036854775807\}")
NSNOTFOUND_ZERO = re.compile(r"\{9223372036854775807, 0\}")
NSNOTFOUND_PFX = re.compile(r"\{9223372036854775807, ")
VALID_ATTRS = re.compile(r"validMarkedAttrs=\[[^\]]*\] ?")
MULTI_SPACE = re.compile(r"  +")

TIME_RE = re.compile(r"T(\d+:\d+:\d+\.\d{3})")

buf = deque(maxlen=args.tail)

with open(LOG, errors="replace") as f:
    for line in f:
        m = LINE_RE.match(line)
        if not m:
            continue
        ts, src, msg = m.group(1), m.group(2), m.group(3)

        if args.since and ts < args.since:
            continue
        if SKIP.match(msg):
            continue

        # Shorten timestamp to HH:MM:SS.mmm
        tm = TIME_RE.search(ts)
        if tm:
            ts = tm.group(1)

        # Condense client descriptions
        msg = UUID_PFX.sub("", msg)
        msg = PROXY_TYPE.sub("", msg)
        msg = NSNOTFOUND_PAIR.sub("{∅}", msg)
        msg = NSNOTFOUND_ZERO.sub("{∅,0}", msg)
        msg = NSNOTFOUND_PFX.sub("{∅,", msg)
        msg = VALID_ATTRS.sub("", msg)
        msg = MULTI_SPACE.sub(" ", msg).rstrip()

        buf.append(f"{ts} {src}: {msg}")

for line in buf:
    print(line)
