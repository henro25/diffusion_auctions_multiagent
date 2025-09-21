#!/bin/bash

DIR="/datastor1/gdaras/diffusion_auctions_multiagent/alignment/alignment_3_agent"
OUTFILE="/datastor1/gdaras/alignment_3_agent_json.tar.gz"

echo "Compressing $DIR ..."
tar -czf "$OUTFILE" -C "$(dirname "$DIR")" "$(basename "$DIR")"

echo "Compression finished. Archive saved as $OUTFILE"

if [ -f "$OUTFILE" ]; then
    echo "Archive size:"
    du -sh "$OUTFILE"
else
    echo "Archive not found!"
fi