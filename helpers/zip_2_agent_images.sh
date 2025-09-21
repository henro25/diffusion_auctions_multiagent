#!/bin/bash

DIR="/datastor1/gdaras/diffusion_auctions_multiagent/images/images_2_agent"
OUTFILE="/datastor1/gdaras/images_2_agent.tar.gz"

echo "Compressing $DIR ..."
tar -czf "$OUTFILE" -C "$(dirname "$DIR")" "$(basename "$DIR")"

echo "Compression finished. Archive saved as $OUTFILE"

OUTFILE="/datastor1/gdaras/images_2_agent.tar.gz"

if [ -f "$OUTFILE" ]; then
    echo "Archive size:"
    du -sh "$OUTFILE"
else
    echo "Archive not found!"
fi