#!/bin/bash
for old_name in Ex2_Raissi_*.npz; do
    new_name="${old_name/Raissi/PIGP}"
    mv "$old_name" "$new_name"
done