#!/bin/bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# One-command vast.ai setup for headless G1 locomanipulation data generation.
# Creates an instance from the pre-built datagen image, waits for SSH, logs into
# HuggingFace, and prints the SSH command to jump in and run Steps 2–5.
#
# Usage:
#   bash scripts/imitation_learning/locomanipulation_sdg/vast_ai_datagen_setup.sh \
#       --offer-id <OFFER_ID> [--hf-token <TOKEN>] [--disk <GB>] [--image <IMG>]
#
# Defaults:
#   --disk 200   (2× expected HDF5 for 1000 demos; bump for NuRec backgrounds)
#   --image par4ker/isaaclab3-datagen:develop
set -euo pipefail

OFFER_ID=""
HF_TOKEN=""
DISK_SIZE="200"
IMAGE="par4ker/isaaclab3-datagen:develop"

while [[ $# -gt 0 ]]; do
    case $1 in
        --offer-id) OFFER_ID="$2"; shift 2 ;;
        --hf-token) HF_TOKEN="$2"; shift 2 ;;
        --disk) DISK_SIZE="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if [ -z "$OFFER_ID" ]; then
    echo "Usage: bash $0 --offer-id <OFFER_ID> [--hf-token <TOKEN>] [--disk <GB>] [--image <IMG>]"
    exit 1
fi

# Create the instance. The datagen image mounts /workspace/datasets as a
# VOLUME, so a single --disk value covers both the image and the dataset.
echo "==> Creating instance from offer $OFFER_ID (image: $IMAGE, disk: ${DISK_SIZE}GB)..."
CREATE_OUTPUT=$(vastai create instance "$OFFER_ID" --image "$IMAGE" --disk "$DISK_SIZE" 2>&1)
echo "    $CREATE_OUTPUT"

# Parse instance ID — vast.ai prints a Python-ish dict with a 'new_contract' key.
INSTANCE_ID=$(echo "$CREATE_OUTPUT" | python3 -c "
import sys, re
text = sys.stdin.read()
try:
    match = re.search(r'\{.*\}', text)
    if match:
        d = eval(match.group())
        print(d.get('new_contract', ''))
except Exception:
    pass
" 2>/dev/null)

if [ -z "$INSTANCE_ID" ]; then
    INSTANCE_ID=$(echo "$CREATE_OUTPUT" | grep -oP "'new_contract':\s*\K\d+")
fi

if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: Could not parse instance ID from: $CREATE_OUTPUT"
    exit 1
fi
echo "==> Instance ID: $INSTANCE_ID"

# Wait for vast.ai to assign SSH host/port.
echo "==> Waiting for instance to be ready..."
MAX_RETRIES=40
RETRY_COUNT=0
while true; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ "$RETRY_COUNT" -gt "$MAX_RETRIES" ]; then
        echo "ERROR: Timed out waiting for instance after $((MAX_RETRIES * 15))s"
        echo "    Debug: run 'vastai show instances --raw' to check instance state"
        exit 1
    fi

    INSTANCE_INFO=$(vastai show instances --raw 2>/dev/null) || {
        echo "    vastai command failed, retrying in 15s..."
        sleep 15
        continue
    }

    read -r SSH_HOST SSH_PORT INST_STATUS < <(echo "$INSTANCE_INFO" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
except json.JSONDecodeError:
    print('JSON_ERR JSON_ERR json_parse_failed')
    sys.exit(0)
for inst in data:
    if str(inst.get('id')) == '$INSTANCE_ID':
        status = inst.get('actual_status', inst.get('status_msg', 'unknown'))
        addr = inst.get('ssh_host') or inst.get('public_ipaddr') or ''
        port = inst.get('ssh_port') or inst.get('direct_port_start') or ''
        if not port:
            ports = inst.get('ports', {})
            if isinstance(ports, dict) and '22/tcp' in ports:
                mapping = ports['22/tcp']
                if isinstance(mapping, list) and mapping:
                    port = mapping[0].get('HostPort', '')
                elif isinstance(mapping, (int, str)):
                    port = str(mapping)
        print(f'{addr} {port} {status}')
        sys.exit(0)
print('NO_MATCH NO_MATCH instance_not_found')
" 2>&1) || true

    if [ "$RETRY_COUNT" -eq 1 ] || [ $((RETRY_COUNT % 4)) -eq 0 ]; then
        echo "    [debug] host='$SSH_HOST' port='$SSH_PORT' status='$INST_STATUS' (attempt $RETRY_COUNT)"
    fi

    if [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] \
        && [ "$SSH_HOST" != "None" ] && [ "$SSH_PORT" != "None" ] \
        && [ "$SSH_HOST" != "JSON_ERR" ] && [ "$SSH_HOST" != "NO_MATCH" ]; then
        break
    fi
    echo "    Instance not ready yet, retrying in 15s..."
    sleep 15
done

echo "==> Host: $SSH_HOST, Port: $SSH_PORT"

echo "==> Waiting for SSH to accept connections..."
while ! ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=5 -p "$SSH_PORT" "root@$SSH_HOST" "echo ok" &>/dev/null; do
    echo "    SSH not ready yet, retrying in 10s..."
    sleep 10
done

SSH_CMD="ssh -o StrictHostKeyChecking=accept-new -p $SSH_PORT root@$SSH_HOST"

# Sanity: the datagen image ships IsaacLab pre-installed at /workspace/IsaacLab.
# Verify the overlay worked (the NGC base ships 3.0.0-beta1; we replaced it with
# the develop-branch checkout that carries the locomanipulation SDG scripts).
echo "==> Verifying IsaacLab install on remote..."
$SSH_CMD "test -f /workspace/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/generate_data.py" || {
    echo "ERROR: /workspace/IsaacLab/scripts/imitation_learning/locomanipulation_sdg/generate_data.py missing on remote."
    echo "       The image at '$IMAGE' does not appear to be the datagen overlay. Rebuild with docker/Dockerfile.datagen and re-push."
    exit 1
}

if [ -n "$HF_TOKEN" ]; then
    echo "==> Logging into HuggingFace..."
    $SSH_CMD "hf auth login --token $HF_TOKEN"
fi

echo ""
echo "==> Setup complete! Instance ID: $INSTANCE_ID"
echo "    SSH in with:"
echo "    $SSH_CMD"
echo ""
echo "    Dataset volume is mounted at /workspace/datasets inside the container."
echo "    Point --dataset / --output_file flags at that path."
echo ""
echo "    To destroy when finished:"
echo "    vastai destroy instance $INSTANCE_ID"
