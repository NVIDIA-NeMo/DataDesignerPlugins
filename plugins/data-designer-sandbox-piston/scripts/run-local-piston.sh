#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

PISTON_IMAGE="${PISTON_IMAGE:-ghcr.io/engineer-man/piston:latest}"
PISTON_PORT="${PISTON_PORT:-2000}"
CONTAINER_NAME="${CONTAINER_NAME:-data-designer-piston}"
PISTON_DATA_VOLUME="${PISTON_DATA_VOLUME:-${CONTAINER_NAME}-data}"

if ! command -v docker >/dev/null 2>&1; then
    echo "docker is required to start a local Piston sandbox" >&2
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run \
    --privileged \
    --detach \
    --interactive \
    --tty \
    --publish "${PISTON_PORT}:2000" \
    --volume "${PISTON_DATA_VOLUME}:/piston" \
    --name "${CONTAINER_NAME}" \
    "${PISTON_IMAGE}" >/dev/null

echo "Piston sandbox started at http://localhost:${PISTON_PORT}"
echo "Piston runtime packages are stored in Docker volume ${PISTON_DATA_VOLUME}"
