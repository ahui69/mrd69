#!/bin/bash
#
# test_env.sh
# A script to validate the .env configuration file.
#
# Usage:
# 1. Copy .env.example to .env and fill in your values.
# 2. Make this script executable: chmod +x test_env.sh
# 3. Run the script from your terminal: ./test_env.sh

set -o nounset  # Exit on unset variables
set -o errexit  # Exit on error
set -o pipefail # Exit on pipe fails

# --- Test Setup ---
# Source the environment file.
# The script expects the .env file to be in the same directory.
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found. Please create one from .env.example."
    exit 1
fi

# Source the variables. `set -a` exports them, making them available.
set -a
source .env
set +a

echo "🚀 Running environment configuration tests..."

# Test case for RUNPOD_ENDPOINT_ID
# The :-0 provides a default value if USE_RUNPOD is not set or empty,
# preventing an error from `set -o nounset`.
if [[ "${USE_RUNPOD:-0}" == "1" ]]; then
    echo "🔎 USE_RUNPOD is enabled. Checking related variables..."
    : "${RUNPOD_API_KEY:?❌ TEST FAILED: RUNPOD_API_KEY must be set when USE_RUNPOD is 1.}"
    : "${RUNPOD_ENDPOINT_ID:?❌ TEST FAILED: RUNPOD_ENDPOINT_ID must be set when USE_RUNPOD is 1.}"
    echo "✅ PASSED: RunPod variables are set."
else
    echo "⚪️ USE_RUNPOD is not enabled. Skipping RunPod-specific variable checks."
fi

echo "🎉 All environment tests passed!"
exit 0