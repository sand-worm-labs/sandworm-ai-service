#!/usr/bin/env bash

# =====================================
# Config
# =====================================

BASE_URL="http://localhost:8000"
API_KEY="${OPENROUTER_API_KEY}"
MODEL="~anthropic/claude-haiku-latest"
HANDSHAKE_TOKEN="1fcd4bf970d180bb56394fece06029ebf526555c9226ce38"


# =====================================
# Test Cases   [message]|[expected]
# =====================================

CASES=(
  # should complete
  "analyze wash trading on 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13d on Ethereum|complete"
  "show Uniswap V3 trading volume on Base this month|complete"
  "track all USDC transfers on Polygon in the last 7 days|complete"
  "show Aave liquidations on Ethereum in the last 30 days|complete"
  "analyze top holders of 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13d on Ethereum|complete"
  "show Uniswap V3 trading volume on all pools on Base this month|complete"
  "show me Curve TVL on Ethereum over the last 90 days|complete"
  "analyze NFT floor price for 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13d on Ethereum|complete"
  "show Compound utilization rate on Ethereum right now|complete"
  "track WBTC mint and burn events on Ethereum last 6 months|complete"
  "show dYdX revenue on Ethereum this quarter|complete"
  "analyze Lido staking inflows on Ethereum this year|complete"
  "show GMX open interest on Arbitrum last 14 days|complete"
  "track Blur marketplace volume on Ethereum last 30 days|complete"
  "analyze wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 on Ethereum|complete"
  "show top 50 ARB token holders on Arbitrum|complete"
  "track USDC supply changes on Ethereum this year|complete"
  "analyze ETH/USDC pool on Uniswap V3 on Ethereum this month|complete"
  "show MakerDAO DAI collateral ratios on Ethereum last 90 days|complete"
  "analyze PEPE token buy sell pressure on Ethereum since launch|complete"
  "show Velodrome TVL breakdown on Optimism this month|complete"
  "track vitalik.eth transfers on Ethereum this month|complete"
  "show OpenSea wash trade volume on Ethereum last 30 days|complete"
  "analyze BAYC royalty revenue on Ethereum in 2024|complete"
  "show Uniswap V3 fee revenue on Polygon last 7 days|complete"
  "track 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 DeFi positions on Base|complete"
  "analyze ETH burn rate on Ethereum since EIP-1559|complete"
  "show Balancer pool impermanent loss on Ethereum in 2024|complete"
  "track Azuki flippers on Ethereum this week|complete"
  "show Stargate bridge volume on Arbitrum last 30 days|complete"
  "analyze OP token accumulation on Optimism before last price pump|complete"
  "show Pudgy Penguins holder distribution on Ethereum|complete"
  "show USDC/ETH Uniswap V3 pool fee APR on Ethereum this month|complete"
  "analyze Aave V3 liquidations on Base last 7 days|complete"
  "show top wallets by gas spent on Ethereum this month|complete"
  "track NFT mints on 0xBC4CA0EdA7647A8aB7C2061c2E118A18a936f13d on Ethereum|complete"
  "show Curve FRAX pool health on Ethereum today|complete"
  "analyze CryptoPunks floor price on Ethereum last 90 days|complete"
  "show Uniswap V3 fee revenue on Polygon last 7 days|complete"
  "analyze BAYC holder distribution on Ethereum|complete"

  # should clarify
  "show me top traders on the exchange|clarify"
  "analyze this wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045|clarify"
  "compare Uniswap and Curve|clarify"
  "analyze V3 liquidity|clarify"
  "track USDC|clarify"
  "show me bridge activity|clarify"
  "analyze the whale moving ETH to Base|clarify"
  "show wash trading activity|clarify"
  "track the top NFT collection|clarify"
  "analyze my portfolio|clarify"
  "show me what got rugged last week|clarify"
  "compare this wallet across chains 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045|clarify"
  "analyze Binance wallet activity|clarify"
  "show trading volume|clarify"
  "track the bridge to Base|clarify"
  "show me suspicious wallets|clarify"
  "analyze 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045 and 0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8|clarify"
  "track whale movements|clarify"
  "show DEX volume this month|clarify"
  "analyze USDC|clarify"
)


# =====================================
# Runner
# =====================================

run_test() {
  local message="$1"
  local expected="$2"

  local body
  body=$(jq -n \
    --arg msg "$message" \
    --arg key "$API_KEY" \
    --arg model "$MODEL" \
    '{message: $msg, openrouter_api_key: $key, model: $model,
      context: {user_id: "user_123", workspace_id: "ws_123", document_id: "doc_123"},
      history: []}')

  local raw
  raw=$(curl -s --no-buffer -X POST "$BASE_URL/intent/parse_intent" \
    -H "Content-Type: application/json" \
    -H "x-handshake-token: $HANDSHAKE_TOKEN" \
    -d "$body")

  local status
  status=$(echo "$raw" | jq -r '.status // "unknown"' 2>/dev/null) || status="parse_error"

  if [[ "$status" == "$expected" ]]; then
    printf "✓  [%-7s]  %s\n" "$status" "${message:0:70}"
    return 0
  else
    printf "✗  [got: %-7s expected: %-7s]  %s\n" "$status" "$expected" "${message:0:55}"
    echo "   raw: ${raw:0:200}"
    return 1
  fi
}


# =====================================
# Main
# =====================================

total=${#CASES[@]}
passed=0
failed=0

echo ""
echo "Running $total tests..."
echo ""

for case in "${CASES[@]}"; do
  message="${case%|*}"
  expected="${case##*|}"

  if run_test "$message" "$expected"; then
    ((passed++))
  else
    ((failed++))
  fi
done

echo ""
echo "─────────────────────────────────────────"
echo "Passed: $passed / $total"
echo "Failed: $failed / $total"
