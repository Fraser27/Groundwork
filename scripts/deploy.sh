#!/usr/bin/env bash
#
# Deploy Groundwork into an AWS account, start to finish, without prompting.
#
# Replaces steps 3, 4 and 5 of the manual install: resolving availability zones, bootstrapping and
# deploying, and closing the circular callback-URL requirement. Creating the first user is left out
# on purpose -- it is the one step a workshop participant should do themselves, and the script
# prints the exact commands at the end.
#
#   ./scripts/deploy.sh                 # us-east-1
#   ./scripts/deploy.sh eu-west-1       # or REGION=eu-west-1 ./scripts/deploy.sh
#
# Non-interactive by design: no prompts, no --require-approval, and every precondition is checked
# before the first 25-minute deploy rather than discovered inside it.

set -euo pipefail

REGION="${1:-${REGION:-us-east-1}}"
HOME_TENANT="${HOME_TENANT:-demo-firm}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDK_DIR="$REPO_ROOT/cdk"
CDK_JSON="$CDK_DIR/cdk.json"

# The two models every default needs. Checked rather than assumed: enabling access is a console
# action with no CLI equivalent, and without it the failure surfaces at the first document upload as
# an AccessDeniedException, which reads as a broken app rather than a missing checkbox.
#
# The **global inference profile** is what the app actually calls, so that is what is tested. A
# region-pinned id can be present while the global profile is not, and `get-foundation-model` only
# says a model exists in the region rather than that this account may invoke it. So each one is
# invoked for real, with the smallest possible request.
NOVA_PROFILE="global.amazon.nova-2-lite-v1:0"
TITAN_MODEL="amazon.titan-embed-text-v2:0"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# ── 1. Preconditions ────────────────────────────────────────────────────────────
say "Checking prerequisites"

for tool in aws node npm docker python3; do
  command -v "$tool" >/dev/null || die "$tool is not installed"
done

node_major="$(node --version | sed 's/^v\([0-9]*\).*/\1/')"
[ "$node_major" -ge 20 ] || die "Node 20+ required, found $(node --version)"

# `docker ps` rather than `docker --version`: the daemon has to be running, because the app image
# and the UI bundle are both built in containers.
docker ps >/dev/null 2>&1 || die "Docker is installed but not running"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
  || die "No usable AWS credentials. Run 'aws configure' or export a profile."

note "account   $ACCOUNT"
note "region    $REGION"
note "tenant    $HOME_TENANT"

# ── 2. Bedrock model access ─────────────────────────────────────────────────────
say "Checking Bedrock model access in $REGION"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
missing=()

if aws bedrock-runtime converse --region "$REGION" --model-id "$NOVA_PROFILE" \
    --messages '[{"role":"user","content":[{"text":"hi"}]}]' \
    --inference-config '{"maxTokens":1}' >/dev/null 2>"$tmp/nova.err"; then
  note "ok        $NOVA_PROFILE"
else
  missing+=("$NOVA_PROFILE")
  note "MISSING   $NOVA_PROFILE"
fi

# `--cli-binary-format raw-in-base64-out` because AWS CLI v2 expects `--body` base64-encoded and
# rejects raw JSON with "Invalid base64", which reads as a malformed request rather than a CLI
# convention. Embeddings have no Converse API, so this is invoke-model or nothing.
if aws bedrock-runtime invoke-model --region "$REGION" --model-id "$TITAN_MODEL" \
    --body '{"inputText":"hi"}' --content-type application/json \
    --cli-binary-format raw-in-base64-out \
    "$tmp/titan.out" >/dev/null 2>"$tmp/titan.err"; then
  note "ok        $TITAN_MODEL"
else
  missing+=("$TITAN_MODEL")
  note "MISSING   $TITAN_MODEL"
fi

if [ "${#missing[@]}" -gt 0 ]; then
  cat >&2 <<EOF

FAILED: this account cannot invoke these models in $REGION:
$(printf '  - %s\n' "${missing[@]}")

Enable them in the Bedrock console under Model access, then re-run. It is a console
action with no CLI equivalent, and access can take a few minutes to take effect.

  https://$REGION.console.aws.amazon.com/bedrock/home?region=$REGION#/modelaccess

The error was:
$(sed 's/^/  /' "$tmp"/*.err 2>/dev/null | head -6)
EOF
  exit 1
fi

# ── 3. Availability zones ───────────────────────────────────────────────────────
#
# AZ *names* are shuffled per account, so us-east-1a is a different physical zone in every account.
# AgentCore Runtime supports only a subset of zones, and putting a subnet in the wrong one fails
# LexGraphMcp with an error naming the subnet rather than the zone. So the names are resolved from
# the zone IDs the network stack declares as supported.
say "Resolving availability zones"

# Read the IDs straight out of the TypeScript rather than importing it: the file would need
# compiling, and the list is a literal. One source of truth, no build step.
SUPPORTED_IDS="$(
  sed -n "s/.*'$REGION': \[\(.*\)\].*/\1/p" "$CDK_DIR/lib/network-stack.ts" \
    | tr -d "' " | tr ',' '\n' | grep . || true
)"

[ -n "$SUPPORTED_IDS" ] || die "$REGION is not in SUPPORTED_AZ_IDS in cdk/lib/network-stack.ts.
Add the zone IDs that both AgentCore Runtime and OpenSearch Serverless support there first."

AZ_NAMES=()
while read -r name id; do
  if grep -qx "$id" <<<"$SUPPORTED_IDS"; then
    AZ_NAMES+=("$name")
  fi
done < <(aws ec2 describe-availability-zones --region "$REGION" \
  --query 'AvailabilityZones[].[ZoneName,ZoneId]' --output text)

# Two, because Neptune requires a subnet group spanning at least two zones.
[ "${#AZ_NAMES[@]}" -ge 2 ] \
  || die "Fewer than two supported zones in $REGION for this account. Found: ${AZ_NAMES[*]:-none}"

AZ_A="${AZ_NAMES[0]}"
AZ_B="${AZ_NAMES[1]}"
note "using     $AZ_A, $AZ_B"

# ── 4. Write the context ────────────────────────────────────────────────────────
say "Updating cdk.json"

set_context() {
  # One key at a time, parsed as JSON but written by patching the file, so a hand-maintained
  # config keeps its own formatting. Reserialising the whole document is easier and turns every
  # deploy into a 17-line diff over blank lines and array wrapping, which buries the one value
  # that actually changed.
  python3 - "$CDK_JSON" "$1" "$2" <<'PY'
import json, pathlib, re, sys

path, key, raw = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(raw)
text = path.read_text()
rendered = json.dumps(value)

# Match `"key": <anything up to the line-ending comma>`, arrays on one line included.
pattern = re.compile(rf'^(\s*){re.escape(json.dumps(key))}:\s*.*?(,?)$', re.M | re.S)
match = pattern.search(text)
if match:
    text = text[: match.start()] + f'{match.group(1)}{json.dumps(key)}: {rendered}{match.group(2)}' + text[match.end() :]
else:
    # Not present: add it as the last entry of "context", keeping the closing braces intact.
    ctx = re.search(r'^(\s*)"context":\s*\{', text, re.M)
    if ctx is None:
        raise SystemExit('no "context" object in cdk.json')
    depth, i = 0, ctx.end() - 1
    while i < len(text):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            break
        i += 1
    head = text[:i].rstrip()
    sep = "," if not head.endswith("{") else ""
    text = f'{head}{sep}\n{ctx.group(1)}  {json.dumps(key)}: {rendered}\n{ctx.group(1)}}}' + text[i + 1 :]

# Parse before writing. A regex that produced invalid JSON must not reach the file.
json.loads(text)
path.write_text(text)
print(f"    set       {key}={rendered}")
PY
}

set_context availabilityZones "[\"$AZ_A\", \"$AZ_B\"]"
set_context homeTenant "\"$HOME_TENANT\""

# ── 5. Install and bootstrap ────────────────────────────────────────────────────
say "Installing CDK dependencies"
(cd "$CDK_DIR" && npm install --silent)

say "Bootstrapping CDK in $REGION"
# Idempotent: re-bootstrapping an already-bootstrapped account is a no-op.
(cd "$CDK_DIR" && npx cdk bootstrap "aws://$ACCOUNT/$REGION" --require-approval never)

# ── 6. First pass ───────────────────────────────────────────────────────────────
say "Deploying all stacks (25-30 minutes, most of it Neptune)"
(cd "$CDK_DIR" && CDK_DEFAULT_REGION="$REGION" npx cdk deploy --all --require-approval never)

WEB_URL="$(aws cloudformation describe-stacks --stack-name LexGraphWeb --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='WebUrl'].OutputValue" --output text)"

[ -n "$WEB_URL" ] && [ "$WEB_URL" != "None" ] || die "LexGraphWeb produced no WebUrl output"
note "web       $WEB_URL"

# ── 7. Second pass, closing the circular requirement ────────────────────────────
#
# The Cognito hosted UI needs the CloudFront domain as a callback URL, and CloudFront does not exist
# until the first deploy. Two stacks read `webOrigin`, not one: Auth for the callback, and Data for
# the S3 CORS rule that lets the browser POST a file straight to the bucket. Deploying only Auth
# leaves uploads failing CORS, which looks like a broken upload button.
say "Setting webOrigin and redeploying Auth and Data"

set_context webOrigin "\"${WEB_URL%/}\""

(cd "$CDK_DIR" && CDK_DEFAULT_REGION="$REGION" \
  npx cdk deploy LexGraphAuth LexGraphData --require-approval never)

# ── 8. Verify ───────────────────────────────────────────────────────────────────
say "Checking the deployment"

health="$(curl -fsS -m 30 "$WEB_URL/api/health" 2>/dev/null || true)"
if [ -z "$health" ]; then
  note "health    no response yet; CloudFront can take a few minutes to serve"
else
  note "health    $health"
  # `graph: connected` is the field worth reading. A healthy container with a degraded graph means
  # Neptune is unreachable, which is almost always the TLS or SigV4 half of the Bolt handshake.
  grep -q '"graph":"connected"' <<<"$health" \
    || note "WARNING   the graph is not connected; check the LexGraphApp logs"
fi

POOL_ID="$(aws cognito-idp list-user-pools --max-results 20 --region "$REGION" \
  --query "UserPools[?starts_with(Name,'LexGraph')].Id | [0]" --output text 2>/dev/null || true)"

cat <<EOF

$(printf '\033[1m%s\033[0m' "Deployed.")

  Web       $WEB_URL
  Region    $REGION
  Account   $ACCOUNT
  Pool      ${POOL_ID:-unknown}

Nobody can sign in yet: the tenant a user belongs to is fixed at creation, so
self-service signup produces a user with no tenant who authenticates and is then
refused. Create the first user, which is the one step left to do by hand:

  POOL=${POOL_ID:-<user-pool-id>}
  EMAIL=you@example.com

  aws cognito-idp admin-create-user --region $REGION --user-pool-id "\$POOL" \\
    --username "\$EMAIL" \\
    --user-attributes Name=email,Value="\$EMAIL" Name=email_verified,Value=true \\
                      Name=custom:tenant_id,Value=$HOME_TENANT

  aws cognito-idp admin-add-user-to-group --region $REGION --user-pool-id "\$POOL" \\
    --username "\$EMAIL" --group-name platform-admin

Cognito emails a temporary password. $HOME_TENANT is the home tenant, whose admins may
create and delete other tenants from the Platform page.

EOF
