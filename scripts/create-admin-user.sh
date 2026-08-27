#!/usr/bin/env bash
#
# Create a Cognito user for a deployed Groundwork stack and bind them to a tenant.
#
# This exists because of a gap in the manual first-user instructions `deploy.sh` used to
# print: `admin-create-user` sets the `custom:tenant_id` attribute on the Cognito user, but
# the API never reads that attribute at request time -- it reads only ACCESS tokens
# (src/auth.py, TokenVerifier), and Cognito puts custom attributes on the ID token only.
# So the API falls back to a DynamoDB lookup (TenantDirectory / TenantTable) to resolve a
# user's tenant, and a user created by the raw CLI commands alone has no row there. They
# sign in successfully, hold a token with the right role, and then get 401 on the very
# first tenant-scoped request ("user is not bound to a tenant") -- which the UI treats as
# an expired session and logs them straight back out.
#
# This script does both halves in one place: creates (or reuses) the Cognito user, adds
# them to a role group, and writes the matching row into TenantTable so the API can
# actually resolve them.
#
#   ./scripts/create-admin-user.sh you@example.com
#   ./scripts/create-admin-user.sh you@example.com reviewer,matter-owner
#   REGION=eu-west-1 HOME_TENANT=acme ./scripts/create-admin-user.sh you@example.com
#
# Non-interactive: everything but the email has a default, and nothing supplied is ever
# confirmed back. The email is prompted for only when there is no argument *and* a
# terminal to prompt on, so this is safe to run from nohup, SSM or user data.
#
# Non-idempotent by design where it matters: re-running for an existing email reuses the
# Cognito user (rather than failing) and still (re)writes the tenant binding, since that is
# exactly the repair this script exists to make possible.

set -euo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }
die() { printf '\n\033[31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }
usage() { printf 'usage: %s <email> [role[,role...]]\n' "$0" >&2; exit 1; }

# One trap, registered once: a second `trap ... EXIT` later would silently replace this
# one and leak whatever the first was cleaning up.
TMP_DIR=""
CREATE_ERR=""
cleanup() {
  [ -n "$TMP_DIR" ] && rm -rf "$TMP_DIR"
  [ -n "$CREATE_ERR" ] && rm -f "$CREATE_ERR"
  return 0
}
trap cleanup EXIT

# ── 1. AWS CLI present, or install it ───────────────────────────────────────────
say "Checking for the AWS CLI"

if command -v aws >/dev/null 2>&1; then
  note "found     $(aws --version 2>&1)"
else
  note "not found -- installing"
  case "$(uname -s)" in
    Darwin)
      if command -v brew >/dev/null 2>&1; then
        brew install awscli
      else
        # Official Apple Silicon / Intel universal installer. Needs sudo, since it
        # installs into /usr/local; there is no Homebrew-free way around that.
        TMP_DIR="$(mktemp -d)"
        curl -fsSL -o "$TMP_DIR/AWSCLIV2.pkg" "https://awscli.amazonaws.com/AWSCLIV2.pkg" \
          || die "could not download the AWS CLI installer"
        sudo installer -pkg "$TMP_DIR/AWSCLIV2.pkg" -target / \
          || die "AWS CLI install failed"
      fi
      ;;
    Linux)
      # unzip is not on a bare Ubuntu or minimal AL2023 image, and the installer is a zip.
      # Worth installing here rather than failing: this is the one path every participant hits.
      if ! command -v unzip >/dev/null 2>&1; then
        note "unzip missing -- installing it first"
        if command -v dnf >/dev/null 2>&1; then sudo dnf install -y unzip
        elif command -v yum >/dev/null 2>&1; then sudo yum install -y unzip
        elif command -v apt-get >/dev/null 2>&1; then sudo apt-get update -qq && sudo apt-get install -y unzip
        else die "no unzip and no package manager I recognise. Install unzip, then re-run."
        fi
        command -v unzip >/dev/null 2>&1 || die "could not install unzip"
      fi
      command -v curl >/dev/null 2>&1 || die "curl is required to download the AWS CLI installer"
      TMP_DIR="$(mktemp -d)"
      curl -fsSL -o "$TMP_DIR/awscliv2.zip" \
        "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" \
        || die "could not download the AWS CLI installer"
      # Without an explicit die, set -e would abort here with no message at all.
      (cd "$TMP_DIR" && unzip -q awscliv2.zip && sudo ./aws/install) \
        || die "AWS CLI install failed. Check that this user has sudo."
      ;;
    *)
      die "don't know how to install the AWS CLI on $(uname -s). Install it yourself: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
      ;;
  esac
  command -v aws >/dev/null 2>&1 || die "AWS CLI installation did not put 'aws' on PATH"
  note "installed $(aws --version 2>&1)"
fi

# ── 2. Parameters ────────────────────────────────────────────────────────────────
say "User details"

EMAIL="${1:-}"
if [ -z "$EMAIL" ]; then
  # The one value with no defensible default. Prompted only if there is a terminal:
  # under nohup or SSM there is not, and blocking on a read nobody can answer is worse
  # than saying what the argument should have been.
  [ -t 0 ] || usage
  read -r -p "Email address: " EMAIL || true
fi
[[ "$EMAIL" =~ ^[^[:space:]@]+@[^[:space:]@]+\.[^[:space:]@]+$ ]] \
  || die "'$EMAIL' does not look like an email address"

# AWS_REGION before AWS_DEFAULT_REGION, matching the CLI's own precedence. CloudShell sets
# only the former, from the console's Region selector, so reading just the latter would
# fall back to us-east-1 for someone who opened CloudShell somewhere else entirely.
REGION="${REGION:-${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}}"

# Matches the `homeTenant` default in cdk/lib/config.ts. A user bound to any other tenant
# cannot reach the Platform page, so getting this wrong looks like missing permissions.
TENANT_ID="${TENANT_ID:-${HOME_TENANT:-demo-firm}}"

# Defaults to platform-admin: this creates the *first* user, and a first user who can
# administer nothing leaves the deployment with no way in. Every argument after the email
# is taken, and comma or space both separate, so no form silently drops a role.
ROLES_RAW="${*:2}"
ROLE_GROUPS=()
IFS=', ' read -ra _requested <<<"${ROLES_RAW:-${ROLES:-platform-admin}}"
for r in "${_requested[@]}"; do
  case "$r" in
    platform-admin | matter-owner | reviewer) ROLE_GROUPS+=("$r") ;;
    "") ;;
    *) die "unknown role '$r'. Choose from: platform-admin, matter-owner, reviewer" ;;
  esac
done
[ "${#ROLE_GROUPS[@]}" -gt 0 ] || die "at least one role is required"

note "email     $EMAIL"
note "region    $REGION"
note "tenant    $TENANT_ID"
note "roles     ${ROLE_GROUPS[*]}"

# ── 3. Credentials ───────────────────────────────────────────────────────────────
say "Checking AWS credentials"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)" \
  || die "No usable AWS credentials. Run 'aws configure' or export a profile."
note "account   $ACCOUNT"

# ── 4. Discover the deployed stacks ──────────────────────────────────────────────
say "Reading the deployed stacks"

# Named to match cdk/bin/app.ts's stack prefix. If a deployment used a different prefix
# (e.g. an older deploy from before the LexGraph -> Groundwork rename), pass the two
# names as env vars instead:
#   AUTH_STACK=LexGraphAuth DATA_STACK=LexGraphData ./scripts/create-admin-user.sh
AUTH_STACK="${AUTH_STACK:-GroundworkAuth}"
DATA_STACK="${DATA_STACK:-GroundworkData}"

USER_POOL_ID="$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$AUTH_STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='UserPoolId'].OutputValue | [0]" --output text 2>/dev/null || true)"
[ -n "$USER_POOL_ID" ] && [ "$USER_POOL_ID" != "None" ] \
  || die "could not read UserPoolId from stack '$AUTH_STACK' in $REGION. Deploy the CDK app first, or set AUTH_STACK to the right stack name."
note "user pool $USER_POOL_ID"

# TenantTable has no CfnOutput (app-stack.ts wires it by construct reference, not by
# export), so it is found the same way agentcore/deploy_agent.py finds untagged
# resources: by resource type and logical-id prefix within the stack.
TENANT_TABLE="$(aws cloudformation list-stack-resources --region "$REGION" --stack-name "$DATA_STACK" \
  --query "StackResourceSummaries[?ResourceType=='AWS::DynamoDB::Table' && starts_with(LogicalResourceId,'TenantTable')].PhysicalResourceId | [0]" \
  --output text 2>/dev/null || true)"
[ -n "$TENANT_TABLE" ] && [ "$TENANT_TABLE" != "None" ] \
  || die "could not find the tenant table in stack '$DATA_STACK' in $REGION. Deploy the CDK app first, or set DATA_STACK to the right stack name."
note "tenant table $TENANT_TABLE"

# ── 5. Create (or reuse) the Cognito user ────────────────────────────────────────
say "Creating the Cognito user"

CREATE_ERR="$(mktemp)"

if aws cognito-idp admin-create-user --region "$REGION" --user-pool-id "$USER_POOL_ID" \
    --username "$EMAIL" \
    --user-attributes Name=email,Value="$EMAIL" Name=email_verified,Value=true \
                      Name=custom:tenant_id,Value="$TENANT_ID" \
    >/dev/null 2>"$CREATE_ERR"; then
  note "created   $EMAIL"
else
  if grep -q UsernameExistsException "$CREATE_ERR"; then
    note "exists    $EMAIL (reusing; not touching their password)"
  else
    die "could not create $EMAIL: $(cat "$CREATE_ERR")"
  fi
fi

SUB="$(aws cognito-idp admin-get-user --region "$REGION" --user-pool-id "$USER_POOL_ID" \
  --username "$EMAIL" --query "UserAttributes[?Name=='sub'].Value | [0]" --output text)"
[ -n "$SUB" ] && [ "$SUB" != "None" ] || die "created the user but could not read back their sub"
note "sub       $SUB"

# ── 6. Group membership (the role) ───────────────────────────────────────────────
say "Assigning roles"

for group in "${ROLE_GROUPS[@]}"; do
  aws cognito-idp admin-add-user-to-group --region "$REGION" --user-pool-id "$USER_POOL_ID" \
    --username "$EMAIL" --group-name "$group" \
    || die "could not add $EMAIL to group $group"
  note "group     $group"
done

# ── 7. The tenant binding the API actually reads ─────────────────────────────────
say "Writing the tenant binding"

# Mirrors src/tenant_directory.py: partition key is USER#{sub}, and the tenant lives in
# an attribute named "tenant" (not "tenant_id" -- that name is owned by the table's own
# key). This is the step the manual `admin-create-user` instructions were missing.
EMAIL_LOWER="$(echo "$EMAIL" | tr '[:upper:]' '[:lower:]')"

aws dynamodb put-item --region "$REGION" --table-name "$TENANT_TABLE" \
  --item "{
    \"tenant_id\": {\"S\": \"USER#${SUB}\"},
    \"sub\": {\"S\": \"${SUB}\"},
    \"tenant\": {\"S\": \"${TENANT_ID}\"},
    \"email\": {\"S\": \"${EMAIL_LOWER}\"}
  }" || die "could not write the tenant binding to $TENANT_TABLE"
note "bound     $EMAIL -> tenant $TENANT_ID"

cat <<EOF

$(printf '\033[1m%s\033[0m' "Done.")

  Email     $EMAIL
  Tenant    $TENANT_ID
  Roles     ${ROLE_GROUPS[*]}
  Sub       $SUB

If this was a new user, Cognito has emailed $EMAIL a temporary password; they will be
forced to change it at first sign-in. If it already existed, their password is
unchanged and they can sign in as before -- they were just missing the tenant binding
that lets /api requests resolve past the login page.

EOF
