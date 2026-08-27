"""Deploy the Groundwork MCP server to Bedrock AgentCore Runtime.

    python agentcore/deploy_agent.py --plan       # print what would change, call nothing
    python agentcore/deploy_agent.py              # create or update the runtime
    python agentcore/deploy_agent.py --cleanup    # delete the runtime and its endpoint

`cdk deploy GroundworkMcp` does the same thing and is the supported path. This script
exists for the case CloudFormation is bad at: iterating on the runtime alone against an
already-deployed stack, where a `cdk deploy` is a five-minute changeset for a container
image swap. It reads its configuration from the deployed stacks rather than taking
arguments, so it cannot drift from what CDK built — and it deliberately does not create
IAM roles or Cognito clients. Those belong to `app` and `auth`, and a script that also
created them would produce a second set nothing else knows about.

The security shape it must preserve, and the reason this file has any comments at all:

- **The runtime runs as the API's task role.** Not a role of its own. An MCP tool must
  not be able to do anything the REST API cannot.
- **Inbound auth is the same Cognito pool and the same client the UI uses.** A tool call
  therefore arrives with a token AgentCore has already verified, carrying the real user's
  `tenant_id` claim, and `src/mcp/auth.py` scopes to it. There is no machine-to-machine
  client here on purpose: an M2M token has no user, so a tool call carrying one could not
  be scoped to a tenant and would have to invent one.
- **Same image as the API,** entered at `src.mcp.server:app`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("deploy")

REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"

NETWORK_STACK = "GroundworkNetwork"
AUTH_STACK = "GroundworkAuth"
APP_STACK = "GroundworkApp"

RUNTIME_NAME = "Groundwork_mcp"
ENDPOINT_NAME = "live"

#: What `Dockerfile` switches on. The only thing that differs from the API container.
MCP_APP_MODULE = "src.mcp.server:app"

#: AgentCore's MCP contract fixes the container port. Mirrors `src/constants.py`.
APP_PORT = "8000"

#: Terminal states. Anything else means creation or update is still in flight.
_SETTLED = frozenset({"READY", "CREATE_FAILED", "UPDATE_FAILED", "DELETE_FAILED"})


class DeployError(RuntimeError):
    pass


# ── discovery ────────────────────────────────────────────────────────────────────


def _stack_outputs(cfn: Any, stack: str) -> dict[str, str]:
    try:
        described = cfn.describe_stacks(StackName=stack)["Stacks"][0]
    except ClientError as e:
        raise DeployError(
            f"stack {stack} not found in {REGION}. Deploy the CDK app first: "
            "`cd cdk && npx cdk deploy --all`."
        ) from e
    return {o["OutputKey"]: o["OutputValue"] for o in described.get("Outputs", [])}


def _stack_resources(cfn: Any, stack: str) -> list[dict[str, str]]:
    """The stack's resources, for the ones CDK does not publish as outputs."""
    out: list[dict[str, str]] = []
    paginator = cfn.get_paginator("list_stack_resources")
    for page in paginator.paginate(StackName=stack):
        out.extend(page["StackResourceSummaries"])
    return out


def _find(resources: list[dict[str, str]], resource_type: str, prefix: str, kind: str) -> str:
    """Physical id of the one resource of `resource_type` whose logical id starts with `prefix`.

    Type *and* prefix, because neither alone is unique. CDK appends a hash to logical ids so
    an exact id cannot be hardcoded; and the prefix alone is ambiguous — `TaskDef` also
    matches `TaskDefExecutionRole`, `Vpc` matches every subnet and route table.

    Fails loudly on more than one match rather than taking the first: silently picking the
    wrong security group yields a runtime that starts and then cannot reach Neptune, which
    is a far worse afternoon than an error here.
    """
    hits = [
        r["PhysicalResourceId"]
        for r in resources
        if r["ResourceType"] == resource_type and r["LogicalResourceId"].startswith(prefix)
    ]
    if not hits:
        raise DeployError(f"no {kind} ({resource_type} matching {prefix!r}) in the stack")
    if len(hits) > 1:
        raise DeployError(f"{len(hits)} candidates for {kind} ({prefix!r}): {hits}")
    return hits[0]


def discover() -> dict[str, Any]:
    """Read everything the runtime needs from the deployed stacks.

    Deliberately not parameterised. Every value here has exactly one correct source, and a
    flag would let someone point the runtime at a different VPC or a different user pool
    than the API is using — which for the user pool means a token this runtime accepts that
    the API would reject.
    """
    cfn = boto3.client("cloudformation", region_name=REGION)
    ec2 = boto3.client("ec2", region_name=REGION)

    auth = _stack_outputs(cfn, AUTH_STACK)
    app_outputs = _stack_outputs(cfn, APP_STACK)
    app_resources = _stack_resources(cfn, APP_STACK)
    network_resources = _stack_resources(cfn, NETWORK_STACK)

    vpc_id = _find(network_resources, "AWS::EC2::VPC", "Vpc", "VPC")
    security_group = _find(
        network_resources, "AWS::EC2::SecurityGroup", "AppSg", "app security group"
    )

    # PRIVATE_WITH_EGRESS: the runtime needs NAT for Bedrock, and must sit in the same
    # subnets as the API so it reaches Neptune by the rules the API already has.
    subnets = [
        s["SubnetId"]
        for s in ec2.describe_subnets(
            Filters=[
                {"Name": "vpc-id", "Values": [vpc_id]},
                {"Name": "tag:aws-cdk:subnet-name", "Values": ["private"]},
            ]
        )["Subnets"]
    ]
    if not subnets:
        raise DeployError(f"no private subnets tagged by CDK found in {vpc_id}")

    config = {
        "image_uri": app_outputs["AppImageUri"],
        # The API's role, not a new one. See the module docstring.
        "role_arn": app_outputs["TaskRoleArn"],
        "subnets": sorted(subnets),
        "security_groups": [security_group],
        "discovery_url": auth["CognitoDiscoveryUrl"],
        "client_id": auth["UserPoolClientId"],
        "environment": _environment(app_resources, auth),
    }
    logger.info("image     %s", config["image_uri"])
    logger.info("role      %s", config["role_arn"])
    logger.info("subnets   %s", ", ".join(config["subnets"]))
    logger.info("authorizer %s (client %s)", config["discovery_url"], config["client_id"])
    return config


def _environment(
    app_resources: list[dict[str, str]], auth: dict[str, str]
) -> dict[str, str]:
    """Reconstruct the API container's environment, with APP_MODULE swapped.

    Read back from the deployed task definition rather than rebuilt from stack outputs:
    `app-stack.ts` owns this list, and a copy here would go stale the first time someone
    adds a variable. AUTH_DEV_BYPASS_TENANT is stripped defensively — `src/config.py`
    already refuses to start with it set outside local, and this makes it impossible to
    carry across by accident.
    """
    ecs = boto3.client("ecs", region_name=REGION)
    task_def_arn = _find(
        app_resources, "AWS::ECS::TaskDefinition", "TaskDef", "API task definition"
    )
    task_def = ecs.describe_task_definition(taskDefinition=task_def_arn)["taskDefinition"]

    # The image this script deploys is the one the task runs, so the task's architecture is
    # the image's. AgentCore Runtime takes ARM64 only, and it reports a mismatch as a
    # CREATE_FAILED minutes later rather than rejecting the call, so it is checked here.
    arch = task_def.get("runtimePlatform", {}).get("cpuArchitecture", "X86_64")
    if arch != "ARM64":
        raise DeployError(
            f"{APP_STACK} was built for {arch}, and AgentCore Runtime accepts ARM64 only. "
            "Set agentCoreMcp to true in cdk/cdk.json and redeploy GroundworkApp first."
        )

    container = task_def["containerDefinitions"][0]

    env = {e["name"]: e["value"] for e in container.get("environment", [])}
    env.pop("AUTH_DEV_BYPASS_TENANT", None)
    env.update(
        {
            "APP_MODULE": MCP_APP_MODULE,
            "PORT": APP_PORT,
            "ENVIRONMENT": env.get("ENVIRONMENT", "production"),
            # Belt and braces with the values already read from the task definition: if the
            # API's environment ever loses these, a runtime without them would start and
            # then accept unverified tokens.
            "COGNITO_ISSUER_URL": env.get("COGNITO_ISSUER_URL") or auth["CognitoIssuerUrl"],
            "COGNITO_CLIENT_ID": env.get("COGNITO_CLIENT_ID") or auth["UserPoolClientId"],
        }
    )
    if not env["COGNITO_ISSUER_URL"]:
        raise DeployError("COGNITO_ISSUER_URL is empty; the runtime would not verify tokens")
    return env


# ── runtime ──────────────────────────────────────────────────────────────────────


def _runtime_payload(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "agentRuntimeArtifact": {"containerConfiguration": {"containerUri": config["image_uri"]}},
        "roleArn": config["role_arn"],
        "networkConfiguration": {
            "networkMode": "VPC",
            # No vpcId: AgentCore derives it from the subnets, which is also why an
            # unsupported AZ fails with an error naming the subnet rather than the zone.
            # See SUPPORTED_AZ_IDS in cdk/lib/network-stack.ts.
            "networkModeConfig": {
                "subnets": config["subnets"],
                "securityGroups": config["security_groups"],
            },
        },
        # MCP, not HTTP: the runtime then speaks the streamable-HTTP MCP transport and any
        # MCP client connects without a shim.
        "protocolConfiguration": {"serverProtocol": "MCP"},
        "authorizerConfiguration": {
            "customJWTAuthorizer": {
                "discoveryUrl": config["discovery_url"],
                "allowedClients": [config["client_id"]],
            }
        },
        "environmentVariables": config["environment"],
    }


def find_runtime(client: Any) -> dict[str, Any] | None:
    paginator = client.get_paginator("list_agent_runtimes")
    for page in paginator.paginate():
        for runtime in page.get("agentRuntimes", []):
            if runtime.get("agentRuntimeName") == RUNTIME_NAME:
                return runtime
    return None


def upsert_runtime(client: Any, config: dict[str, Any]) -> str:
    payload = _runtime_payload(config)
    existing = find_runtime(client)

    if existing is None:
        logger.info("creating runtime %s", RUNTIME_NAME)
        response = client.create_agent_runtime(
            agentRuntimeName=RUNTIME_NAME,
            description="Groundwork governed semantic layer — MCP tools",
            **payload,
        )
    else:
        runtime_id = existing["agentRuntimeId"]
        logger.info("updating runtime %s (%s)", RUNTIME_NAME, runtime_id)
        response = client.update_agent_runtime(agentRuntimeId=runtime_id, **payload)

    runtime_id = response["agentRuntimeId"]
    logger.info("version %s, status %s", response.get("agentRuntimeVersion"), response["status"])
    logger.info("arn %s", response["agentRuntimeArn"])
    return runtime_id


def wait_until_settled(client: Any, runtime_id: str, *, timeout: int = 600) -> str:
    """Poll to a terminal state and report `failureReason` on failure.

    Worth the code: a failed create leaves a runtime whose status alone says nothing, and
    the reason string is where the real answer lives — almost always an unsupported AZ or
    a role the runtime cannot assume.
    """
    deadline = time.monotonic() + timeout
    status = "UNKNOWN"
    while time.monotonic() < deadline:
        described = client.get_agent_runtime(agentRuntimeId=runtime_id)
        status = described["status"]
        if status in _SETTLED:
            if status != "READY":
                raise DeployError(
                    f"runtime {status}: {described.get('failureReason') or 'no reason given'}"
                )
            return status
        logger.info("  %s ...", status)
        time.sleep(10)
    raise DeployError(f"runtime did not settle within {timeout}s (last status {status})")


def upsert_endpoint(client: Any, runtime_id: str) -> str:
    """A named endpoint clients point at, so versions can move beneath it."""
    existing = {
        e["name"]: e
        for e in client.list_agent_runtime_endpoints(agentRuntimeId=runtime_id).get(
            "runtimeEndpoints", []
        )
    }
    if ENDPOINT_NAME in existing:
        logger.info("endpoint %s exists; pointing it at the current version", ENDPOINT_NAME)
        response = client.update_agent_runtime_endpoint(
            agentRuntimeId=runtime_id, endpointName=ENDPOINT_NAME
        )
    else:
        logger.info("creating endpoint %s", ENDPOINT_NAME)
        response = client.create_agent_runtime_endpoint(
            agentRuntimeId=runtime_id,
            name=ENDPOINT_NAME,
            description="Stable endpoint — clients point here, versions move beneath it",
        )
    return response["agentRuntimeEndpointArn"]


def cleanup(client: Any) -> None:
    """Delete the endpoint then the runtime. Deletes nothing CDK owns."""
    existing = find_runtime(client)
    if existing is None:
        logger.info("no runtime named %s", RUNTIME_NAME)
        return

    runtime_id = existing["agentRuntimeId"]
    for endpoint in client.list_agent_runtime_endpoints(agentRuntimeId=runtime_id).get(
        "runtimeEndpoints", []
    ):
        logger.info("deleting endpoint %s", endpoint["name"])
        client.delete_agent_runtime_endpoint(
            agentRuntimeId=runtime_id, endpointName=endpoint["name"]
        )

    logger.info("deleting runtime %s", runtime_id)
    client.delete_agent_runtime(agentRuntimeId=runtime_id)
    logger.info(
        "done. The image, role, user pool and VPC belong to the CDK stacks and are untouched."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the Groundwork MCP server to AgentCore")
    parser.add_argument(
        "--plan", action="store_true", help="print the payload and exit without calling AgentCore"
    )
    parser.add_argument("--cleanup", action="store_true", help="delete the runtime and endpoint")
    parser.add_argument(
        "--no-wait", action="store_true", help="return before the runtime reaches READY"
    )
    args = parser.parse_args()

    client = boto3.client("bedrock-agentcore-control", region_name=REGION)

    try:
        if args.cleanup:
            cleanup(client)
            return 0

        config = discover()
        if args.plan:
            print(json.dumps(_runtime_payload(config), indent=2, sort_keys=True))
            return 0

        runtime_id = upsert_runtime(client, config)
        if not args.no_wait:
            wait_until_settled(client, runtime_id)
        endpoint_arn = upsert_endpoint(client, runtime_id)

        logger.info("endpoint %s", endpoint_arn)
        logger.info(
            "clients connect with a Cognito access token for client %s — the same token the "
            "UI uses, so a tool call is scoped to the user who made it",
            config["client_id"],
        )
        return 0
    except DeployError as e:
        logger.error("%s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
