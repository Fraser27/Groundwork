import * as cdk from 'aws-cdk-lib';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as avp from 'aws-cdk-lib/aws-verifiedpermissions';
import { Construct } from 'constructs';

import { PROJECT, PROJECT_SLUG, tagStack } from './config';

export interface AuthStackProps extends cdk.StackProps {
  /** Callback origins for the hosted UI. The CloudFront domain is added by `web`. */
  readonly extraCallbackUrls?: string[];
}

/**
 * Cognito plus a Cedar policy store.
 *
 * The division of labour matters: Cognito answers *who* the caller is and, via a
 * custom claim, which tenant they belong to. Cedar answers *which matters* they
 * may read. Tenant comes from the verified token and is never a request
 * parameter, because `src/graph/scope.py` treats it as non-negotiable — a caller
 * able to name their own tenant could read another firm's privileged material.
 */
export class AuthStack extends cdk.Stack {
  readonly userPool: cognito.UserPool;
  readonly userPoolClient: cognito.UserPoolClient;
  readonly userPoolDomain: cognito.UserPoolDomain;
  readonly policyStore: avp.CfnPolicyStore;
  /** OIDC issuer the app and the MCP runtime both validate tokens against. */
  readonly issuerUrl: string;

  constructor(scope: Construct, id: string, props: AuthStackProps = {}) {
    super(scope, id, props);
    tagStack(this, 'auth');

    this.userPool = new cognito.UserPool(this, 'UserPool', {
      userPoolName: `${PROJECT}-users`,
      // Whitelabel SaaS: a firm's users are invited by their administrator, not
      // self-signed-up, or the tenant claim below would be self-asserted.
      selfSignUpEnabled: false,
      signInAliases: { email: true },
      autoVerify: { email: true },
      standardAttributes: { email: { required: true, mutable: false } },
      customAttributes: {
        // Written by an admin at invite time and copied into the JWT. Immutable
        // because a user who could change their own tenant would defeat the only
        // isolation boundary in the system.
        tenant_id: new cognito.StringAttribute({ minLen: 2, maxLen: 63, mutable: false }),
      },
      passwordPolicy: {
        minLength: 12,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      mfa: cognito.Mfa.OPTIONAL,
      mfaSecondFactor: { sms: false, otp: true },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      // ESSENTIALS, not PLUS. Threat protection — compromised-credential detection
      // and adaptive auth — requires PLUS, which roughly triples the per-MAU price.
      // Worth buying before real client data lands; not worth it for an empty
      // pool. Switching to PLUS plus `standardThreatProtectionMode` is the change.
      featurePlan: cognito.FeaturePlan.ESSENTIALS,
      // DESTROY for now. Before the first tenant this becomes RETAIN: deleting a
      // user pool orphans every tenant_id claim, and there is no import path back.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    this.userPoolDomain = this.userPool.addDomain('Domain', {
      cognitoDomain: { domainPrefix: `${PROJECT_SLUG}-${cdk.Aws.ACCOUNT_ID}` },
    });

    /**
     * Roles are Cognito groups, not a table.
     *
     * `src/auth.py` reads the `cognito:groups` claim straight off the verified JWT, so a
     * group membership *is* the authorization fact — there is no second store to keep in
     * sync and nothing to go stale between a revocation and the next request.
     *
     * Group names are load-bearing: they must match `Grants.is_platform_admin` and
     * `can_review` in `src/auth.py`. Renaming one here without renaming it there silently
     * removes everyone's access to whatever it gated.
     *
     * Membership is not granted here. A group with members in a template would put the
     * firm's admin list in git; see the README for the `admin-add-user-to-group` command.
     */
    for (const [name, description] of [
      ['platform-admin', 'Full administrative access: settings, sources, matter access'],
      ['reviewer', 'May approve or reject staged assertions'],
      ['matter-owner', 'May assign users to the matters they own'],
    ]) {
      new cognito.CfnUserPoolGroup(this, `Group${name.replace(/(^|-)(.)/g, (_, __, c) => c.toUpperCase())}`, {
        userPoolId: this.userPool.userPoolId,
        groupName: name,
        description,
      });
    }

    this.userPoolClient = this.userPool.addClient('WebClient', {
      userPoolClientName: `${PROJECT}-web`,
      // SRP only. No implicit grant: it puts tokens in the URL fragment, and
      // browser history is not somewhere a privileged-data token belongs.
      authFlows: { userSrp: true },
      oAuth: {
        flows: { authorizationCodeGrant: true },
        scopes: [cognito.OAuthScope.OPENID, cognito.OAuthScope.EMAIL, cognito.OAuthScope.PROFILE],
        callbackUrls: ['http://localhost:5173/', ...(props.extraCallbackUrls ?? [])],
        logoutUrls: ['http://localhost:5173/', ...(props.extraCallbackUrls ?? [])],
      },
      accessTokenValidity: cdk.Duration.hours(1),
      idTokenValidity: cdk.Duration.hours(1),
      refreshTokenValidity: cdk.Duration.days(30),
      preventUserExistenceErrors: true,
      readAttributes: new cognito.ClientAttributes()
        .withStandardAttributes({ email: true, emailVerified: true })
        .withCustomAttributes('tenant_id'),
      // Deliberately no write access to tenant_id — see the attribute comment.
      writeAttributes: new cognito.ClientAttributes(),
    });

    this.issuerUrl = `https://cognito-idp.${this.region}.amazonaws.com/${this.userPool.userPoolId}`;

    this.policyStore = this.buildPolicyStore();

    new avp.CfnIdentitySource(this, 'CognitoIdentitySource', {
      policyStoreId: this.policyStore.attrPolicyStoreId,
      principalEntityType: `${PROJECT}::User`,
      configuration: {
        cognitoUserPoolConfiguration: {
          userPoolArn: this.userPool.userPoolArn,
          clientIds: [this.userPoolClient.userPoolClientId],
          groupConfiguration: { groupEntityType: `${PROJECT}::Role` },
        },
      },
    });

    this.emitOutputs();
  }

  /**
   * Cedar schema for matter-level authorisation.
   *
   * STRICT validation is the point of using Cedar at all: a typo in a policy that
   * silently matches nothing is indistinguishable from a policy that correctly
   * denies, and "the ethical wall quietly stopped working" is the failure mode
   * this whole design exists to prevent.
   *
   * `forbid` overriding `permit` is a Cedar guarantee, which is why the
   * denylist-beats-allowlist rule in `src/graph/scope.py` mirrors it rather than
   * inventing its own precedence.
   */
  private buildPolicyStore(): avp.CfnPolicyStore {
    const schema = {
      [PROJECT]: {
        entityTypes: {
          User: {
            shape: {
              type: 'Record',
              attributes: {
                tenant_id: { type: 'String', required: true },
              },
            },
            memberOfTypes: ['Role'],
          },
          Role: { shape: { type: 'Record', attributes: {} } },
          Matter: {
            shape: {
              type: 'Record',
              attributes: {
                tenant_id: { type: 'String', required: true },
                // Ethical walls are expressed as a forbid over this, so staffing
                // changes are a policy edit rather than a data migration.
                walled: { type: 'Boolean', required: false },
              },
            },
          },
        },
        actions: {
          ReadMatter: { appliesTo: { principalTypes: ['User'], resourceTypes: ['Matter'] } },
          WriteAssertion: { appliesTo: { principalTypes: ['User'], resourceTypes: ['Matter'] } },
          ReviewAssertion: { appliesTo: { principalTypes: ['User'], resourceTypes: ['Matter'] } },
          RunQuery: { appliesTo: { principalTypes: ['User'], resourceTypes: ['Matter'] } },
        },
      },
    };

    const store = new avp.CfnPolicyStore(this, 'PolicyStore', {
      description: 'Groundwork matter-level authorisation',
      validationSettings: { mode: 'STRICT' },
      schema: { cedarJson: JSON.stringify(schema) },
      deletionProtection: { mode: 'DISABLED' },
    });

    // A tenant boundary check in Cedar as well as in Cypher. Redundant on
    // purpose: `scope.py` is the enforcement, this is the second pair of eyes,
    // and neither depends on the other being correct.
    new avp.CfnPolicy(this, 'SameTenantPolicy', {
      policyStoreId: store.attrPolicyStoreId,
      definition: {
        static: {
          description: 'Cross-tenant access is never permitted',
          statement: `forbid (
  principal,
  action,
  resource
) unless {
  principal has tenant_id &&
  resource has tenant_id &&
  principal.tenant_id == resource.tenant_id
};`,
        },
      },
    });

    return store;
  }

  private emitOutputs(): void {
    new cdk.CfnOutput(this, 'UserPoolId', { value: this.userPool.userPoolId });
    new cdk.CfnOutput(this, 'UserPoolClientId', { value: this.userPoolClient.userPoolClientId });
    new cdk.CfnOutput(this, 'CognitoIssuerUrl', { value: this.issuerUrl });
    new cdk.CfnOutput(this, 'CognitoDiscoveryUrl', {
      value: `${this.issuerUrl}/.well-known/openid-configuration`,
      description: 'AgentCore JWT authorizer discovery URL',
    });
    new cdk.CfnOutput(this, 'HostedUiDomain', {
      value: `https://${this.userPoolDomain.domainName}.auth.${this.region}.amazoncognito.com`,
    });
    new cdk.CfnOutput(this, 'PolicyStoreId', { value: this.policyStore.attrPolicyStoreId });
  }
}
