# Environment variables - Lightdash

**Source:** https://docs.lightdash.com/self-host/customize-deployment/environment-variables

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Customize deployment
Environment variables
##### Introduction
  * Welcome to Lightdash
  * Setting up a new project


##### Explore and analyze
  * Quickstart
  * Guides
  * Reference


##### Build your semantic layer
  * Developer quickstart
  * CLI quickstart
  * Guides
  * Reference


##### Workspace and user management
  * Guides
  * Reference


##### Integrations


##### Embedding and SDKs
  * Embedding quickstart
  * Embedding reference


##### Self-hosting and deployment
  * Self-Host Lightdash
  * Lightdash Cloud vs. Self-Hosted
  * Updating Lightdash
  * Customize deployment
    * Standalone scheduler worker
    * Environment variables
    * Resource recommendations
    * Secure Lightdash with HTTPS


##### Contact


On this page
  * Analytics & Event Tracking
  * OpenAI Configuration
  * Anthropic Configuration
  * Azure AI Configuration
  * OpenRouter Configuration
  * Slack Integration
  * GitHub Integration
  * Microsoft Teams Integration
  * Google Cloud Platform
  * Service account
  * Intercom & Pylon
  * Organization appearance
  * Initialize instance
  * Update instance


This is a reference to all environment variables that can be used to configure a Lightdash deployment. Variable | Description | Required? | Default  
---|---|---|---  
`PGHOST` | Hostname of postgres server to store Lightdash data  
`PGPORT` | Port of postgres server to store Lightdash data  
`PGUSER` | Username of postgres user to access postgres server to store Lightdash data  
`PGPASSWORD` | Password for PGUSER  
`PGDATABASE` | Database name inside postgres server to store Lightdash data  
`PGCONNECTIONURI` | Connection URI for postgres server to store Lightdash data in the format postgresql://user:password@host:port/db?params. This is an alternative to providing the previous PG variables.  
`PGMAXCONNECTIONS` | Maximum number of connections to the database  
`PGMINCONNECTIONS` | Minimum number of connections to the database  
`LIGHTDASH_SECRET` | Secret key used to secure various tokens in Lightdash. This **must** be fixed between deployments. If the secret changes, you won’t have access to Lightdash data.  
`SECURE_COOKIES` | Only allows cookies to be stored over a https connection. We use cookies to keep you logged in. This is recommended to be set to true in production. | `false`  
`COOKIES_MAX_AGE_HOURS` | How many hours a user session exists before the user is automatically signed out. For example if 24, then the user will be automatically after 24 hours of inactivity.  
`TRUST_PROXY` | This tells the Lightdash server that it can trust the X-Forwarded-Proto header it receives in requests. This is useful if you use `SECURE_COOKIES=true` behind a HTTPS terminated proxy that you can trust. | `false`  
`SITE_URL` | Site url where Lightdash is being hosted. It should include the protocol. E.g https://lightdash.mycompany.com | `http://localhost:8080`  
`INTERNAL_LIGHTDASH_HOST` | Internal Lightdash host for the Headless browser to send requests when your Lightdash instance is not accessible from the Internet. Needs to support `https` if `SECURE_COOKIES=true` | Same as `SITE_URL`  
`STATIC_IP` | Server static IP so users can add the IP to their warehouse allow-list. | `http://localhost:8080`  
`LIGHTDASH_QUERY_MAX_LIMIT` | Query max rows limit | `5000`  
`LIGHTDASH_QUERY_DEFAULT_LIMIT` | Default number of rows to return in a query | `500`  
`LIGHTDASH_QUERY_MAX_PAGE_SIZE` | Maximum page size for paginated queries | `2500`  
`SCHEDULER_ENABLED` | Enables/Disables the scheduler worker that triggers the scheduled deliveries. | `true`  
`SCHEDULER_CONCURRENCY` | How many scheduled delivery jobs can be processed concurrently.  
`SCHEDULER_JOB_TIMEOUT` | After how many milliseconds the job should be timeout so the scheduler worker can pick other jobs. |  `600000` (10 minutes)  
`SCHEDULER_SCREENSHOT_TIMEOUT` | Timeout in milliseconds for taking screenshots  
`SCHEDULER_INCLUDE_TASKS` | Comma-separated list of scheduler tasks to include  
`SCHEDULER_EXCLUDE_TASKS` | Comma-separated list of scheduler tasks to exclude  
`LIGHTDASH_CSV_CELLS_LIMIT` | Max cells on CSV file exports | `100000`  
`LIGHTDASH_CHART_VERSION_HISTORY_DAYS_LIMIT` | Configure how far back the chart versions history goes in days  
`LIGHTDASH_PIVOT_TABLE_MAX_COLUMN_LIMIT` | Configure maximum number of columns in pivot table  
`GROUPS_ENABLED` | Enables/Disables groups functionality | `false`  
`AUTH_ENABLE_OIDC_LINKING` | Enables/Disables linking the new OIDC(aka SSO) identity to an existing user if they already have another OIDC with the same email | `false`  
`AUTH_ENABLE_OIDC_TO_EMAIL_LINKING` | Enables/Disables linking OIDC identity to an existing user by email | `false`  
`CUSTOM_VISUALIZATIONS_ENABLED` | Enables/Disables custom chart functionality | `false`  
`LIGHTDASH_MAX_PAYLOAD` | Maximum HTTP request body size | `5mb`  
`LIGHTDASH_LICENSE_KEY` | License key for Lightdash Enterprise Edition. Talk to us about Lightdash Enterprise Edition  
`HEADLESS_BROWSER_HOST` | Hostname for the headless browser | —  
`HEADLESS_BROWSER_PORT` | Port for the headless browser | `3001`  
`ALLOW_MULTIPLE_ORGS` | If set to `true`, new users registering on Lightdash will have their own organization, separated from others | `false`  
`LIGHTDASH_MODE` | Mode for Lightdash (default, demo, pr, etc.) | `default`  
`DISABLE_PAT` | Disables Personal Access Tokens | `false`  
`PAT_ALLOWED_ORG_ROLES` | Comma-separated list of organization roles allowed to use Personal Access Tokens | All roles  
`PAT_MAX_EXPIRATION_TIME_IN_DAYS` | Maximum expiration time in days for Personal Access Tokens  
`MAX_DOWNLOADS_AS_CODE` | Maximum number of downloads as code | `100`  
`EXTENDED_USAGE_ANALYTICS` | Enables extended usage analytics | `false`  
`USE_SECURE_BROWSER` | Use secure WebSocket connections for headless browser | `false`  
Lightdash also accepts all standard postgres environment variables
##  SMTP
This is a reference to all the SMTP environment variables that can be used to configure a Lightdash email client. Variable | Description | Required? | Default  
---|---|---|---  
`EMAIL_SMTP_HOST` | Hostname of email server  
`EMAIL_SMTP_PORT` | Port of email server | `587`  
`EMAIL_SMTP_SECURE` | Secure connection | `true`  
`EMAIL_SMTP_USER` | Auth user  
`EMAIL_SMTP_PASSWORD` | Auth password | `[1]`  
`EMAIL_SMTP_ACCESS_TOKEN` | Auth access token for Oauth2 authentication | `[1]`  
`EMAIL_SMTP_ALLOW_INVALID_CERT` | Allow connection to TLS server with self-signed or invalid TLS certificate | `false`  
`EMAIL_SMTP_SENDER_EMAIL` | The email address that sends emails  
`EMAIL_SMTP_SENDER_NAME` | The name of the email address that sends emails | `Lightdash`  
[1] `EMAIL_SMTP_PASSWORD` or `EMAIL_SMTP_ACCESS_TOKEN` needs to be provided
##  SSO
These variables enable you to control Single Sign On (SSO) functionality. Variable | Description | Required? | Default  
---|---|---|---  
`AUTH_DISABLE_PASSWORD_AUTHENTICATION` | If “true” disables signing in with plain passwords | `false`  
`AUTH_ENABLE_GROUP_SYNC` | If “true” enables assigning SSO groups to Lightdash groups | `false`  
`AUTH_GOOGLE_OAUTH2_CLIENT_ID` | Required for Google SSO  
`AUTH_GOOGLE_OAUTH2_CLIENT_SECRET` | Required for Google SSO  
`AUTH_OKTA_OAUTH_CLIENT_ID` | Required for Okta SSO  
`AUTH_OKTA_OAUTH_CLIENT_SECRET` | Required for Okta SSO  
`AUTH_OKTA_OAUTH_ISSUER` | Required for Okta SSO  
`AUTH_OKTA_DOMAIN` | Required for Okta SSO  
`AUTH_OKTA_AUTHORIZATION_SERVER_ID` | Optional for Okta SSO with a custom authorization server  
`AUTH_OKTA_EXTRA_SCOPES` | Optional for Okta SSO scopes (e.g. groups) without a custom authorization server  
`AUTH_ONE_LOGIN_OAUTH_CLIENT_ID` | Required for One Login SSO  
`AUTH_ONE_LOGIN_OAUTH_CLIENT_SECRET` | Required for One Login SSO  
`AUTH_ONE_LOGIN_OAUTH_ISSUER` | Required for One Login SSO  
`AUTH_AZURE_AD_OAUTH_CLIENT_ID` | Required for Azure AD  
`AUTH_AZURE_AD_OAUTH_CLIENT_SECRET` | Required for Azure AD  
`AUTH_AZURE_AD_OAUTH_TENANT_ID` | Required for Azure AD  
`AUTH_AZURE_AD_OIDC_METADATA_ENDPOINT` | Optional for Azure AD  
`AUTH_AZURE_AD_X509_CERT_PATH` | Optional for Azure AD  
`AUTH_AZURE_AD_X509_CERT` | Optional for Azure AD  
`AUTH_AZURE_AD_PRIVATE_KEY_PATH` | Optional for Azure AD  
`AUTH_AZURE_AD_PRIVATE_KEY` | Optional for Azure AD  
##  S3
These variables allow you to configure S3 Object Storage. Variable | Description | Required? | Default  
---|---|---|---  
`S3_ENDPOINT` | S3 endpoint for storing results  
`S3_BUCKET` | Name of the S3 bucket for storing files  
`S3_REGION` | Region where the S3 bucket is located  
`S3_ACCESS_KEY` | Access key for authenticating with the S3 bucket  
`S3_SECRET_KEY` | Secret key for authenticating with the S3 bucket  
`S3_EXPIRATION_TIME` | Expiration time for scheduled deliveries files | 259200 (3d)  
`S3_FORCE_PATH_STYLE` | Force path style addressing, needed for MinIO setup e.g. `http://your.s3.domain/BUCKET/KEY` instead of `http://BUCKET.your.s3.domain/KEY` | `false`  
`RESULTS_S3_BUCKET` | Name of the S3 bucket used for storing query results | `S3_BUCKET`  
`RESULTS_S3_REGION` | Region where the S3 query storage bucket is located | `S3_REGION`  
`RESULTS_S3_ACCESS_KEY` | Access key for authenticating with the S3 query storage bucket | `S3_ACCESS_KEY`  
`RESULTS_S3_SECRET_KEY` | Secret key for authenticating with the S3 query storage bucket | `S3_SECRET_KEY`  
##  Cache
Note that you will need an Enterprise License Key for this functionality.
Variable | Description | Required? | Default  
---|---|---|---  
`RESULTS_CACHE_ENABLED` | Enables caching for chart results | `false`  
`AUTOCOMPLETE_CACHE_ENABLED` | Enables caching for filter autocomplete results | `false`  
`CACHE_STALE_TIME_SECONDS` | Defines how long cached results remain valid before being considered stale | 86400 (24h)  
These variables are **deprecated** ; use the `RESULTS_S3_*` versions instead.
Variable | Description | Required? | Default  
---|---|---|---  
`RESULTS_CACHE_S3_BUCKET` | Deprecated - use RESULTS_S3_BUCKET | `S3_BUCKET`  
`RESULTS_CACHE_S3_REGION` | Deprecated - use RESULTS_S3_REGION | `S3_REGION`  
`RESULTS_CACHE_S3_ACCESS_KEY` | Deprecated - use RESULTS_S3_ACCESS_KEY | `S3_ACCESS_KEY`  
`RESULTS_CACHE_S3_SECRET_KEY` | Deprecated - use RESULTS_S3_SECRET_KEY | `S3_SECRET_KEY`  
##  Logging
Variable | Description | Required? | Default  
---|---|---|---  
`LIGHTDASH_LOG_LEVEL` | The minimum level of log messages to display | `INFO`  
`LIGHTDASH_LOG_FORMAT` | The format of log messages | `pretty`  
`LIGHTDASH_LOG_OUTPUTS` | The outputs to send log messages to | `console`  
`LIGHTDASH_LOG_CONSOLE_LEVEL` | The minimum level of log messages to display on the console | `LIGHTDASH_LOG_LEVEL`  
`LIGHTDASH_LOG_CONSOLE_FORMAT` | The format of log messages on the console | `LIGHTDASH_LOG_FORMAT`  
`LIGHTDASH_LOG_FILE_LEVEL` | The minimum level of log messages to write to the log file | `LIGHTDASH_LOG_LEVEL`  
`LIGHTDASH_LOG_FILE_FORMAT` | The format of log messages in the log file | `LIGHTDASH_LOG_FORMAT`  
`LIGHTDASH_LOG_FILE_PATH` | The path to the log file | `./logs/all.log`  
##  Prometheus
Variable | Description | Required? | Default  
---|---|---|---  
`LIGHTDASH_PROMETHEUS_ENABLED` | Enables/Disables Prometheus metrics endpoint | `false`  
`LIGHTDASH_PROMETHEUS_PORT` | Port for Prometheus metrics endpoint | `9090`  
`LIGHTDASH_PROMETHEUS_PATH` | Path for Prometheus metrics endpoint | `/metrics`  
`LIGHTDASH_PROMETHEUS_PREFIX` | Prefix for metric names.  
`LIGHTDASH_GC_DURATION_BUCKETS` | Buckets for duration histogram in seconds. | `0.001, 0.01, 0.1, 1, 2, 5`  
`LIGHTDASH_EVENT_LOOP_MONITORING_PRECISION` | Precision for event loop monitoring in milliseconds. Must be greater than zero.  
`LIGHTDASH_PROMETHEUS_LABELS` | Labels to add to all metrics. Must be valid JSON  
##  Security
Variable | Description | Required? | Default  
---|---|---|---  
`LIGHTDASH_CSP_REPORT_ONLY` | Enables Content Security Policy (CSP) reporting only mode. This is recommended to be set to false in production. | `true`  
`LIGHTDASH_CSP_ALLOWED_DOMAINS` | List of domains that are allowed to load resources from. Values must be separated by commas.  
`LIGHTDASH_CSP_REPORT_URI` | URI to send CSP violation reports to.  
`LIGHTDASH_CORS_ENABLED` | Enables Cross-Origin Resource Sharing (CORS) | `false`  
`LIGHTDASH_CORS_ALLOWED_DOMAINS` | List of domains that are allowed to make cross-origin requests. Values must be separated by commas.  
##  Analytics & Event Tracking
Variable | Description | Required? | Default  
---|---|---|---  
`RUDDERSTACK_WRITE_KEY` | RudderStack key used to track events (by default Lightdash’s key is used)  
`RUDDERSTACK_DATA_PLANE_URL` | RudderStack data plane URL to which events are tracked (by default Lightdash’s data plane is used)  
`RUDDERSTACK_ANALYTICS_DISABLED` | Set to true to disable RudderStack analytics  
`POSTHOG_PROJECT_API_KEY` | API key for Posthog (by default Lightdash’s key is used)  
`POSTHOG_FE_API_HOST` | Hostname for Posthog’s front-end API  
`POSTHOG_BE_API_HOST` | Hostname for Posthog’s back-end API  
##  AI Analyst
These variables enable you to configure the AI Analyst functionality. Note that you will need an **Enterprise Licence Key** for this functionality. Variable | Description | Required? | Default  
---|---|---|---  
`AI_COPILOT_ENABLED` | Enables/Disables AI Analyst functionality  
`AI_DEFAULT_PROVIDER` | Default AI provider to use (`openai`, `azure`, `anthropic`, `openrouter`) | `openai`  
`AI_COPILOT_DEBUG_LOGGING_ENABLED` | Enables debug logging for AI Copilot | `false`  
`AI_COPILOT_TELEMETRY_ENABLED` | Enables telemetry for AI Copilot | `false`  
`AI_COPILOT_REQUIRES_FEATURE_FLAG` | Requires a feature flag to use AI Copilot | `false`  
The AI Analyst supports multiple providers for flexibility. Choose one of the provider configurations below based on your preferred AI service. **OpenAI integration is the recommended option as it is the most tested and stable implementation.**
####  OpenAI Configuration
Variable | Description | Required? | Default  
---|---|---|---  
`OPENAI_API_KEY` | API key for OpenAI | Required when using OpenAI  
`OPENAI_MODEL_NAME` | OpenAI model name to use | `gpt-4.1`  
`OPENAI_BASE_URL` | Optional base URL for OpenAI compatible API  
####  Anthropic Configuration
Variable | Description | Required? | Default  
---|---|---|---  
`ANTHROPIC_API_KEY` | API key for Anthropic | Required when using Anthropic  
`ANTHROPIC_MODEL_NAME` | Anthropic model name to use | `claude-4-sonnet-20250514`  
####  Azure AI Configuration
Variable | Description | Required? | Default  
---|---|---|---  
`AZURE_AI_API_KEY` | API key for Azure AI | Required when using Azure AI  
`AZURE_AI_ENDPOINT` | Endpoint for Azure AI | Required when using Azure AI  
`AZURE_AI_API_VERSION` | API version for Azure AI | Required when using Azure AI  
`AZURE_AI_DEPLOYMENT_NAME` | Deployment name for Azure AI | Required when using Azure AI  
####  OpenRouter Configuration
Variable | Description | Required? | Default  
---|---|---|---  
`OPENROUTER_API_KEY` | API key for OpenRouter | Required when using OpenRouter  
`OPENROUTER_MODEL_NAME` | OpenRouter model name to use | `openai/gpt-4.1-2025-04-14`  
`OPENROUTER_SORT_ORDER` | Provider sorting method (`price`, `throughput`, `latency`) | `latency`  
`OPENROUTER_ALLOWED_PROVIDERS` | Comma-separated list of allowed providers (`anthropic`, `openai`, `google`) | `openai`  
##  Slack Integration
These variables enable you to configure the Slack integration. Variable | Description | Required? | Default  
---|---|---|---  
`SLACK_SIGNING_SECRET` | Required for Slack integration  
`SLACK_CLIENT_ID` | Required for Slack integration  
`SLACK_CLIENT_SECRET` | Required for Slack integration  
`SLACK_STATE_SECRET` | Required for Slack integration | `slack-state-secret`  
`SLACK_APP_TOKEN` | App token for Slack  
`SLACK_PORT` | Port for Slack integration | `4351`  
`SLACK_SOCKET_MODE` | Enable socket mode for Slack | `false`  
`SLACK_CHANNELS_CACHED_TIME` | Time in milliseconds to cache Slack channels |  `600000` (10 minutes)  
`SLACK_SUPPORT_URL` | URL for Slack support  
##  GitHub Integration
These variables enable you to configure Github integrations Variable | Description | Required? | Default  
---|---|---|---  
`GITHUB_PRIVATE_KEY` | GitHub private key for GitHub App authentication  
`GITHUB_APP_ID` | GitHub Application ID  
`GITHUB_CLIENT_ID` | GitHub OAuth client ID  
`GITHUB_CLIENT_SECRET` | GitHub OAuth client secret  
`GITHUB_APP_NAME` | Name of the GitHub App  
`GITHUB_REDIRECT_DOMAIN` | Domain for GitHub OAuth redirection  
##  Microsoft Teams Integration
These variables enable you to configure Microsoft Teams integration. Variable | Description | Required? | Default  
---|---|---|---  
`MICROSOFT_TEAMS_ENABLED` | Enables Microsoft Teams integration | `false`  
##  Google Cloud Platform
These variables enable you to configure Google Cloud Platform integration. Variable | Description | Required? | Default  
---|---|---|---  
`GOOGLE_CLOUD_PROJECT_ID` | Google Cloud Platform project ID  
`GOOGLE_DRIVE_API_KEY` | Google Drive API key  
`AUTH_GOOGLE_ENABLED` | Enables Google authentication | `false`  
`AUTH_ENABLE_GCLOUD_ADC` | Enables Google Cloud Application Default Credentials | `false`  
##  Embedding
Note that you will need an **Enterprise Licence Key** for this functionality. Variable | Description | Required? | Default  
---|---|---|---  
`EMBEDDING_ENABLED` | Enables embedding functionality | `false`  
`LIGHTDASH_IFRAME_EMBEDDING_DOMAINS` | List of domains that are allowed to embed Lightdash in an iframe. Values must be separated by commas.  
##  Service account
Note that you will need an **Enterprise Licence Key** for this functionality. Variable | Description | Required? | Default  
---|---|---|---  
`SERVICE_ACCOUNT_ENABLED` | Enables service account functionality | `false`  
##  SCIM
Note that you will need an **Enterprise Licence Key** for this functionality. Variable | Description | Required? | Default  
---|---|---|---  
`SCIM_ENABLED` | Enables SCIM (System for Cross-domain Identity Management) | `false`  
##  Sentry
These variables enable you to configure Sentry for error tracking. Variable | Description | Required? | Default  
---|---|---|---  
`SENTRY_DSN` | Sentry DSN for both frontend and backend  
`SENTRY_BE_DSN` | Sentry DSN for backend only  
`SENTRY_FE_DSN` | Sentry DSN for frontend only  
`SENTRY_BE_SECURITY_REPORT_URI` | URI for Sentry backend security reports  
`SENTRY_TRACES_SAMPLE_RATE` | Sample rate for Sentry traces (0.0 to 1.0) | `0.1`  
`SENTRY_PROFILES_SAMPLE_RATE` | Sample rate for Sentry profiles (0.0 to 1.0) | `0.2`  
`SENTRY_ANR_ENABLED` | Enables Sentry Application Not Responding detection | `false`  
`SENTRY_ANR_CAPTURE_STACKTRACE` | Captures stacktrace for ANR events | `false`  
`SENTRY_ANR_TIMEOUT` | Timeout in milliseconds for ANR detection  
##  Intercom & Pylon
These variables enable you to configure Intercom and Pylon for customer support and feedback. Variable | Description | Required? | Default  
---|---|---|---  
`INTERCOM_APP_ID` | Intercom application ID  
`INTERCOM_APP_BASE` | Base URL for Intercom API | `https://api-iam.intercom.io`  
`PYLON_APP_ID` | Pylon application ID  
`PYLON_IDENTITY_VERIFICATION_SECRET` | Secret for verifying Pylon identities  
##  Kubernetes
These variables enable you to configure Kubernetes integration. Variable | Description | Required? | Default  
---|---|---|---  
`K8S_NODE_NAME` | Name of the Kubernetes node  
`K8S_POD_NAME` | Name of the Kubernetes pod  
`K8S_POD_NAMESPACE` | Namespace of the Kubernetes pod  
`LIGHTDASH_CLOUD_INSTANCE` | Identifier for Lightdash cloud instance  
##  **Organization appearance**
These variables allow you to customize the default appearance settings for your Lightdash instance’s organizations. This color palette will be set for all organizations in your instance. You can’t choose another one while these env vars are set. Variable | Description | Required? | Default  
---|---|---|---  
`OVERRIDE_COLOR_PALETTE_NAME` | Name of the default color palette  
`OVERRIDE_COLOR_PALETTE_COLORS` | Comma-separated list of hex color codes for the default color palette (must be 20 colors)  
##  Initialize instance
When a new Lightdash instance is created, and there are no orgs and projects. You can initialize a new org and project using ENV variables to simplify the deployment process.
**Initialize instance is only available on Lightdash Enterprise plans.** For more information on our plans, visit our pricing page.
Currently we only support Databricks project types and Github dbt configuration.
Variable | Description | Required? | Default  
---|---|---|---  
`LD_SETUP_ADMIN_NAME` | Name of the admin user for initial setup | `Admin User`  
`LD_SETUP_ADMIN_EMAIL` | Email of the admin user for initial setup  
`LD_SETUP_ORGANIZATION_EMAIL_DOMAIN` | Email domain for the organization whitelisting  
`LD_SETUP_ORGANIZATION_DEFAULT_ROLE` | Default role for new organization members | `viewer`  
`LD_SETUP_ORGANIZATION_NAME` | Name of the organization  
`LD_SETUP_ADMIN_API_KEY` | API key for the admin user, must start with `ldpat_` prefix  
`LD_SETUP_API_KEY_EXPIRATION` | Number of days until API key expires (0 for no expiration)  
`LD_SETUP_SERVICE_ACCOUNT_TOKEN` | A pre-set token for the service account, must start with `ldsvc_` prefix  
`LD_SETUP_SERVICE_ACCOUNT_EXPIRATION` | Number of days until service account token expires (0 for no expiration)  
`LD_SETUP_PROJECT_NAME` | Name of the project  
`LD_SETUP_PROJECT_CATALOG` | Catalog name for Databricks project  
`LD_SETUP_PROJECT_SCHEMA` | Schema/database name for the project  
`LD_SETUP_PROJECT_HOST` | Hostname for the Databricks server  
`LD_SETUP_PROJECT_HTTP_PATH` | HTTP path for Databricks connection  
`LD_SETUP_PROJECT_PAT` | Personal access token for Databricks  
`LD_SETUP_START_OF_WEEK` | Day to use as start of week | `SUNDAY`  
`LD_SETUP_PROJECT_COMPUTE` | JSON string with Databricks compute configuration like `{"name": "string", "httpPath": "string"}`  
`LD_SETUP_DBT_VERSION` | Version of dbt to use (eg: v1.8) | `latest`  
`LD_SETUP_GITHUB_PAT` | GitHub personal access token  
`LD_SETUP_GITHUB_REPOSITORY` | GitHub repository for dbt project  
`LD_SETUP_GITHUB_BRANCH` | GitHub branch for dbt project  
`LD_SETUP_GITHUB_PATH` | Subdirectory path within GitHub repository  
In order to login as the admin user using SSO, you must enable the following ENV variable too:
Copy
Ask AI
```
AUTH_ENABLE_OIDC_TO_EMAIL_LINKING=true

```

This will alow you to link your SSO account with the email provided without using an invitation code. This email will be trusted, and any user with an OIDC account with that email will access the admin user.
##  Update instance
On server start, we will check the following variables, and update some configuration of the organization or project. Having more than one org or project can throw runtime errors and prevent the server from starting.
**Initialize instance is only available on Lightdash Enterprise plans.** For more information on our plans, visit our pricing page.
Variable | Description | Required? | Default  
---|---|---|---  
`LD_SETUP_ADMIN_EMAIL` | Email of the admin to update its Personal access token | Required if `LD_SETUP_ADMIN_API_KEY` is present  
`LD_SETUP_ADMIN_API_KEY` | API key for the admin user, must start with `ldpat_` prefix  
`LD_SETUP_PROJECT_HTTP_PATH` | HTTP path for Databricks connection  
`LD_SETUP_DBT_VERSION` | Version of dbt to use (eg: v1.8) | `latest`  
`LD_SETUP_GITHUB_PAT` | GitHub personal access token  
`LD_SETUP_SERVICE_ACCOUNT_TOKEN` | A pre-set token for the service account, must start with `ldsvc_` prefix  
Assistant
Responses are generated using AI and may contain mistakes.


