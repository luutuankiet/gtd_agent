# Personal access tokens - Lightdash

**Source:** https://docs.lightdash.com/references/personal_tokens

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Reference
Personal access tokens
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
    * Roles and permissions
    * Personal access tokens


##### Integrations


##### Embedding and SDKs
  * Embedding quickstart
  * Embedding reference


##### Self-hosting and deployment
  * Self-Host Lightdash
  * Lightdash Cloud vs. Self-Hosted
  * Updating Lightdash
  * Customize deployment


##### Contact


On this page
  * Creating a personal access token
  * How to authenticate
  * Rotating a personal access token


You can create a personal access token (PAT) to authenticate in the CLI or with the API.
##  Creating a personal access token
To create a personal access token, head to the project settings, then to `personal access tokens`, then click `generate token`. Everybody in your organization can create a personal access token (PAT). To provide additional security, we highly recommend adding an expiration to your personal access tokens.
###  How to authenticate
Note that not all endpoints in the API will be accessible with PAT authentication. The PAT authentication is done via the HTTP Authorization request header.
Copy
Ask AI
```
Authorization: ApiKeytoken

```

Examples:
Copy
Ask AI
```
curl --location --request GET 'https://my.lightdash.com/api/v1/org/projects' \
--header 'Authorization: ApiKey eeaa81d8bcd89a770bd8a581cabd052b'

```

Copy
Ask AI
```
GET /api/v1/org/projects HTTP/1.1
Host: my.lightdash.com
Authorization: ApiKey eeaa81d8bcd89a770bd8a581cabd052b

```

##  Rotating a personal access token
To maintain security, you may want to rotate your personal access tokens periodically. Follow these steps to rotate an existing token. Requirements for Rotation
  * **Token UUID** : To rotate a token, you’ll need its unique identifier (UUID), which you can find on the tokens page in your account settings.
  * **Expiration Date** : You must specify a new expiration date when rotating a token. Tokens without an expiration cannot be rotated.
  * **Time Interval** : Tokens can only be rotated once per hour.

Use the following curl command to rotate a personal access token. Replace `<personal-access-token-uuid>` with the UUID of the token you wish to rotate, and `<personal-access-token>` with your current token.
Copy
Ask AI
```
curl --location --request PATCH 'https://my.lightdash.com/api/v1/user/me/personal-access-tokens/<personal-access-token-uuid>/rotate' \
--header 'Content-Type: application/json' \
--header 'Authorization: ApiKey <personal-access-token>' \
--data '{
    "expiresAt": "2025-12-13T16:10:04.976Z"


```

This process will invalidate the previous token and return a new token with the specified expiration date.
Assistant
Responses are generated using AI and may contain mistakes.


