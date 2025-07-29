# SQL Variables - Lightdash

**Source:** https://docs.lightdash.com/references/sql-variables

Lightdash home page
Search...
⌘KAsk AI
Search...
Navigation
Reference
SQL Variables
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
    * Lightdash CLI reference
    * Feature Maturity Levels
    * SQL Runner


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


##### Contact


  * `${field}` - reference a field in the current model
  * `${model.field}` - reference a field in another model
  * `${TABLE}` - reference the current table’s sql reference
  * `${lightdash.attributes.my_attr_1}` - a user attribute called `my_attr_1`
    * (optional) `ld` as an alias for `lightdash`
    * (optional) `attribute` or `attr` as an alias for `attributes`
  * `${lightdash.user.<intrinsic_attribute>}` - reference an `intrinsic_attribute` of the current Lightdash user 
    * (optional) `ld` as an alias for `lightdash`
    * available intrinsic user attributes: 
      * `email`


**The user email attribute is only available when the email is verified.** This is a security measure to prevent users from creating/updating an account with any email they don’t own and gain access to data they shouldn’t see.If the user email is not verified you will get the following error:
Copy
Ask AI
```
models:
name: example
    meta:
      sql_filter: ${lightdash.user.email} = 'example@lightdash.com'

```

If you are self hosting you can enable SMTP or SSO authentication to allow users to verify their email address.
Dashboards as codeCaching
Assistant
Responses are generated using AI and may contain mistakes.


