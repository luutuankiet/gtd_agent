# React SDK - Lightdash

**Source:** https://docs.lightdash.com/references/react-sdk

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Embedding and SDKs
React SDK
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


##### Contact


On this page
  * Installing the Lightdash SDK
  * Using the Lightdash SDK
  * Generating an embed token
  * Applying styles
  * Filtering data
  * Available fields
  * Video overview of Localization
  * Translation maps
  * Runtime translation
  * Translation tools
  * What can be translated?


##  Set up CORS
To make the React SDK work, you need to update your “Cross-Origin Resource Sharing” (CORS) policy. This is done using environment variables. For Lightdash Cloud customers, you’ll need to contact the Lightdash team to update these for you.
Copy
Ask AI
```
LIGHTDASH_CORS_ENABLED=true
LIGHTDASH_CORS_ALLOWED_DOMAINS=https://domain-where-you-are-going-to-use-the-sdk.com

```

**Why this is needed?** Enabling CORS (Cross-Origin Resource Sharing) is necessary because browsers enforce security policies that prevent web applications from making requests to a different domain than the one that served the app (known as the **Same-Origin Policy**).Since the **Lightdash React SDK** interacts with an Lightdash API, you need to configure CORS on your Lightdash instance to allow your frontend application to communicate with the Lightdash server without being blocked by the browser.
##  Installing the Lightdash SDK
In your frontend project, use your preferred package manager to install the SDK.
Copy
Ask AI
```
npm install @lightdash/sdk
# or
pnpm add @lightdash/sdk
# or
yarn add @lightdash/sdk

```

At the moment, we support React 18 and 19, so make sure your frontend is using React 18 or later. For Next.js, version 15 or later is required.
##  Using the Lightdash SDK
In your frontend project, import and use the `Lightdash.Dashboard` component in your desired location to mount the Lightdash dashboard.
Copy
Ask AI
```
import Lightdash from '@lightdash/sdk';
Lightdash.Dashboard
    instanceUrl="{{ your instance URL (e.g., https://lightdash.yourcompanydomain.com) }}"
    projectUuid="{{ project ID that the dashboard is part of }}"
    token="see the section below on how to generate a token for the SDK"


```

##  Generating an embed token
###  Development
When developing you can quickly and easily generate tokens through the Lightdash UI. Since this requires manual steps we don’t recommend you do this in production.
###  Production
To run the Lightdash SDK in production you need:
  1. A frontend to embed the Lightdash dashboard.
  2. A backend (or other privileged environment) where you can safely generate embedding tokens. This component will be able to generate access to any content, so this shouldn’t be accessible to your end users.

**JWT Authentication Flow** To generate a signed JWT token, you need to use your **embed secret** , which is located in **Settings → Embed** in Lightdash. There, you can generate or regenerate the secret. This secret is used to sign JWT tokens, allowing secure use of the SDK. You can use the form to configure which **Dashboard** you want to show in the SDK, enable specific features, and set up **User Attributes** for advanced security controls. For more details on **User Attributes** , check out the guide here. After you configure the form you should end up with the snippet which looks like this:
Copy
Ask AI
```
import jwt from 'jsonwebtoken';
const LIGHTDASH_EMBED_SECRET = '{{ keep this secret and do not expose it in the frontend }}';
const projectUuid = '{{ your project uuid }} ';
const data = {
    content: {
        type: 'dashboard',
        dashboardUuid: '{{your dashboard uuid}}',
        dashboardFiltersInteractivity: {
            enabled: "none",
            allowedFilters: undefined,

        canExportCsv: false,
        canExportImages: false,
        canExportPagePdf: true,
        canDateZoom: false,

    user: {
        externalId: undefined,
        email: "{{ authenticated user email }} "

    userAttributes: {"":""},

const token = jwt.sign(data, LIGHTDASH_EMBED_SECRET, { expiresIn: '1 hour' });
console.log({ projectUuid, token })
// ^^^ use generated projectUuid and token in the `<Lightdash.Dashboard` props

```

The Lightdash SDK works the same way as our embedding feature, which embeds the Lightdash dashboard as an “iframe”. We recommend reading the documentation on Embedding, which provides a detailed explanation of how token generation works and how to securely use Lightdash content using the SDK.
To ensure security, the JWT token generation code should run **in your backend** , and the **Lightdash embed secret** must never be exposed in frontend code. This prevents unauthorized access and protects sensitive data.
##  Applying styles
It’s possible to override some styles within the `<Lightdash.Dashbaord/>` component to match the surrounding page better. Some styles will cascade, but some charts and components set things like `font-family` explicitly, so it is necessary to pass them to the component. Supported style overrides are:
  * `fontFamily` - which will set all fonts within the dashboard to the specified font family. Note that only font family will be updated; font sizes and other properties will be preserved.
  * `backgroundColor` - sets the background for the dashboard (not the tiles). This can be set to any color or `transparent` .

Both of these properties accept normal css values and are set on a `styles` object passed to the dashboard. For example:
Copy
Ask AI
```
Lightdash.Dashboard
    instanceUrl={lightdashUrl}
    token={lightdashToken}
    styles={{
        backgroundColor: 'transparent',
        fontFamily: 'Comic Sans MS',



```

##  Filtering data
Filters can be passed to the `<Lightdash.Dashbaord/>` to filter dimensions by values. Filters are passed as an array to the dashboard like this:
Copy
Ask AI
```
Lightdash.Dashboard
    instanceUrl={lightdashUrl}
    token={lightdashToken}
    filters={[

            model: 'dbt_users',
            field: 'created_date_week',
            operator: FilterOperator.IN_BETWEEN,
            value: ['2024-08', '2024-10'],


            model: 'dbt_users',
            field: 'browser',
            operator: FilterOperator.INCLUDE,
            value: ['chrome', 'safari'],




```

Each entry in the array specifies
  * `field` - the name of the dimension to filter by
  * `model` - the model the dimension is a part of
  * `operator` - the fitler operator, specified with the `FilterOperator` enum
  * `value` - the value or values to fitler against. Some opperators, such as `IN_BETWEEN` expect an array, others take only a value

Filter objects will each be applied as AND operations, each further restricting results.
###  Available fields
Note that only field that are available for filtering will be filterable. These are specified in the token passed to the SDK. To generate such a token, see the Lightdash embed configuration.
##  Localization
The **Lightdash SDK** supports multilingual translation using standard i18n translation objects. To display a translated Lightdash dashboard, an app simply needs to pass a correctly formatted translation object to the SDK. We recommend using the following tools (or similar) to create and manage translations efficiently:
  * **Translation maps** – The Lightdash CLI can generate translation maps when downloading content as code. These maps include keys and original text for all translatable strings in a dashboard, making them easy to use with translation tools like **Locize** or for manual translation. See below for details.
  * **Runtime translation management** – In apps using the SDK, we recommend using a translation library like `i18next` to handle translation parameters and generate translation objects dynamically. See below for an example.
  * **Translation production tools** – Tools like **Locize** help translators manage translations efficiently during production. Locize, for instance, can process translation objects from the Lightdash CLI translation maps and generate the dictionaries required by the SDK. These dictionaries can be included statically in the app or fetched dynamically from Locize servers. See the demo video below for an overview.


###  Video overview of Localization
###  Translation maps
The Lightdash CLI can produce translation maps, which specify translatable strings in a structured object which can be translated in a Lightdash dashboard. To include translation maps when downloading content with the Lightdash CLI, add the `--language-map` flag:
Copy
Ask AI
```
lightdash download --language-map

```

Alongside each downloaded dashboard and chart there will be a `<file name>.language.map.yml` file containing a nested object and its existing translatable strings. Here’s a simple dashboard language map:
Copy
Ask AI
```
dashboard:
  sdk-dash:
    name: SDK dashboard demo
    description: "A dashboard demonstrating SDK features"
    tiles:
type: markdown
        properties:
          title: SDK demo dashboard
          content: -
            This dashboard contains various tile types for showing SDK
            features.
type: saved_chart
        properties:
          title: "How do payment methods vary across different amount ranges?"

```

These translation maps can be imported directly into a tool like Locize to begin translation.
###  Runtime translation
At runtime, if a translation object is passed to the Lightdash SDK `contentOverrides` prop, the specified translations will be made. We suggest using `i18Next` to load translations, specify a language and produce the translation object (with `getResoruceBundle`):
Copy
Ask AI
```
Lightdash.Dashboard
    instanceUrl={lightdashUrl}
    token={lightdashToken}
    contentOverrides={i18n.getResourceBundle(
        i18n.language// Specify language
        'demo-dashboard', // Specify namespace



```

There are a number of ways that `i18Next` can be set up to fetch translations, but it can be set up to fetch translations directly from Locize or another language server.
###  Translation tools
We suggest using Locize (or equivalent) to produce and store translations. Locize can load the language maps produced by the Lightdash CLI. It will show the translatable strings in the dashboard or chart nested under the correct keys. Translators can then add the necessary translation strings in the target language (Locize can also fill them in from google translate). The resulting translation objects can be exported and included in an app, or loaded dynamically. The basic setup for loading translations dynamically from Locize looks like this:
Copy
Ask AI
```
i18next
use(Locize) // Add Locize
use(initReactI18next) // Bind react-i18next to the instance
init({

        // Initialization properties, including app ID



```

###  What can be translated?
Strings that can be specified in the Lightdash UI can be translated. For example, titles, descriptions, chart names, axis names and series names can all be translated. Strings originating from a client warehouse cannot be translated by the SDK. String data within a chart will be presented as it exists in the database.
Embedding referenceSelf-Host Lightdash
Assistant
Responses are generated using AI and may contain mistakes.


