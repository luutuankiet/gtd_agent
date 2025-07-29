# Dimensions reference - Lightdash

**Source:** https://docs.lightdash.com/references/dimensions

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Reference
Dimensions reference
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


On this page
  * Adding dimensions to your project
  * Dimension configuration
  * Quotes for escaping
  * Greater than symbol for folded text blocks
  * Vertical bar for preserving line breaks
  * Using dbt doc blocks
  * Example format expressions:
  * Time intervals
  * By default, the time intervals we use are:
  * To turn off time intervals for a dimension, set time_intervals: OFF
  * To customize the time intervals for a dimension, you can use the time_intervals parameter.
  * Numeric options
  * String options
  * How to add custom URLs
  * You can reference values from other columns in your URLs
  * Liquid Templating
  * Required attributes
  * Current limitations
  * Using special characters or capital letters in your column names
  * Additional dimensions
  * Adding different formatting
  * Comparing or combining columns
  * Parsing JSON columns
  * Adding multiple timezones for the same dimension
  * Using additional dimensions in metrics


Dimensions usually match 1:1 with columns in your dbt models (see additional dimensions for counterexamples).
##  Adding dimensions to your project
Read more about adding dimensions to your project in our docs here. For a dimension to appear in Lightdash, you just need to declare it in your dbt model’s YAML file.
Copy
Ask AI
```
models:
name: my_model
    columns:
name: user_id # will be "User id" in LightDash
        description: "Unique identifier for a user."

```

##  Dimension configuration
To customize the dimension, you can do it in your dbt model’s YAML file. If you want to declare multiple dimensions based on the same column, check additional dimensions section.
Copy
Ask AI
```
models:
name: sales_stats
    meta:
      group_details:
        finance:
          label: Finance
          description: Finance-related fields.
      joins:
join: web_sessions
          sql_on: ${web_sessions.date} = ${sales_stats.date}
    columns:
name: revenue_gbp_total_est
        description: 'Total estimated revenue in GBP based on forecasting done by the finance team.'
        meta:
          dimension:
            type: number
            label: 'Total revenue' # this is the label you'll see in Lightdash
            description: 'My custom description' # you can override the description you'll see in Lightdash here
: 'IF(${TABLE}.revenue_gbp_total_est = NULL, 0, ${registered_user_email})' # custom SQL applied to the column from dbt used to define the dimension
            hidden: false
            format: '[$£]#,##0.00' # GBP rounded to two decimal points
            groups: ['finance']
name: forecast_date
        description: 'Date of the forecasting.'
        meta:
          dimension:
            type: date
            time_intervals: ['DAY', 'WEEK', 'MONTH', 'QUARTER'] # not required: the default time intervals for dates are `['DAY', 'WEEK', 'MONTH', 'YEAR']`
            urls:
label: 'Open in forecasting tool'
: 'https://finance.com/forceasts/weeks/${ value.raw }'
label: Open in Google Calendar
: 'https://calendar.google.com/calendar/u/0/r/day/${ value.formatted |split: "-" |join: "/"}'
            required_attributes:
              is_admin: 'true'
name: date
        meta:
          dimension:
            type: date
            day_of_week_num:
              type: number
              label: 'Day of Week (number)'
: 'day_of_week(${date})'
            day_of_week_day:
              type: date
              label: 'Day of Week (day)'
: ${date}
              format: 'dddd'
            is_weekday_or_weekend:
              type: boolean
              label: 'Weekday or Weekend'
: "case when day_of_week(${date}) < 5 then 'weekday' else 'weekend' end"

```

All the properties you can customize: Property | Required | Value | Description  
---|---|---|---  
label | No | string | Custom label. If you set this property, this is what you’ll see in Lightdash instead of the dimension name.  
No | Dimension type | The dimension type is automatically pulled from your table schemas in Lightdash but you can override the type using this property.  
description | No | string | Description of the dimension in Lightdash. You can use this to override the description you have for the dimension in dbt.  
sql | No | string | Custom SQL applied to the column used to define the dimension.  
time_intervals | No | ’default’ or OFF or an array[] containing elements of date, numeric or string options | ’default’ (or not setting the time_intervals property) will be converted into [‘DAY’, ‘WEEK’, ‘MONTH’, ‘QUARTER’, ‘YEAR’] for dates and [‘RAW’, ‘DAY’, ‘WEEK’, ‘MONTH’, ‘QUARTER’, ‘YEAR’] for timestamps; if you want no time intervals set ‘OFF’.  
hidden | No | boolean | If set to true, the dimension is hidden from Lightdash. By default, this is set to false if you don’t include this property.  
No | string | This option will compact the number value (e.g. 1,500 to 1.50K). Currently supports one of the following: [‘thousands’, ‘millions’, ‘billions’, ‘trillions’, ‘kilobytes’, ‘megabytes’, ‘gigabytes’, ‘terabytes’, ‘petabytes’, ‘kibibytes’, ‘mebibytes’, ‘gibibytes’, ‘tebibytes’, ‘pebibytes’]  
No | string | This option will format the output value on the results table and CSV export. Supports spreadsheet-style formatting (e.g. #,##0.00). Use this website to help build your custom format.`  
No | string or string[] | If you set this property, the dimension will be grouped in the sidebar with other dimensions with the same group label.  
No | Array of { url, label } | Adding urls to a dimension allows your users to click dimension values in the UI and take actions, like opening an external tool with a url, or open at a website. You can use liquid templates to customise the link based on the value of the dimension.  
required_attributes | No | Object with { user_attribute, value } | Limits access to users with those attributes  
No | Object with { value, color } | Color for the values in the chart  
##  Type
The types of your dimensions are pulled from your data warehouse, automatically. You can override these types using the `type` meta tag in your .yml file. If you run `lightdash generate` to generate your .yml files, then Lightdash will add the `type` from your data warehouse to your .yml files automatically.
Copy
Ask AI
```
- name: user_created_date
  meta:
    dimension:
      type: date

```

We currently support these dimension types: Dimension Types  
---  
string  
number  
timestamp  
date  
boolean  
##  Description
Column descriptions in your YAML file are automatically pulled into Lightdash and you can spot them if you hover over the dimension name. Descriptions support any formatting that works with YAML, but the three characters used most often are:
###  Quotes for escaping
When you surround text with double or single quotes it will escape the text between so that any special characters recognized by YAML will still pass through to the Lightdash UI.
Copy
Ask AI
```
description: 'The contents of this column include this & that.'

```

###  Greater than symbol for folded text blocks
When you use `>-` it allows you to type descriptions that are multiple lines long in the YAML file, but the text will be combined into a single line when parsed. The `lightdash generate` command will automatically add this to keep YAML files easy to read. This description in YAML:
Copy
Ask AI
```
- name: product_tier
  description: -
    This is a longer description...
    ...that requires multiple lines
    and it will be combined in the Lightdash UI

```

Will appear like this in the Lightdash UI:
Copy
Ask AI
```
This is a longer description......that requires multiple lines and it will be combined in the Lightdash UI

```

###  Vertical bar for preserving line breaks
If you need line breaks to stay in place when they show up in the Lightdash UI, you can use a `|` character like this:
Copy
Ask AI
```
- name: product_tier
  description: |
    This is a longer description...
    ...that requires multiple lines
    and it will stay on multiple lines

```

And in Lightdash UI it will appear like this:
Copy
Ask AI
```
This is a longer description...
...that requires multiple lines
and it will stay on multiple lines

```

###  Using dbt doc blocks
You can also use dbt docs blocks in descriptions, more on that here.
##  Format
You can use the `format` parameter to have your dimensions show in a particular format in Lightdash. Lightdash supports spreadsheet-style format expressions for all dimension types. To help you build your format expression, we recommend using https://customformats.com/.
Copy
Ask AI
```
- name: us_revenue
  meta:
    dimension:
      type: number
      description: 'Revenue in USD, with two decimal places, compacted to thousands'
      format: '$#,##0.00," K"' # 505,430 will appear as '$505.43 K'
    additional_dimensions:
      percent_of_global_revenue:
        type: number
        description: 'Percent of total global revenue coming from US revenue.'
        sql: ${us_revenue} / ${global_revenue}
        format: '0.00%' # 0.67895243 will appear as '67.89%

```

###  Example format expressions:
Description | Format Expression | Raw Value | Formatted Output  
---|---|---|---  
**Adds “km” suffix to the value** | `#,##0.00" km"` | 100000.00 | 100,000.00 km  
15000.25 | 15,000.25 km  
500 | 500.00 km  
**Format date with 12-hour clock** | `m/d/yyyy h:mm AM/PM` | 2023-09-05T15:45:00Z | 9/5/2023 3:45 PM  
2024-01-20T08:30:00Z | 1/20/2024 8:30 AM  
**Display the full name of the day** | `dddd` | 2023-09-05T15:45:00Z | Tuesday  
2024-01-20T08:30:00Z | Saturday  
**Format positive, negative, and zero values** | `"⬆️ "0;"⬇️ "0;0` | -500 | ⬇️ 500  
200 | ⬆️ 200  
**Text formatting** | `"Delivered in "@` | 2 weeks | Delivered in 2 weeks  
18 hours | Delivered in 18 hours  
**Percentage formatting** | `#,##0.00%` | 0.6758 | 67.58%  
0.1 | 10.00%  
0.002 | 0.20%  
**No formatting** | 12345232 | 12345232  
56.7856 | 57  
**Currency formatting** | `[$]#,##0.00` | 15430.75436 | $15,430.75  
15430.75436 | $15,430.75  
**Compact currency in thousands** | `[$]#,##0,"K"` | 15430.75436 | $15K  
15430.75436 | $15.43K  
**Compact currency in millions** | `[$]#,##0.00,,"M"` | 13334567 | $13.33M  
120000000 | $120.00M  
(Legacy) format and round options
Spreadsheet-style format expressions are the recommended way of adding formatting to your metrics in Lightdash. There are legacy formatting options, listed below, which are less flexible than the spreadsheet-style formatting.
If you use both legacy and spreadsheet-style formatting options for a single dimension, Lightdash will ignore the legacy `format` and `round` options and only apply the spreadsheet-style formatting expression.
####  Format (legacy)
Copy
Ask AI
```
models:
name: sales_stats
    columns:
name: revenue
        description: 'Total estimated revenue in GBP based on forecasting done by the finance team.'
        meta:
          dimension:
            format: 'gbp'

```

These are the options: Option | Equivalent format expression | Description | Raw value | Displayed value  
---|---|---|---|---  
km | ’#,##0.00” km“‘ | Adds the suffix `km` to your value | 10 | 10 km  
mi | ’#,##0.00” mi“‘ | Adds the suffix `mile` to your value | 10 | 10 mi  
usd | ’[$]#,##0.00’ | Adds the `$` symbol to your number value | 10 | $10.00  
gbp | ’[$£]#,##0.00’ | Adds the `£` symbol to your number value | 10 | £10.00  
eur | ’[$€]#,##0.00’ | Adds the `€` symbol to your number value | 10 | €10.00  
jpy | ’[$¥]#,##0.00’ | Adds the `¥` symbol to your number value | 10 | ¥10  
percent | ’#,##0.00%‘ | Adds the `%` symbol and multiplies your value by 100 | 0.1 | %10  
id | ’0’ | Removes commas and spaces from number or string types so that they appear like IDs. | 12389572 | 12389572  
####  Round (legacy)
You can round values to appear with a certain number of decimal points.
Copy
Ask AI
```
	models:
name: sales
	  columns:
name: revenue
	      meta:
	        dimension:
	          round: 0 # equivalent format expression: '#,##0.0'

```

##  Compact
You can compact values in your YAML. For example, if I wanted all of my revenue values to be shown in thousands (e.g. `1,500` appears as `1.50K`), then I would write something like this in my .yml:
Copy
Ask AI
```
models:
name: sales
      columns:
name: revenue
            meta:
                dimension:
                    compact: thousands # You can also use 'K'

```

Value | Alias | Equivalent format expression | Example output  
---|---|---|---  
thousands | ”K” and “thousand” | ’#,##0,” K”’ or ’#,##0.00,” K“‘ | 1K  
millions | ”M” and “million” | ’#,##0,,” M”’ or ’#,##0.00,,” M“‘ | 1M  
billions | ”B” and “billion” | ’#,##0,,,” B”’ or ’#,##0.00,,,” B“‘ | 1B  
trillions | ”T” and “trillion” | ’#,##0,,,,” T”’ or ’#,##0.00,,,,” T“‘ | 1T  
kilobytes | ”KB” and “kilobyte” | 1KB  
megabytes | ”MB” and “megabyte” | 1MB  
gigabytes | ”GB” and “gigabyte” | 1GB  
terabytes | ”TB” and “terabyte” | 1TB  
petabytes | ”PB” and “petabyte” | 1PB  
kibibytes | ”KiB” and “kibibyte” | 1KiB  
mebibytes | ”MiB” and “mebibyte” | 1MiB  
gibibytes | ”GiB” and “gibibyte” | 1GiB  
tebibytes | ”TiB” and “tebibyte” | 1TiB  
pebibytes | ”PiB” and “pebibyte” | 1PiB  
##  Time intervals
Lightdash automatically adds intervals for dimensions that are timestamps or dates, so you don’t have to! For example, here we have the timestamp dimension `created` defined in our dbt project:
Copy
Ask AI
```
- name: created
  description: 'Timestamp when the user was created.'

```

Lightdash breaks this out into the default intervals automatically. So, this is how `created` appears in our Lightdash project:
**Formatting added to a date or timestamp dimension will be applied to all of the time intervals for that dimension.**If you want to apply different formats for different time intervals, we recommend creating additional dimensions for time intervals where you want to customize the format.
###  By default, the time intervals we use are:
**Date** : [‘DAY’, ‘WEEK’, ‘MONTH’, ‘QUARTER’, ‘YEAR’] **Timestamp** : [‘RAW’, ‘DAY’, ‘WEEK’, ‘MONTH’, ‘QUARTER’, ‘YEAR’]
###  To turn off time intervals for a dimension, set `time_intervals: OFF`
If you want to turn off time intervals for a dimension, you can simply set the `time_intervals` property to `OFF`. In this example, `created` would now appear as a single, timestamp dimension without a drop-down list of time intervals in Lightdash:
Copy
Ask AI
```
- name: created
  description: 'Timestamp when the user was created.'
  meta:
    dimension:
      time_intervals: OFF

```

###  To customize the time intervals for a dimension, you can use the `time_intervals` parameter.
If you specify time intervals manually, then this overrides the default time intervals used by Lightdash.
Copy
Ask AI
```
- name: created
  description: 'Timestamp when the user was created.'
  meta:
    dimension:
      time_intervals: ['DAY', 'DAY_OF_MONTH_NUM', 'MONTH', 'QUARTER_NAME', 'YEAR']

```

You can see all of the interval options for date and timestamp fields below.
###  Date options
Option | Description | Type | Displayed value | Notes  
---|---|---|---|---  
RAW | Original value | Date / DateTime | 2019-01-01 / 2019-01-01, 09:30:30:300 UTC  
YEAR | Date truncated to the nearest year | Date | 2019  
QUARTER | Date truncated to the nearest quarter | Date | 2019-Q1  
MONTH | Date truncated to the nearest month | Date | 2019-01-01  
WEEK | Date truncated to the nearest week | Date | 2019-01-01 | The start of the week depends on your warehouse configuration  
DAY | Date truncated to the nearest day | Date | 2019-01-01  
HOUR | Datetime truncated to the nearest hour | DateTime | 2019-01-01, 09 UTC  
MINUTE | Datetime truncated to the nearest minute | DateTime | 2019-01-01, 09:30 UTC  
SECOND | Datetime truncated to the nearest second | DateTime | 2019-01-01, 09:30:30 UTC  
MILLISECOND | Datetime truncated to the nearest millisecond | DateTime | 2019-01-01, 09:30:30:300 UTC  
###  Numeric options
Option | Description | Type | Displayed value | Notes  
---|---|---|---|---  
DAY_OF_WEEK_INDEX | Index of the day of the week | Number | 0 | The value range and start of the week depends on your warehouse configuration  
DAY_OF_MONTH_NUM | Day of the month | Number | 21  
DAY_OF_YEAR_NUM | Day of the year | Number | 127  
WEEK_NUM | Week number | Number | 37  
MONTH_NUM | Month number | Number | 7  
QUARTER_NUM | Quarter number | Number | 3  
YEAR_NUM | Year number | Number | 2019  
MINUTE_OF_HOUR_NUM | Minute number | Number | 50  
HOUR_OF_DAY_NUM | Hour number | Number | 22  
###  String options
Option | Description | Type | Displayed value  
---|---|---|---  
DAY_OF_WEEK_NAME | Day of the week | String | Monday  
MONTH_NAME | Month name | String | March  
QUARTER_NAME | Quarter name | String | Q3  
##  Groups
You can group your dimensions and metrics in the sidebar using the `groups` parameter. To do this, you need to set up `group_details` in the model’s configuration. Then, you can use these groups to organize metrics and dimensions. You can create nested groups up to 3 levels.
Copy
Ask AI
```
models:
name: baskets
    meta:
      group_details:
        product_details:
          label: Product Details
          description: 'Fields that have information about the products in the basket.'
        item_details:
          label: Item Details
          description: 'Fields that have information about the items in the basket.'
    columns:
name: basket_item_id
        description: 'ID for the product item within the basket.'
        meta:
          dimension:
            groups: ['product_details', 'item_details'] # this would add the dimension to a nested group: `product details` --> `item details`
name: product_name
        description: 'Full name of the product.'
        meta:
          dimension:
            label: 'Product name'
            groups: ['product_details'] # this would add the dimension under the group label: `product_details`

```

This example would look like this in the sidebar:
##  URLs
Lightdash users can interact with dimension values by clicking on them. If you’re already storing URLs in your models, you can create hyperlinks to those URLs in Lightdash, like so:
Copy
Ask AI
```
columns:
name: candidate_profile_url
    label: URL of the candidate profile
    meta:
      dimension:
        urls:
label: Open in CRM
: ${ value.raw }

```

###  How to add custom URLs
By adding custom urls you can configure the actions available to your users. Like linking to external tools, or taking actions in other tools. In the example below, users can click on a company name and open a corresponding record in their CRM or search for the company in google or open that company’s Slack channel.
Copy
Ask AI
```
columns:
name: company_name
    label: Registered trading name of the company
    meta:
      dimension:
        urls:
label: Search for company in Google
: 'https://google.com/search?${ value.formatted | url_encode }'
label: Open in CRM
: 'https://mycrm.com/companies/${ row.company.company_id.raw | url_encode }'

```

The `${ value.formatted }` will be replaced with the value of the company name in the Lightdash UI at query run time. The `${ row.company.company_id.raw }` will be replaced with the value of the company id in the Lightdash UI at query run time. The action will be disabled if the column “company_id” from table “company” is not part of the query.
###  You can reference values from other columns in your URLs
You can reference another dimension from your table in your URL. For these URLs to work, the other column you’ve referenced needs to be included in your results table. For example, say I’ve added a URL to `company_name` and it uses the field `customer_id`:
Copy
Ask AI
```
columns:
name: company_name
    label: Registered trading name of the company
    meta:
      dimension:
        urls:
label: "Open company"
: "https://example.com/company/${row.customers.customer_id.raw | url_encode }"

```

This URL will only work if I have `customer_id` included in my results table.
###  Liquid Templating
Use templates to configure the url values depending on the query, this allows your urls to depend on the results of queries. **Available liquid tags** Tag | Description  
---|---  
`${ value.formatted }` | The exact value of the dimension as seen in the Lightdash UI. For example “$1,427.20”  
`${ value.raw }` | The raw value of the dimension returned from the underlying SQL query. For example “1427.2 “  
`${ row.table_name.column_name.formatted }` | The exact value of the column as seen in the Lightdash UI. For example “$1,427.20”  
`${ row.table_name.column_name.raw }` | The raw value of the dimension returned from the underlying SQL query. For example “1427.2 ”  
**Available liquid filters** Filters can be used to make small transformations of your values: Filter | Description | Example usage  
---|---|---  
url_encode | Encode a string as url safe, for example it replaces spaces with %20 | ${value.formatted | url_encode }  
downcase | Convert string to all lowercase | ${value.formatted | downcase }  
append | Append a string to another | ${value.formatted | append: “.html”}  
There are many more filters available in the Liquid documentation.
##  Required attributes
Lightdash can use `user attributes` to limit some dimensions to some users. In the example below, only users with `is_admin` attribute `true` can use the `salary` dimension on `user` table. Users without access to this dimension will not see it or the custom metrics created from this dimension on the `explore page`.
Copy
Ask AI
```
columns:
name:
    description: User name
salary:
    description: User salary
    meta:
      dimension:
        required_attributes:
          is_admin: "true"

```

If a user without access to this dimension runs a query that contains this dimension, they will get a `Forbidden` error.
###  Current limitations
Lightdash dimensions and custom metrics are protected by this feature, however, it is possible to write custom SQL to bypass this filter, for example:
  * Developers and admins running SQL queries on SQL runner.
  * Custom SQL or subqueries on `table calculations`


Scheduler deliveries will run against the user who created the scheduled delivery, be careful when sharing required attributes with other users.
##  Color
You can predefine colors for your string type dimensions, these colors will be used instead of your default organization colors for the right value when you use a grouped bar chart or a pie chart.
Copy
Ask AI
```
name: status
    description: "{{ doc(\"orders_status\") }}"
    meta:
      dimension:
        colors:
          "placed": "#e6fa0f"
          "completed": "#558B2F"
          "shipped": "#29B6F6"
          "return_pending": "#FF6F00"
          "returned": "#E91E63"

```

We recommend using #HEX colors, other color types like rgba,rgba or color name (eg: orange) are also supported on charts, but they are not yet supported on the chart config.
You can manually override these dimension colors by going into the chart config and manually picking a color for that serie.These colors will also take precedence over the organization color palette.
##  Using special characters or capital letters in your column names
If you use special characters on your column names, you might get errors when using those columns on explore. For example, having a column named `Status` with capital S on a table named `orders` in postgres throws the following error:
Copy
Ask AI
```
column orders.status does not exist

```

To fix this, we can add the quoted column to our `sql` meta tag on dimensions
Copy
Ask AI
```
- name: status
  meta:
    dimension:
      type: string
      sql: '"orders"."Status"' # you can also use '${TABLE}."Status"'

```

This will quote the `Status` columns on the SQL query
Copy
Ask AI
```
SELECT
  "orders".order_id AS "orders_order_id",
  "orders"."Status" AS "orders_status"
FROM "postgres"."jaffle"."orders" AS "orders"

```

##  Additional dimensions
Additional dimensions let you define multiple dimensions off of a single column from your dbt model. This is useful when adding different formatting to a column, comparing or combining columns, parsing JSON columns, or creating persisted groups/buckets based off of a column. A “normal” dimension is a column created in your .sql file in dbt that is written to your data warehouse. An additional dimension is not included in your dbt .sql file, so it’s not written to your data warehouse. When used in Lightdash, it just adds the dimension definition to your SQL query (so it’s “created” at runtime). All dimension configurations are available for additional dimensions. You can also use additional dimensions when defining metrics.
Additional dimensions names need to be unique in the model.
###  Adding different formatting
Copy
Ask AI
```
columns:
name: revenue
    meta:
      dimension:
        type: number
      additional_dimensions:
        revenue_in_thousands:
          type: number
          format: '#,##0," K"'
        revenue_in_millions:
          type: number
          format: '#,##0,," M"'

```

###  Comparing or combining columns
When defining additional dimensions, you can reference other dimensions, even from joined tables (`organizations` is a joined table in the example below).
Copy
Ask AI
```
columns:
name: created_date
    meta:
      dimension:
        type: date
      additional_dimensions:
        days_to_first_query_run:
          type: number
          description: 'Number of days between a user being created and their first query run.'
          sql: ${first_query_date} - ${created_date}
        days_to_organization_first_payment:
          type: number
          description: 'Number of days between a user being created and their organization making its first payment. This will be negative for users who joined after the first payment.'
          sql: ${created_date} - ${organizations.first_payment_date}

```

###  Parsing JSON columns
Usually you’ll want to add `hidden:true` for the main JSON dimension since raw JSON is not useful in charts.
Copy
Ask AI
```
columns:
name: metadata # this is a jsonb column with metadata
    meta:
      dimension:
        hidden: true
      additional_dimensions:
        version:
          type: number
          sql: JSON_VALUE(${metadata}, '$.version') # custom SQL applied to get the "version" value inside metadata

```

###  Adding multiple timezones for the same dimension
You can use additional dimensions to convert a timestamp into multiple timezones:
Copy
Ask AI
```
columns:
name: created_at
    description: 'The time that the thing was created'
    meta:
      dimension:
        label: 'Created (UTC)'
        type: timestamp
      additional_dimensions:
        created_at_est:
          type: timestamp
          label: 'Created (EST)'
          description: 'The time that the thing was created, in EST'
          sql: "convert_timezone('UTC', 'America/New_York', ${TABLE}.created_at)"

```

###  Using additional dimensions in metrics
To define metrics based on additional dimensions, you need to add them to the model’s meta metrics, or use custom SQL in defining them under the column’s meta.
Copy
Ask AI
```
models:
name: users
    meta:
      metrics:
        highest_version_model_metric_example:
          type: max
          sql: ${version}
    columns:
name: metadata
        meta:
          dimension:
            hidden: true
          additional_dimensions:
            version:
              type: number
: JSON_VALUE(${metadata}, '$.version')
          metrics:
            highest_version:
              type: max
: ${version}

```

Lightdash CLI referenceMetrics
Assistant
Responses are generated using AI and may contain mistakes.


