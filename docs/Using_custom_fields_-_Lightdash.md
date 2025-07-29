# Using custom fields - Lightdash

**Source:** https://docs.lightdash.com/references/custom-fields

Lightdash home page
Search...
Ctrl KAsk AI
Search...
Navigation
Reference
Using custom fields
##### Introduction
  * Welcome to Lightdash
  * Setting up a new project


##### Explore and analyze
  * Quickstart
  * Guides
  * Reference
    * Creating Dashboards
    * Filters reference
    * Using custom fields


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
  * Custom metrics
  * Custom metric types
  * Adding filters to your custom metric
  * Custom dimensions
  * When to use custom SQL dimensions
  * When not to use custom SQL dimensions
  * Example: Categorizing order totals


##  Custom metrics
Custom metrics are only saved in the chart they’re used in and will not be saved in the list of `metrics` available if you open the same Table to build a new chart. Metrics written to your dbt project (or YAML metrics) are reusable by everyone in your project and will always show up in the side bar of available metrics for a Table. We recommend writing back custom metrics that you’re using/creating frequently so they become reusable, governed and available for everyone else to build from. To create a custom metric:
  1. Click on the dimension’s three-dot `options` menu
  2. Click on one of the options available (e.g. `Count Distinct`)
  3. Confirm your changes

Then, your new metric will be added to your results table automatically and will appear in the `custom metrics` space in your sidebar. If you want to delete the custom metric, you can just click on the three-dot `options` menu and hit `Remove custom metric`
###  Custom metric types
To learn more about the custom metric types, read the metrics reference documentation here. Only aggregate metric types are available as custom metrics.
###  Adding filters to your custom metric
You can add filters to limit the rows included in your metric aggregation. You can add filters when:
  1. You create a custom metric:


  1. You edit an existing custom metric:


##  Custom dimensions
Dimensions are used to group results in your query, or filter values from your results. Sometimes, a group or filter that you need hasn’t been added to your project, so you can use a custom dimension to create a grouping or filter on-the-fly. These dimensions are not saved to the table permanently. They are only available in the saved chart or query that they were created in.
###  Bin
Bins are used to split out values of a numeric dimension into custom sets of ranges. Bins can only be used with dimensions that are numeric. Fixed bins (fixed number and fixed width) will update automatically if the min or max values of your dimension change. To create a bin custom dimension, you just need to:
  1. Click on a dimension’s three-dot `options` menu
  2. Click on the `bin` option
  3. Setup + create your dimension

Then, your new dimension will be added to your results table automatically and will appear in the `custom dimensions` space in your sidebar. There are three ways that you can set up your bins: **1. Fixed number of bins** You pick the number of bins that you want to group your dimension into and Lightdash automatically defines the ranges for your bins. The ranges that Lightdash generates for the bins are equal to the range of the values in your dimension divided by the number of bins you selected. **2. Fixed width** You pick the width (a.k.a. ranges) for your bins and Lightdash automatically defines a set of bins with those widths. The number of bins that Lightdash generates is equal to the range of values in your dimension divided by the fixed width that you picked for your bins. **3. Custom range** You manually define the min and max values for each bin. The custom range option is the only binning option that will not automatically update if the min or max values of your dimension change (you would need to manually adjust your custom bins or some values could not be included in your bins).
###  Custom SQL
Custom SQL dimensions let you pull in new data directly from your database, which means they can sometimes be used to bypass access restrictions. **Only users with admin or developer permissions can create custom SQL dimensions**. Regular users will not see this option.
**Security note:** Custom SQL dimensions can query any data from your warehouse, so they should only be created by trusted users (admins and developers). For most cases, it’s safer and more reusable to define dimensions in your dbt files. If you want to make a custom SQL dimension reusable, use the dbt write-back feature to move it into your dbt project.
####  When to use custom SQL dimensions
  * **Quickly create new groupings or calculations** that aren’t yet available in your project.
  * **Reference other dimensions** from the main table and joined tables in your explore.
  * **Prototype new logic** before moving it to your dbt project.


####  When _not_ to use custom SQL dimensions
  * If you need the dimension to be **reusable** or **governed** for your whole team.
  * If you want to ensure **consistent access control** and security.


####  Example: Categorizing order totals
In this example, let’s explore the `orders` table and create a custom dimension that categorizes my orders’ totals into three categories: `low`, `medium`, and `high`. You can click on the button to add a custom dimension on the sidebar: Then you can write your SQL query to create your custom dimension:
Copy
Ask AI
```
CASE
  WHEN ${orders.subtotal}  100 THEN 'low'
  WHEN ${orders.subtotal 500 THEN 'medium'
  ELSE 'high'


```

And after you run your query, you will see your new custom dimension categorising your orders’ totals:
Table CalculationsCustom Tooltips
Assistant
Responses are generated using AI and may contain mistakes.


