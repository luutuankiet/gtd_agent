# Chart types - Lightdash

**Source:** https://docs.lightdash.com/references/chart-types

Lightdash home page
Search...
⌘KAsk AI
Search...
Navigation
Reference
Chart types
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
  * Chart types and options
  * Columns, Rows, and Metrics
  * Pivot tables (dimensions as columns)
  * Transposing your table (metrics as rows)
  * My Totals are not what I expect
  * Locking columns from scroll
  * Conditional formatting
  * General settings
  * Stacked bar chart
  * 100% Stacked Bar Chart
  * Horizontal bar chart
  * Scatter chart
  * Configure your bar, line and scatter charts
  * Reference lines


In Lightdash, the data in your results tables can be visualized in a bunch of different ways:
  * Horizontal bar chart
  * Scatter chart

The visualization type that you pick determines how Lightdash shows the data series in your chart. To change how your data is displayed, go into the `charts` tab in the Explore view. You have the option to change the chart type shown by selecting a style from the drop-down: You can also adjust all of the configuration settings for your chart type by clicking on the `configure` button: Once you’ve finished creating your chart, you can share it using the URL, save the chart, download it as an image, or save it to a dashboard.
##  Chart types and options
Each chart type has its own configuration options. Click the `configure` button next to the chart type in the `chart` tab to see your options.
###  Big value
The Big value option is for displaying a single value, well, big. The Big value works for any type of value: string, numeric, boolean, you name it! We always display the first value from the field in your results table. The options for Big value include:
  * Selecting which field you want to use for your Big value.
  * Updating the label below the Big value value.
  * Adding a comparison to the previous row


###  Table
The Table option is good for looking at (surprise, surprise) tabular data, or for lists of things like user IDs or transactions. The options for Tables include:
  * Renaming the columns in your table.
  * Showing/hiding the columns in your table.
  * Showing/hiding the table name from the column labels.
  * Showing/hiding the totals for your columns.
  * Pivoting by a column.
  * Transposing your table (a.k.a. pivoting your metrics)
  * Locking a column from scrolling in your table.
  * Adding conditional formatting to your cells.

By default, we attach the table name to your field name (just in case you’ve got any duplicate fields from joined tables). But, you can easily turn this off in your table viz with a toggle.
####  Columns, Rows, and Metrics
Table visualizations have three components:
  * **Rows** : When a field is chosen for the row area, all of the unique values for that field are populated as values in the _rows_ of your table.
  * **Columns** : When a field is chosen for the column area, all of the unique values for that field are populated as values in the _columns_ of your table.
  * **Metrics** : If you have metrics in your table, then each metric cell shows the summarized information for a given row + column combination.


####  Pivot tables (dimensions as columns)
Pivot tables help you more easily summarize larger sets of data in table visualizations. They’re also helpful to identify trends between two dimensions in your data using a table visualization. To add a pivot in your table, move a dimension to the `column` section of your table configuration. This will change the dimension from having its values populate the rows values of your table, to having it populate the column values of your table. You can move up to 3 dimensions as columns and there is a limit of 60 column values in your table (e.g. pivoting by a dimension with 30 months, and another with 4 statuses won’t work because that will generate 4 x 30 = 120 column values).
####  Transposing your table (metrics as rows)
You can transpose your table so that your metrics become rows in your table instead of columns. This can be helpful if you want to compare a lot of metrics to each other (e.g. how my metrics are changing over time). To move your metrics to rows in your table, you need to have at least 1 dimension as a column (i.e. at least one pivot) then, you can click on `show metrics as rows` in the table configuration.
####  Totals
You can add column totals or row totals (in pivot tables) to your tables by selecting `Show column totals` or `Show row totals` in the chart configuration panel. The column totals in your results and table visualizations are calculated using the underlying data from your table, not only the values that are visible in the table.
Totals are not calculated for table calculations. Also, in pivot tables, totals are not shown for `count distinct`, `average`, `max`, `min`, and `median` metric types.
####  My Totals are not what I expect
**Why are my totals lower?** When using the `count_distinct` metric type, you can sometimes get totals that are smaller than if you sum up the values seen in the table. For example, if you count the distinct number of devices that viewed pages on our website each month, it would look something like this: If you manually add each row in the `Anonymous device count` column, the value you get is much higher than the total shown in the table. This is because the same device can view pages on our website across many months. So, when you add up the values in the table, you’ll be counting some devices more than once. Lightdash uses a SQL query to calculate the distinct number of devices across all of the months so we avoid double-counting devices. **Why are my totals higher?** There are two reasons why this could be happening:
  1. You’ve set a row limit in your query that’s truncating the results. If the number of possible results from your query is larger than the row limit you’ve set, Lightdash will calculate the totals using all of the results (including the rows that have been removed from your table because of the limit).
  2. You’re using metric or table calculation filters. When you use metric or table calculation filters, the totals are calculated before the filters are applied.

**How do I calculate totals based on what’s shown in my table?** If you want to calculate totals based on just the values shown in your table, you can create a new column using a table calculation to do this. Here’s the table calculation you’ll need to use to do this:
Copy
Ask AI
```
SUM(${my_table_name.my_metric_name}) OVER()

```

This calculation isn’t a “true” total when you’re using metrics types that are `count_distinct`!
####  Subtotals
You can add subtotals to your tables by selecting `Show subtotals` in the chart configuration panel. To use subtotals, you need to have at least 2 or more dimensions in your table visualization.
####  Locking columns from scroll
If you have a very wide table, you may want some columns to always be locked to the left while you’re scrolling. In Lightdash, you can just click on the lock icon beside the columns you want to keep pinned to the left of your table visualization. Now, if someone is looking at your table, they’ll always see those columns, locked in view.
####  Conditional formatting
Sometimes it’s helpful to highlight certain values in your tables when they meet a specific condition. You can set up conditional formatting rules by going to the Configure tab then clicking on Conditional Formatting. When you add a new rule, you’ll first need to pick which column should be affected, the type of rule you’d like to use (`single color` or `color range`), then you’ll need to choose condition you want to match and the colour to change the cell if it matches that condition. You can set as many rules on a table as you want. If two or more rules disagree with each other, the rule that’s on the bottom of your list of rules will win. To use color ranges for your rules, select `color range` as the type. The color range option highlights cells with color shades according to the value in a particular cell. By default, the color range will use a set of 5 colors mapped across the min and max colors selected in your rule.
###  Pie chart
Pie charts can be useful visualize data that adds up to a meaningful “whole”. For example, the split of total revenue across each of our product lines. To use a pie chart type, you need at least one metric and one dimension in your query results.
####  Layout
To use a pie chart type, you need at least one metric and one dimension in your query results. You can add multiple dimensions to create more granular `groups` in your pie charts. By default, pie charts are shown in a donut shape (i.e. with the center removed) but you can easily toggle the `donut` toggle to switch between a traditional pie and donut chart.
####  Series
You can adjust the labels and colors for each of the slices of your pie chart in the `series` tab. You also have the option to show or hide value labels in your chart.
####  Display
You can choose to show or hide the legend for your pie chart in the `display` tab.
###  Funnel chart
Funnel charts work well for showing pipeline performance, product conversion metrics, onboarding flows, or any other process where things move from one discrete stage to another. To use a funnel chart, you should have discrete stages of a process and numeric counts of entities in each stage. The funnel will show the count at each stage and the percentage that number represents of the starting count, making it easy to visualize the proportion moving from one stage to the next.
####  General settings
With the Lightdash funnel chart, you can use either a row or a column to describe the stages in your funnel. Use the `Data orientation` switch to specify which orientation you would like to use. When `rows` is selected, each row of your data represent one step. The first column in the data will be treated as the labels for the steps, and you can choose which **numeric** column to treat as the value of the step. When `columns` is selected, you should have a numeric column for each step you want to display. Only the first row of data will be represented and the column names will be used as the step labels. Non-numeric columns will be ignored.
###  Bar chart
Bar charts are helpful to:
  * compare things between different groups (e.g. the number of orders you have by product type)
  * track how a number changes over time if you have a _smaller number of x-axis values_ (e.g. number of new users per month over a year).


####  Stacked bar chart
You can also stack bar charts to compare proportions across different groups. Stacked bar charts work best when:
  * _**the focus of the chart is to compare the totals and one part of the totals.**_ It’s hard to compare bars if they don’t start at the same baseline. So, if you’re trying to build a chart to compare multiple parts of your total with each other, consider keeping your bar chart unstacked!
  * _**you’re trying to show the parts of multiple totals**_. If you only want to show parts of one total, consider an unstacked bar chart instead. If you only want to communicate one part of one total, consider if you should be using a big value chart instead.

Check out more details about bar chart configurations here.
####  100% Stacked Bar Chart
You can create a 100% stacked bar chart by setting up a table calculation for percent of group total. Use your x-axis dimension as the `column_i_want_to_group_by`, then in your chart configuration choose your x-axis, group dimension, and the new table calculation as your y-axis. 100% Stacked Bar Charts are useful when you need to visualize how the composition of a metric changes over time, or how the composition changes across different groups.
###  Area chart
Area charts are best if:
  * **you want to show how values develop over time.** If you want to show how values differ in different categories, consider a (stacked) bar, or horizontal bar chart instead.
  * **the total is as important as its parts**. If the total (= the height of all your stacked areas) is not important, consider a line chart instead. Many readers will have an easier time understanding a line chart than an area chart.
  * **there are big differences between your values**. If the differences between your values are very small, consider a line chart instead. Compared to an area chart, the y-axis of a line chart doesn’t need to start at zero. This means that your y-axis can be stretched to show the tiny differences.
  * **you’re showing multiple series over time**. If you just want to show one value over time, also consider a line chart instead; especially if you don’t want your y-axis to start at zero. If you only have a few dates, you can also consider using a column chart. In both cases, labelling will be better.
  * **you have many data points**. If you have less than ten or so data points, consider a stacked bar chart instead.


###  Line chart
Line charts are used to show changes in a number over a short or long period of time. They’re a good option when:
  * **you have small changes between your values**. The y-axis of a line chart doesn’t need to start at zero. This means that your y-axis can be stretched to show the tiny differences.
  * **you have lots of x-axis values**. In this case, line charts are better to use than bar charts.

Line charts with multiple lines can also be used to compare changes over the same period of time for more than one group. You can see more details about line chart configurations here.
###  Horizontal bar chart
Horizontal bar charts are just bar charts, except the columns are placed on the chart horizontally instead of vertically. Horizontal bar charts are useful when:
  * you’re trying to group a number by something with a lot of possible values.
  * your groups have really long label names.

You can see more details about horizontal bar chart configurations here.
###  Scatter chart
A scatter chart is useful if you want to to look at the relationship, a.k.a. correlation, between two variables. Something like the age of your users vs. the amount of time they’ve spent on your website. You can group your scatter chart using a third variable. This will make the points on the scatter the same colour if they have the same group value. You can see more details about scatter chart configurations here.
###  Mixed chart
You can combine bars, line, and scatter charts on the same chart using a Mixed chart. To use a Mixed chart, you’ll need to start with either a line, scatter or horizontal bar chart type and have two or more series on your chart. Either from having two or more fields selected for your y-axis or from having a group with two or more groups. Once you have the series you want on your chart, you can pick and choose the different chart types you’d like for each series in the `series` tab of the `Configure` space. You can easily revert all of the series on your chart to a single type using the `chart type` toggle list in the `series` tab. You can see more details about mixed chart configurations here.
##  Configure your bar, line and scatter charts
These chart types have very similar configuration options, so we thought it would be easiest to talk about them all together:
###  Layout
This is where you can pick the columns from your results table that you want to plot on your x and y axes or that you want to group by. You can group by up to 3 fields in your chart. For bar charts, this is also where you have the option to stack your bars, or leave them unstacked (grouped).
###  Series
The series tab is where you can adjust how your chart shows each data series. A **data series** is a set of related data points plotted on a chart. For example, the number of new users created each day over a set of dates is a series. In a bar chart, a series is represented by bars of the same color; in a line chart, a series is represented by a single line. You can see a list of the series for your chart in the `series` menu, and on the chart legend. The options available in here will depend on the data in your chart. But, in here you can:
  * Set the chart type of your series.
  * Put your series on a left or right y-axis.
  * Show the value labels on data points.
  * Show or hide a series from your chart.

If you have multiple series in your chart, you can adjust these settings for _**all**_ of the series in your chart at once using the configuration options at the top. Or, you can adjust these setting for each of the series individually.
###  Axes
Here’s where you can customize the axes on your plot. You have the options to:
  * change the text for your axes labels.
  * set the axis limits for your y-axes. You can either leave the auto-axis range or manually enter your own limits.


###  Display
This tab is where you’re able to control the legend and reference lines in your chart.
####  Reference lines
You can add reference lines to your charts to easily visualize goals or thresholds. To add a reference line, you just need to click `+ Add` under the reference lines option, then configure your reference line in the drop-down. You can:
  * select a field you want to apply your reference line to
  * select the value you want to set your line at
  * customize the label on your reference line
  * change the colour of your reference line

If you select a field plotted on your x-axis, then your line will be vertical. If you select a field plotted on your y-axis, then your line will be horizontal.
If your reference line label is cropped off your chart, try adjusting your right margin.
####  Legend
You can add and adjust the legend in your chart to help people understand what data’s been plotted.
  * Show/hide the legend
  * Adjust the position of the legend on your chart 
    * The values in `position` are the coordinates of the legend on your chart. They can either be numbers or percent. We suggest using %.
    * If you want the position to be in the bottom right, you would set: Right = 0%, Bottom = 0%. For the legend to be in the middle of the chart, you’d set: Top = 50%, Left = 50%.
  * Set your legend to scroll (if you have so many groups that they overlap your chart).
  * Orientate the values in your legened to form a list vertically, or horizontally


###  Margins
The Margin tab is where you’re able to add or remove margin around the grid (aka your chart with its plotted values). Removing margin means that your chart will fill more of the space in the chart tab. Adding margin will shrink your chart into a smaller space. You can either add numbers (e.g. 50) or percent values (e.g. 50%) to the margin settings listed. The default margin is set to `top` = 70, `bottom` = 30, `left` = 5%, `right` = 5%
Creating DashboardsFilters reference
Assistant
Responses are generated using AI and may contain mistakes.


