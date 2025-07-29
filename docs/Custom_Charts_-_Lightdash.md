# Custom Charts - Lightdash

**Source:** https://docs.lightdash.com/references/custom-charts

Lightdash home page
Search...
⌘KAsk AI
Search...
Navigation
Reference
Custom Charts
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
  * Known limitations
  * Vega templates
  * Funnel charts
  * Waterfall charts
  * Trellis area chart
  * More examples


**Custom visualizations are a Beta feature.** This means we have limited documentation and support. We may also change the way these options work in the future.
###  Known limitations
These new Vega-lite powered charts offer enhanced flexibility and a broader range of visualizations, but there are important limitations to be aware of, particularly concerning interactivity and dashboard integration like:
  * **Drill-Down Functionality** : The ability to drill into metrics by grouping them with dimensions (e.g., “Drill by”) is not available in Custom Charts.
  * **View Underlying Data** : Users cannot click on data points within Custom Charts to view the underlying records that compose those data points.
  * **Cross-Filtering** : Interactive filtering across dashboard tiles by selecting elements within a chart (also known as cross-filtering) is not supported in Custom Charts.


##  Quickstart
The steps to create a custom chart are as follows:
Gather your data
Return the data you need for your chart as normal using the Lightdash UI to select relevant dimensions and metrics.
Switch to custom chart
Head to the chart configuration options, in the drop down, you should see an option for custom charts as long as it has been enabled. 
Write your config
Select a template to load a sample configuration for the chart selected, or grab an example. 
**Tip:** LLMs are great at updating Vega Lite configs 
Below are some example chart config templates that we got working in Lightdash, along with some tips on when to use each chart type.
##  Vega templates
###  Bar chart
You can already do this with presets in Lightdash, but it’s a simple example you should be able to get working quickly. This chart works best with simple string or date dimensions and a numeric metric
Here is an example config with tooltips and a pivot
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "mark": "bar",
  "encoding": {
    "x": {
      "field": "orders_order_date_week",
      "type": "temporal" // use quantitative if dimension if string or number

    "y": {
      "field": "orders_total_order_amount",
      "type": "quantitative"

    "color": { // Optional property to pivot the data
      "field": "orders_status",
      "type": "nominal"

    "tooltip": [ // Optional property to show tooltips
"field": "orders_order_date_week", "type": "temporal", "title": "Order Date Week"},
"field": "orders_total_order_amount", "type": "quantitative", "title": "Total Order Amount"},
"field": "orders_status", "type": "nominal", "title": "Status"}




```

###  Heatmaps
Heatmaps can only be created using the Custom Charts feature. This chart works best with string or date dimensions and a numeric metric as color. The config below will output a heatmap with the standard Vega-Lite settings that looks like this:
Example heatmap config
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "mark": "rect",
  "encoding": {
    "x": {
      "field": "orders_order_date_week",
      "type": "temporal"

    "y": {
      "field": "orders_status",
      "type": "nominal"

    "color": {
      "field": "orders_total_order_amount",
      "type": "quantitative",
      "aggregate": "sum",
      "scale": {
        "scheme": "reds"


    "tooltip": [

        "field": "orders_total_order_amount",
        "type": "quantitative",
        "aggregate": "sum"





```

###  Bubble Plots
Bubble plots build on top of standard scatter plot visualizations, by allowing you to adjust the size of a given point based on the output of a field. Here’s one looking at some Healthcare data. This chart works best with string or date dimensions and a numeric metric for the y-axis and another numeric metric for the size. The config below will output a bubble plot with the standard Vega-Lite settings, like this:
Example Bubble blot config
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "mark": "point",
  "encoding": {
    "x": {
      "field": "orders_order_date_week",
      "type": "temporal"

    "y": {
      "field": "orders_total_order_amount",
      "type": "quantitative"

    "size": {
      "field": "customers_unique_customer_count",
      "type": "quantitative"




```

###  Funnel charts
Funnel charts are ideal for visualizing a flow or process where the quantity decreases step-by-step, such as sales pipelines, conversion rates, or order processes. Each stage is represented as a bar whose width reflects the corresponding value. This chart works best when you have a **categorical dimension** (like a status or step name) and a **numeric metric** (such as order amount or count) to measure at each step. This chart works best with a string dimension (like status or step name) and a numeric metric (such as order amount or count) to measure at each step.
**TIP:** If you add extra dimensions, the funnel chart will not render correctly.
This is just a sample of the template. Funnel chart configurations are a bit more complex than other charts, so we recommend loading the template directly in Lightdash instead of copying this code manually. The template will automatically map your data fields into the configuration. The config below will output a funnel chart with the standard Vega-Lite settings that looks like this:
Example funnel chart config
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "config": {
    "view": {
      "strokeWidth": 0


  "transform": [

      "calculate": "datum.orders_total_order_amount + ' ' + datum.orders_status",
      "as": "label"


      "window": [

          "op": "lag",
          "field": "orders_total_order_amount",
          "as": "previous_value"


      "frame": [





      "calculate": "datum.previous_value ? (datum.orders_total_order_amount / datum.previous_value) * 100 : null",
      "as": "percent_of_previous"


      "calculate": "isValid(datum.percent_of_previous) ? '↓ ' + format(datum.percent_of_previous, '.1f') + '%' : 'N/A'",
      "as": "change_label"


  "layer": [

      "mark": {
        "type": "bar",
        "color": "#40817c"

      "encoding": {
        "x": {
          "field": "orders_total_order_amount",
          "type": "quantitative",
          "stack": "center",
          "axis": null

        "y": {
          "field": "orders_status",
          "type": "nominal",
          "axis": null,
          "sort": null,
          "scale": {
            "padding": 0.5


        "color": {
          "field": "orders_status",
          "scale": {
            "range": [
              "#bde4e2",
              "#a2d0ce",
              "#87bcb9",
              "#6ea8a5",
              "#569490",
              "#40817c"






      "mark": {
        "type": "text",
        "color": "white"

      "encoding": {
        "y": {
          "field": "orders_status",
          "type": "nominal",
          "axis": null,
          "sort": null

        "text": {
          "field": "label"




      "mark": {
        "type": "text",
        "color": "black"

      "encoding": {
        "y": {
          "field": "orders_status",
          "type": "nominal",
          "axis": null,
          "sort": null

        "yOffset": {
          "value": -9

        "text": {
          "condition": {
            "test": "datum.change_label !== 'N/A'",
            "field": "change_label"

          "value": ""






```

###  Waterfall charts
Waterfall charts are used to show how an initial value is affected by a series of positive and negative changes over time or across different categories. They are perfect for visualizing how different factors contribute to a total, like revenue changes, customer growth, or budget breakdowns. Each bar represents a step: increases are shown in one color (e.g. green), decreases in another (e.g. red), and the final total in a third (e.g. blue).
  * This chart works best with string or date dimensions for the x-axis (like a year or stage) and two numeric metrics for the y-axis: one representing the starting point and the other representing the ending point. The difference between these two values defines the height and direction (increase or decrease) of each bar.
  * Additionally, you could use a string dimension to color the bars.

The config below will output a waterfall chart with the standard Vega-Lite settings, like this:
Example waterfall config
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
  "encoding": {
    "x": {
      "field": "orders_status",
      "type": "ordinal"

    "y": {
      "field": "orders_total_order_amount",
      "type": "quantitative",
      "axis": {
        "title": "orders_total_order_amount"


    "y2": {
      "field": "customers_unique_customer_count",
      "type": "quantitative"


  "layer": [

      "mark": "bar",
      "encoding": {
        "color": {
          "type": "ordinal",
          "_comment": "chose a field to color by",
          "_field": "type",
          "scale": {
            "domain": [
              "total",
              "increase",
              "decrease"

            "range": [
              "#4FC3F7",
              "#B2FF59",
              "#FF5252"






      "mark": "text",
      "encoding": {
        "y": {
          "field": "orders_total_order_amount",
          "type": "quantitative"

        "text": {
          "field": "orders_total_order_amount",
          "type": "nominal"

        "color": {
          "value": "#1B5E20"






```

###  Map charts
Map charts are perfect for visualizing data that has geographic coordinates, such as customer locations, sales regions, or shipment routes. They combine a background map (usually countries, states, or regions) with your data points overlaid as markers. This chart works best when you have **latitude and longitude fields** (you can use table calculations for that too) to plot the points on the map, and optionally a numeric metric to adjust the size of each point (like total sales or number of customers). In this example, the base world map is drawn using a TopoJSON file of countries, and data points are plotted as circles, with their size based on the total order amount. The config below will output a map chart with the standard Vega-Lite settings, like this:
Example global map config
Copy
Ask AI
```

  "projection": {
    "type": "mercator",
    "scale": 100, // Change scale to zoom into the map
    "center": [




  "layer": [

      "data": {
        "url": "/vega-world-map.json", // Lightdash default world map
        "format": {
          "type": "topojson",
          "feature": "countries"


      "mark": {
        "fill": "lightgray",
        "type": "geoshape",
        "stroke": "white"



      "mark": "circle",
      "encoding": {
        "size": {
          "type": "quantitative",
          "field": "orders_total_order_amount",
          "legend": {
            "title": "Total Order Amount"


        "color": {
          "field": "orders_status",
          "type": "nominal",
          "legend": {
            "title": "Order Status"


        "tooltip": [

            "type": "ordinal",
            "field": "orders_status",
            "title": "Status"


            "type": "quantitative",
            "field": "orders_total_order_amount",
            "title": "Total Order Amount"


        "latitude": {
          "type": "quantitative",
          "field": "latitude"

        "longitude": {
          "type": "quantitative",
          "field": "longitude"






```

To load additional maps, you can specify a full URL on the data layer, and adjust the projection if needed, for example, below is the code for a detailed map of United States, like this:
Example USA map config
Copy
Ask AI
```

  "projection": {
    "type": "albersUsa"

  "layer": [

      "data": {
        "url": "https://vega.github.io/vega-lite/data/us-10m.json",
        "format": {
          "type": "topojson",
          "feature": "counties"


      "mark": {
        "fill": "lightgray",
        "type": "geoshape",
        "stroke": "white"



      "mark": "circle",
      "encoding": {
        "size": {
          "type": "quantitative",
          "field": "orders_total_order_amount",
          "legend": null

        "tooltip": [

            "type": "nominal",
            "field": "orders_total_order_amount"


        "latitude": {
          "type": "quantitative",
          "field": "latitude"

        "longitude": {
          "type": "quantitative",
          "field": "longitude"






```

###  Trellis area chart
Trellis Area charts (also called small multiples or faceted area charts) are used to compare trends across multiple categories over time, with each category shown in its own mini chart. They’re perfect for spotting patterns, outliers, or seasonality within individual segments, like: product lines, user groups, or marketing channels, without crowding everything into a single chart. Each small chart shares the same x- and y-axis structure, making it easy to scan and compare trends side-by-side. This chart works best with a date or time dimension on the x-axis, a numeric metric on the y-axis, and a categorical dimension (like region, product, or browser type) to break the chart into rows or columns. Trellis Area charts are especially helpful when you want to emphasize the shape of trends rather than exact values. The code below will give you a trellis area chart like this (you can also see a live example here).
Example trellis area chart config
Copy
Ask AI
```

  "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
  "description": "Distinct order counts by browser and month.",
  "facet": {
    "row": {
      "field": "dbt_orders_browser",
      "type": "nominal",
      "title": "Browser"


  "spec": {
    "width": 600,
    "height": 60,
    "mark": "area",
    "encoding": {
      "x": {
        "field": "dbt_orders_order_date_week",
        "type": "temporal",
        "title": "Order Month",
        "axis": {
          "grid": true


      "y": {
        "field": "dbt_orders_count_distinct_order_id",
        "type": "quantitative",
        "title": "Distinct Orders",
        "axis": {
          "grid": false


      "color": {
        "field": "dbt_orders_browser",
        "type": "nominal",
        "legend": null

      "tooltip": [

          "field": "dbt_orders_browser",
          "type": "nominal",
          "title": "Browser"


          "field": "dbt_orders_order_date_week",
          "type": "temporal",
          "title": "Order Month"


          "field": "dbt_orders_count_distinct_order_id",
          "type": "quantitative",
          "title": "Distinct Orders"




  "resolve": {
    "scale": {
      "y": "independent"




```

###  More examples
You can find more examples on the Vega lite official website.
Custom TooltipsOverview
Assistant
Responses are generated using AI and may contain mistakes.


